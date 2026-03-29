import os
import contextlib
import re

from tqdm import tqdm

from bsbi import BSBIIndex
from index import InvertedIndexReader, InvertedIndexWriter
from compression import VBEPostings


class SPIMIIndex(BSBIIndex):
    """
    Inverted index builder using SPIMI (Single-Pass In-Memory Indexing).

    Unlike BSBI, SPIMI does not sort (termID, docID) pairs. Instead it builds
    postings lists directly in a hashtable in memory. When the number of
    accumulated postings exceeds max_postings, the in-memory index is flushed
    to disk as an intermediate index. After all documents are processed, all
    intermediate indices are merged into the final index.

    Block boundaries are driven by a memory threshold, not directory structure.

    Parameters
    ----------
    data_dir : str
        Path to the document collection.
    output_dir : str
        Directory where the final and intermediate index files are written.
    postings_encoding : class
        Encoding class for postings (e.g. VBEPostings, StandardPostings).
    index_name : str
        Base name for the merged index files.
    max_postings : int
        Number of unique (term, doc) pairs to accumulate in memory before
        flushing to disk. Acts as a simple memory-usage proxy.
    """

    def __init__(self, data_dir, output_dir, postings_encoding,
                 index_name="main_index", max_postings=100_000):
        super().__init__(data_dir, output_dir, postings_encoding, index_name)
        self.max_postings = max_postings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush(self, in_memory_index, block_num):
        """Write the in-memory index to disk as one intermediate index file.

        Parameters
        ----------
        in_memory_index : dict[int, dict[int, int]]
            Mapping term_id -> {doc_id -> tf}.
        block_num : int
            Sequential block counter used to name the intermediate file.
        """
        index_id = f"intermediate_index_spimi_{block_num}"
        self.intermediate_indices.append(index_id)
        with InvertedIndexWriter(index_id, self.postings_encoding,
                                 directory=self.output_dir) as index:
            for term_id in sorted(in_memory_index.keys()):
                doc_tf = in_memory_index[term_id]
                sorted_doc_ids = sorted(doc_tf.keys())
                tf_list = [doc_tf[d] for d in sorted_doc_ids]
                index.append(term_id, sorted_doc_ids, tf_list)

    # ------------------------------------------------------------------
    # SPIMI indexing
    # ------------------------------------------------------------------

    def index(self):
        """Build an inverted index using SPIMI.

        Iterates over every document in the collection regardless of
        subdirectory boundaries. Postings are accumulated in a hashtable
        {term_id: {doc_id: tf}}. When the number of unique (term, doc) pairs
        reaches max_postings, the block is flushed to disk. After all
        documents are processed any remaining postings are flushed, then all
        intermediate indices are merged into the final index.
        """
        in_memory_index = {}   # term_id -> {doc_id -> tf}
        postings_count = 0     # number of unique (term, doc) pairs in memory
        block_num = 0

        # Collect all document paths upfront for the progress bar.
        # Prefix with "./" to match the path format used by BSBIIndex so that
        # downstream consumers (evaluation regex, qrels lookup) work correctly.
        all_docs = []
        for dirpath, _, filenames in os.walk(self.data_dir):
            for filename in filenames:
                path = os.path.join(dirpath, filename).replace("\\", "/")
                if not path.startswith("./"):
                    path = "./" + path
                all_docs.append(path)

        for docpath in tqdm(all_docs, desc="SPIMI indexing"):
            doc_id = self.doc_id_map[docpath]

            with open(docpath, "r", encoding="utf8", errors="surrogateescape") as f:
                text = f.read()

            tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())

            for token in tokens:
                if token in self.stop_words:
                    continue
                stemmed = self.stemmer.stemWord(token)
                term_id = self.term_id_map[stemmed]

                if term_id not in in_memory_index:
                    in_memory_index[term_id] = {}

                if doc_id not in in_memory_index[term_id]:
                    in_memory_index[term_id][doc_id] = 0
                    postings_count += 1  # new (term, doc) pair

                in_memory_index[term_id][doc_id] += 1

            # Flush when memory threshold is exceeded
            if postings_count >= self.max_postings:
                self._flush(in_memory_index, block_num)
                in_memory_index = {}
                postings_count = 0
                block_num += 1

        # Flush any remaining postings
        if in_memory_index:
            self._flush(in_memory_index, block_num)

        self.save()

        # Merge all intermediate indices into the final index
        with InvertedIndexWriter(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [
                    stack.enter_context(
                        InvertedIndexReader(idx_id, self.postings_encoding,
                                            directory=self.output_dir)
                    )
                    for idx_id in self.intermediate_indices
                ]
                self.merge(indices, merged_index)

        self._precompute_upper_bounds()


if __name__ == "__main__":
    SPIMI_instance = SPIMIIndex(
        data_dir="collection",
        output_dir="index_spimi",
        postings_encoding=VBEPostings,
    )
    SPIMI_instance.index()
