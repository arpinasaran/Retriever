import os
import pickle
import contextlib
import heapq
import time
import math
import re
from bisect import bisect_left

# NLP Imports
import Stemmer
from stop_words import get_stop_words

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings
from tqdm import tqdm

class BSBIIndex:
    """Inverted index builder and retriever using BSBI (Block Sort-Based Indexing).

    Attributes
    ----------
    term_id_map : IdMap
        Bidirectional mapping between term strings and integer term IDs.
    doc_id_map : IdMap
        Bidirectional mapping between document paths and integer doc IDs.
    data_dir : str
        Path to the document collection directory.
    output_dir : str
        Directory where index files are written.
    postings_encoding : class
        Encoding class used for postings (e.g. VBEPostings, StandardPostings).
    index_name : str
        Base filename for the merged index.
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Names of all intermediate index files produced during indexing
        self.intermediate_indices = []

        # NLP pipeline: English stemmer and stopword list
        self.stemmer = Stemmer.Stemmer('english')
        self.stop_words = set(get_stop_words('english'))

    def save(self):
        """Persist term_id_map and doc_id_map to the output directory via pickle."""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Load term_id_map and doc_id_map from the output directory."""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """Parse all text files in a block directory into (termID, docID) pairs.

        Applies tokenization, stopword removal, and English stemming to each
        document. One subdirectory in the collection is treated as one block.

        Parameters
        ----------
        block_dir_relative : str
            Relative path to the subdirectory representing one block.

        Returns
        -------
        List[Tuple[int, int]]
            All (termID, docID) pairs extracted from the block.
            Uses self.term_id_map and self.doc_id_map, which persist across calls.
        """
        dir_path = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []

        for filename in next(os.walk(dir_path))[2]:
            # Normalize path separators for cross-platform consistency
            docname = os.path.join(dir_path, filename).replace("\\", "/")
            doc_id = self.doc_id_map[docname]

            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                text = f.read()

                # Tokenize: keep only alphanumeric tokens, lowercase
                tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())

                for token in tokens:
                    # Stopword removal
                    if token not in self.stop_words:
                        # English stemming
                        stemmed_term = self.stemmer.stemWord(token)

                        term_id = self.term_id_map[stemmed_term]
                        td_pairs.append((term_id, doc_id))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """Invert (termID, docID) pairs and write the result to an index file.

        Builds a term dictionary (hashtable) from the pairs and writes each
        term's sorted postings list together with per-document term frequencies.

        Assumes td_pairs fit entirely in memory.

        Parameters
        ----------
        td_pairs : List[Tuple[int, int]]
            List of (termID, docID) pairs for a single block.
        index : InvertedIndexWriter
            The intermediate index file to write to.
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index):
        """Merge all intermediate inverted indices into a single final index.

        Performs an external merge sort over the intermediate indices using a
        min-heap. Postings lists for the same term are merged and their term
        frequencies accumulated.

        Parameters
        ----------
        indices : List[InvertedIndexReader]
            Iterable intermediate index readers, one per block.
        merged_index : InvertedIndexWriter
            Writer for the final merged index.
        """
        # Assumes at least one term exists across all intermediate indices
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k = 10):
        """Retrieve the top-K documents using TF-IDF scoring (Term-at-a-Time).

        Scoring formula:
            w(t, D) = 1 + log(tf(t, D))   if tf(t, D) > 0, else 0
            w(t, Q) = IDF = log(N / df(t))
            score(D) = sum over query terms of w(t, Q) * w(t, D)

        Document length normalization is not applied.

        Parameters
        ----------
        query : str
            Space-separated query string.
        k : int
            Number of top documents to return.

        Returns
        -------
        List[Tuple[float, str]]
            Top-K (score, document_path) pairs sorted by descending score.
            Terms absent from the collection are silently ignored.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        # Apply the same NLP pipeline used during indexing
        query_tokens = re.findall(r'\b[a-z0-9]+\b', query.lower())
        processed_query = [
            self.stemmer.stemWord(token)
            for token in query_tokens
            if token not in self.stop_words
        ]

        # Map terms to IDs; skip terms not in the vocabulary
        terms = [self.term_id_map[word] for word in processed_query if word in self.term_id_map]

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            # Return top-K results sorted by descending score
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25(self, query, k = 10, k1 = 1.2, b = 0.75):
        """Retrieve the top-K documents using BM25 scoring.

        Scoring formula per query term t:
            IDF(t) * (tf(t,D) * (k1 + 1)) / (tf(t,D) + k1 * (1 - b + b * dl/avdl))

        where:
            IDF(t)  = log(N / df(t))
            dl      = document length (total token count)
            avdl    = average document length across the collection

        Parameters
        ----------
        query : str
            Query string.
        k : int
            Number of top documents to return.
        k1 : float
            Term frequency saturation parameter (default 1.2).
        b : float
            Document length normalization parameter (default 0.75).

        Returns
        -------
        List[Tuple[float, str]]
            Top-K (score, document_path) pairs sorted by descending score.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        # Apply the same NLP pipeline used during indexing
        query_tokens = re.findall(r'\b[a-z0-9]+\b', query.lower())
        processed_query = [
            self.stemmer.stemWord(token)
            for token in query_tokens
            if token not in self.stop_words
        ]

        # Map terms to IDs; skip terms not in the vocabulary
        terms = [self.term_id_map[word] for word in processed_query if word in self.term_id_map]

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            N = len(merged_index.doc_length)
            avdl = sum(merged_index.doc_length.values()) / N

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    idf = math.log(N / df)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        dl = merged_index.doc_length[doc_id]
                        numerator = tf * (k1 + 1)
                        denominator = tf + k1 * (1 - b + b * dl / avdl)
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        scores[doc_id] += idf * numerator / denominator

            # Return top-K results sorted by descending score
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25_wand(self, query, k=10, k1=1.2, b=0.75):
        """Retrieve the top-K documents using BM25 scoring with WAND optimization.

        Instead of scoring every document, WAND skips documents whose
        upper-bound contribution cannot beat the current top-K threshold.
        Uses the precomputed per-term upper bounds from indexing.

        Parameters
        ----------
        query : str
            Query string.
        k : int
            Number of top documents to return.
        k1 : float
            BM25 term frequency saturation parameter (default 1.2).
        b : float
            BM25 document length normalization parameter (default 0.75).

        Returns
        -------
        List[(float, str)]
            Top-K documents sorted by descending BM25 score.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        # Load precomputed upper bounds
        ub_path = os.path.join(self.output_dir, self.index_name + '.ub')
        with open(ub_path, 'rb') as f:
            upper_bounds = pickle.load(f)

        # Process query with the same NLP pipeline
        query_tokens = re.findall(r'\b[a-z0-9]+\b', query.lower())
        processed_query = [
            self.stemmer.stemWord(token)
            for token in query_tokens
            if token not in self.stop_words
        ]
        terms = [self.term_id_map[word] for word in processed_query if word in self.term_id_map]

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            avdl = sum(merged_index.doc_length.values()) / N

            # Load postings lists and metadata for all query terms
            term_data = []  # list of dicts: term_id, postings, tf_list, ub, cursor, df
            for term in terms:
                if term in merged_index.postings_dict:
                    postings, tf_list = merged_index.get_postings_list(term)
                    ub = upper_bounds.get(term, 0.0)
                    term_data.append({
                        'term_id': term,
                        'postings': postings,
                        'tf_list': tf_list,
                        'ub': ub,
                        'cursor': 0,
                        'df': merged_index.postings_dict[term][1],
                    })

            if not term_data:
                return []

            # Sentinel: a doc ID larger than any real doc ID
            LAST_ID = float('inf')

            def current_did(td):
                """Return the doc ID at the current cursor, or LAST_ID if exhausted."""
                if td['cursor'] < len(td['postings']):
                    return td['postings'][td['cursor']]
                return LAST_ID

            def advance_to(td, target):
                """Advance cursor to first doc ID >= target using binary search."""
                postings = td['postings']
                pos = bisect_left(postings, target, td['cursor'])
                td['cursor'] = pos

            def full_eval(doc_id):
                """Compute exact BM25 score for a document across all query terms."""
                score = 0.0
                dl = merged_index.doc_length[doc_id]
                for td in term_data:
                    c = td['cursor']
                    postings = td['postings']
                    pos = bisect_left(postings, doc_id, c)
                    if pos < len(postings) and postings[pos] == doc_id:
                        tf = td['tf_list'][pos]
                        idf = math.log(N / td['df'])
                        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avdl))
                return score

            # Top-K min-heap: stores (score, doc_id)
            top_k_heap = []
            threshold = 0.0
            cur_doc = -1

            # WAND main loop — implements next(θ) from the pseudocode
            while True:
                # Step A: Sort terms by current DID (non-decreasing)
                term_data.sort(key=lambda td: current_did(td))

                # Step B: Find pivot — first term where accumulated UB >= threshold
                acc_ub = 0.0
                p_term_idx = None
                for i, td in enumerate(term_data):
                    if current_did(td) == LAST_ID:
                        break
                    acc_ub += td['ub']
                    if acc_ub >= threshold:
                        p_term_idx = i
                        break

                # No pivot found — no more candidates can beat the threshold
                if p_term_idx is None:
                    break

                pivot = current_did(term_data[p_term_idx])
                if pivot == LAST_ID:
                    break

                if pivot <= cur_doc:
                    # Pivot already evaluated; advance a preceding term past cur_doc
                    aterm_idx = min(range(p_term_idx + 1),
                                   key=lambda i: len(term_data[i]['postings']))
                    advance_to(term_data[aterm_idx], cur_doc + 1)
                else:
                    if current_did(term_data[0]) == pivot:
                        # All preceding terms align on pivot — fully evaluate it
                        cur_doc = pivot
                        score = full_eval(cur_doc)

                        if len(top_k_heap) < k:
                            heapq.heappush(top_k_heap, (score, cur_doc))
                            if len(top_k_heap) == k:
                                threshold = top_k_heap[0][0]
                        elif score > threshold:
                            heapq.heapreplace(top_k_heap, (score, cur_doc))
                            threshold = top_k_heap[0][0]
                    else:
                        # Preceding terms not aligned; advance one toward pivot
                        aterm_idx = min(range(p_term_idx + 1),
                                       key=lambda i: len(term_data[i]['postings']))
                        advance_to(term_data[aterm_idx], pivot)

            # Convert heap to sorted results
            results = [(score, self.doc_id_map[doc_id]) for (score, doc_id) in top_k_heap]
            return sorted(results, key=lambda x: x[0], reverse=True)

    def _precompute_upper_bounds(self, k1=1.2, b=0.75):
        """Precompute per-term BM25 upper bound scores and write them to disk.

        For each term t:
            UB_t = IDF(t) * max_d [ tf*(k1+1) / (tf + k1*(1-b+b*dl/avdl)) ]

        Used by WAND retrieval to skip documents that cannot beat the current
        top-K score threshold.
        """
        ub_path = os.path.join(self.output_dir, self.index_name + '.ub')
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as reader:
            N = len(reader.doc_length)
            avdl = sum(reader.doc_length.values()) / N
            upper_bounds = {}

            for term_id in reader.postings_dict:
                df = reader.postings_dict[term_id][1]
                idf = math.log(N / df)
                postings, tf_list = reader.get_postings_list(term_id)

                max_score = 0.0
                for i in range(len(postings)):
                    tf = tf_list[i]
                    dl = reader.doc_length[postings[i]]
                    score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avdl))
                    if score > max_score:
                        max_score = score
                upper_bounds[term_id] = max_score

        with open(ub_path, 'wb') as f:
            pickle.dump(upper_bounds, f)

    def index(self):
        """Build the inverted index using BSBI (Block Sort-Based Indexing).

        Scans each subdirectory in the collection as one block, parses its
        documents into (termID, docID) pairs, inverts and writes an intermediate
        index per block, then merges all intermediate indices into the final index.
        """
        # Process each subdirectory (block) in sorted order
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)

        self._precompute_upper_bounds()


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index()
