import pickle
import os

class InvertedIndex:
    """Base class for reading and writing an inverted index stored on disk.

    Attributes
    ----------
    postings_dict : dict
        Maps termID -> (start_byte, num_postings, postings_bytes, tf_bytes).

        - start_byte          : byte offset in the index file where the postings list begins.
        - num_postings        : number of doc IDs in the postings list (document frequency).
        - postings_bytes      : byte length of the encoded postings list.
        - tf_bytes            : byte length of the encoded term-frequency list.

        Assumed to fit entirely in memory.

    terms : List[int]
        Ordered list of term IDs in the sequence they were appended to the index.
        Used to iterate through the index in insertion order.

    doc_length : dict
        Maps doc ID (int) -> document length (number of tokens).
        Used for document length normalization in TF-IDF and BM25 scoring.
    """
    def __init__(self, index_name, postings_encoding, directory=''):
        """
        Parameters
        ----------
        index_name : str
            Base name for the index files (without extension).
        postings_encoding : class
            Encoding class for postings (e.g. VBEPostings, StandardPostings).
        directory : str
            Directory where the index files reside.
        """

        self.index_file_path = os.path.join(directory, index_name+'.index')
        self.metadata_file_path = os.path.join(directory, index_name+'.dict')

        self.postings_encoding = postings_encoding
        self.directory = directory

        self.postings_dict = {}
        self.terms = []
        self.doc_length = {}

    def __enter__(self):
        """Load index metadata when entering a context manager.

        Reads postings_dict, terms (as an iterator), and doc_length from the
        metadata pickle file, then opens the binary index file for reading.
        """
        # Open the binary index file
        self.index_file = open(self.index_file_path, 'rb+')

        # Load postings dict, term order, and doc lengths from the metadata file
        with open(self.metadata_file_path, 'rb') as f:
            self.postings_dict, self.terms, self.doc_length = pickle.load(f)
            self.term_iter = self.terms.__iter__()

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Flush metadata to disk and close the index file on context exit."""
        self.index_file.close()

        with open(self.metadata_file_path, 'wb') as f:
            pickle.dump([self.postings_dict, self.terms, self.doc_length], f)


class InvertedIndexReader(InvertedIndex):
    """Sequential and random-access reader for an on-disk inverted index."""

    def __iter__(self):
        return self

    def reset(self):
        """Seek the index file back to the beginning and reset the term iterator."""
        self.index_file.seek(0)
        self.term_iter = self.terms.__iter__()

    def __next__(self):
        """Return the next (termID, postings_list, tf_list) triple.

        Reads only the postings for the current term — the full index is never
        loaded into memory at once.

        Returns
        -------
        Tuple[int, List[int], List[int]]
            (termID, sorted doc IDs, corresponding term frequencies)
        """
        curr_term = next(self.term_iter)
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = self.postings_dict[curr_term]
        postings_list = self.postings_encoding.decode(self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(len_in_bytes_of_tf))
        return (curr_term, postings_list, tf_list)

    def get_postings_list(self, term):
        """Random-access read of the postings list for a single term.

        Seeks directly to the term's byte offset — does not scan the file
        from the beginning.

        Parameters
        ----------
        term : int
            Term ID to look up.

        Returns
        -------
        Tuple[List[int], List[int]]
            (sorted doc IDs, corresponding term frequencies)
        """
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = self.postings_dict[term]
        self.index_file.seek(pos)
        postings_list = self.postings_encoding.decode(self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(len_in_bytes_of_tf))
        return (postings_list, tf_list)


class InvertedIndexWriter(InvertedIndex):
    """Sequential writer for building an on-disk inverted index."""

    def __enter__(self):
        self.index_file = open(self.index_file_path, 'wb+')
        return self

    def append(self, term, postings_list, tf_list):
        """Append a term's postings list and TF list to the index file.

        Steps performed:
        1. Encode postings_list with self.postings_encoding.
        2. Encode tf_list with self.postings_encoding.
        3. Record metadata in self.terms, self.postings_dict, and self.doc_length.
        4. Write the encoded bytes to the end of the index file.

        postings_dict entry format:
            termID -> (start_byte, num_postings, postings_bytes, tf_bytes)

        Parameters
        ----------
        term : int
            Term ID being appended.
        postings_list : List[int]
            Sorted list of doc IDs where the term appears.
        tf_list : List[int]
            Term frequency for each doc ID in postings_list.
        """
        self.terms.append(term)

        # Accumulate token counts into doc_length
        for i in range(len(postings_list)):
            doc_id, freq = postings_list[i], tf_list[i]
            if doc_id not in self.doc_length:
                self.doc_length[doc_id] = 0
            self.doc_length[doc_id] += freq

        self.index_file.seek(0, os.SEEK_END)
        curr_position_in_byte = self.index_file.tell()
        compressed_postings = self.postings_encoding.encode(postings_list)
        compressed_tf_list = self.postings_encoding.encode_tf(tf_list)
        self.index_file.write(compressed_postings)
        self.index_file.write(compressed_tf_list)
        self.postings_dict[term] = (curr_position_in_byte, len(postings_list), \
                                    len(compressed_postings), len(compressed_tf_list))


if __name__ == "__main__":

    from compression import VBEPostings

    with InvertedIndexWriter('test', postings_encoding=VBEPostings, directory='./tmp/') as index:
        index.append(1, [2, 3, 4, 8, 10], [2, 4, 2, 3, 30])
        index.append(2, [3, 4, 5], [34, 23, 56])
        index.index_file.seek(0)
        assert index.terms == [1,2], "terms salah"
        assert index.doc_length == {2:2, 3:38, 4:25, 5:56, 8:3, 10:30}, "doc_length salah"
        assert index.postings_dict == {1: (0, \
                                           5, \
                                           len(VBEPostings.encode([2,3,4,8,10])), \
                                           len(VBEPostings.encode_tf([2,4,2,3,30]))),
                                       2: (len(VBEPostings.encode([2,3,4,8,10])) + len(VBEPostings.encode_tf([2,4,2,3,30])), \
                                           3, \
                                           len(VBEPostings.encode([3,4,5])), \
                                           len(VBEPostings.encode_tf([34,23,56])))}, "postings dictionary salah"

        index.index_file.seek(index.postings_dict[2][0])
        assert VBEPostings.decode(index.index_file.read(len(VBEPostings.encode([3,4,5])))) == [3,4,5], "terdapat kesalahan"
        assert VBEPostings.decode_tf(index.index_file.read(len(VBEPostings.encode_tf([34,23,56])))) == [34,23,56], "terdapat kesalahan"
