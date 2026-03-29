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
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

        # Inisialisasi tools untuk NLP (Stemming & Stopwords)
        self.stemmer = Stemmer.Stemmer('english')
        self.stop_words = set(get_stop_words('english'))

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        dir_path = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        
        for filename in next(os.walk(dir_path))[2]:
            # Normalisasi path agar konsisten
            docname = os.path.join(dir_path, filename).replace("\\", "/")
            doc_id = self.doc_id_map[docname]
            
            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                text = f.read()
                
                # Tokenization: Ambil hanya alphanumeric dan jadikan lowercase
                tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
                
                for token in tokens:
                    # Stopword Removal
                    if token not in self.stop_words:
                        # Stemming Bahasa Inggris
                        stemmed_term = self.stemmer.stemWord(token)
                        
                        term_id = self.term_id_map[stemmed_term]
                        td_pairs.append((term_id, doc_id))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
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
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
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
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        # Proses query menggunakan pipeline NLP yang sama dengan dokumen
        query_tokens = re.findall(r'\b[a-z0-9]+\b', query.lower())
        processed_query = [
            self.stemmer.stemWord(token) 
            for token in query_tokens 
            if token not in self.stop_words
        ]

        # Konversi term menjadi term_id, abaikan jika term tidak ada di collection
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

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25(self, query, k = 10, k1 = 1.2, b = 0.75):
        """
        Melakukan Ranked Retrieval dengan skema scoring BM25.
        Method akan mengembalikan top-K retrieval results.

        Score BM25 = untuk setiap term t di query, akumulasikan:
            IDF(t) * (tf(t,D) * (k1 + 1)) / (tf(t,D) + k1 * (1 - b + b * dl/avdl))

        dimana:
            IDF(t)  = log(N / df(t))
            tf(t,D) = term frequency of t in document D
            dl      = panjang dokumen D (jumlah token)
            avdl    = rata-rata panjang dokumen di seluruh collection

        Parameters
        ----------
        query: str
            Query string
        k: int
            Jumlah dokumen yang dikembalikan
        k1: float
            Parameter BM25 untuk term frequency saturation (default 1.2)
        b: float
            Parameter BM25 untuk document length normalization (default 0.75)

        Result
        ------
        List[(float, str)]
            List of tuple: elemen pertama adalah score, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        # Proses query menggunakan pipeline NLP yang sama dengan dokumen
        query_tokens = re.findall(r'\b[a-z0-9]+\b', query.lower())
        processed_query = [
            self.stemmer.stemWord(token)
            for token in query_tokens
            if token not in self.stop_words
        ]

        # Konversi term menjadi term_id, abaikan jika term tidak ada di collection
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

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25_wand(self, query, k=10, k1=1.2, b=0.75):
        """Ranked Retrieval using BM25 scoring with WAND top-K optimization.

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
            term_data = []  # list of (term_id, postings, tf_list, ub, cursor)
            for term in terms:
                if term in merged_index.postings_dict:
                    postings, tf_list = merged_index.get_postings_list(term)
                    ub = upper_bounds.get(term, 0.0)
                    # cursor is index into postings list
                    term_data.append({
                        'term_id': term,
                        'postings': postings,
                        'tf_list': tf_list,
                        'ub': ub,
                        'cursor': 0,  # points to current position
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
                    # Check if this term is present in doc_id
                    postings = td['postings']
                    # Find doc_id starting from cursor (it might be at cursor or later)
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

                # Step B: Find pivot term — first term where accumulated UB >= threshold
                acc_ub = 0.0
                p_term_idx = None
                for i, td in enumerate(term_data):
                    if current_did(td) == LAST_ID:
                        break
                    acc_ub += td['ub']
                    if acc_ub >= threshold:
                        p_term_idx = i
                        break

                # No pivot found — no more candidates can beat threshold
                if p_term_idx is None:
                    break

                pivot = current_did(term_data[p_term_idx])
                if pivot == LAST_ID:
                    break

                if pivot <= cur_doc:
                    # Pivot already considered, advance a preceding term past cur_doc
                    # pickTerm: choose term with smallest postings list (cheapest to advance)
                    aterm_idx = min(range(p_term_idx + 1),
                                   key=lambda i: len(term_data[i]['postings']))
                    advance_to(term_data[aterm_idx], cur_doc + 1)
                else:
                    # pivot > cur_doc
                    if current_did(term_data[0]) == pivot:
                        # Success: all preceding terms point to pivot doc
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
                        # Not enough mass, advance a preceding term to pivot
                        aterm_idx = min(range(p_term_idx + 1),
                                       key=lambda i: len(term_data[i]['postings']))
                        advance_to(term_data[aterm_idx], pivot)

            # Convert heap to sorted results
            results = [(score, self.doc_id_map[doc_id]) for (score, doc_id) in top_k_heap]
            return sorted(results, key=lambda x: x[0], reverse=True)

    def _precompute_upper_bounds(self, k1=1.2, b=0.75):
        """Precompute per-term BM25 upper bound scores after indexing.

        For each term t, UB_t = IDF(t) * max_d [ tf*(k1+1) / (tf + k1*(1-b+b*dl/avdl)) ]
        These are stored to disk so WAND retrieval can load them at query time.
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
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
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
    BSBI_instance.index() # memulai indexing!
