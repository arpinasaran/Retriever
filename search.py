from bsbi import BSBIIndex
from spimi import SPIMIIndex
from compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex / SPIMIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='collection',
                          postings_encoding=VBEPostings,
                          output_dir='index')

SPIMI_instance = SPIMIIndex(data_dir='collection',
                            postings_encoding=VBEPostings,
                            output_dir='index_spimi')

queries = ["alkylated with radioactive iodoacetate",
           "psychodrama for disturbed children",
           "lipid metabolism in toxemia and normal pregnancy"]

for query in queries:
    print("Query  : ", query)

    print("=== BSBI ===")
    print("Results TF-IDF:")
    for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=10):
        print(f"{doc:30} {score:>.3f}")
    print("Results BM25:")
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k=10):
        print(f"{doc:30} {score:>.3f}")
    print("Results BM25 + WAND:")
    for (score, doc) in BSBI_instance.retrieve_bm25_wand(query, k=10):
        print(f"{doc:30} {score:>.3f}")

    print("=== SPIMI ===")
    print("Results TF-IDF:")
    for (score, doc) in SPIMI_instance.retrieve_tfidf(query, k=10):
        print(f"{doc:30} {score:>.3f}")
    print("Results BM25:")
    for (score, doc) in SPIMI_instance.retrieve_bm25(query, k=10):
        print(f"{doc:30} {score:>.3f}")
    print("Results BM25 + WAND:")
    for (score, doc) in SPIMI_instance.retrieve_bm25_wand(query, k=10):
        print(f"{doc:30} {score:>.3f}")

    print()