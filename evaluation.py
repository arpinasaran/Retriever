import re
import math
from bsbi import BSBIIndex
from compression import VBEPostings

######## >>>>> sebuah IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan 
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score

def dcg(ranking):
  """ menghitung search effectiveness metric score dengan 
      Discounted Cumulative Gain (DCG)

      Interpretasi: expected total volume of relevance
      (expected total gain) yang dikumpulkan seorang user. 

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score DCG
  """
  score = 0.
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    score += ranking[pos] * (1 / math.log2(i + 1))
  return score

def ndcg(ranking):
  """ menghitung search effectiveness metric score dengan 
      Normalized Discounted Cumulative Gain (NDCG)

      Interpretasi: DCG yang dinormalisasi dengan DCG ideal 
      (DCG dari ranking yang sempurna)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score NDCG
  """
  relevant_count = sum(ranking)
  ideal_ranking = [1] * relevant_count + [0] * (len(ranking) - relevant_count)
  ideal_dcg = dcg(ideal_ranking)

  this_dcg = dcg(ranking)
  score = this_dcg / ideal_dcg if ideal_dcg else 0

  return score

def precision(ranking):
  """ menghitung search effectiveness metric score dengan 
      Precision

      Interpretasi: proporsi dokumen yang relevan sama
      di antara semua dokumen yang dikembalikan

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score Precision
  """
  score = sum(ranking) / len(ranking) if len(ranking) > 0 else 0
  
  return score

def ap(ranking):
  """ menghitung search effectiveness metric score dengan 
      Average Precision (AP)

      Interpretasi: rata-rata dari precision pada setiap 
      posisi relevan

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score AP
  """
  score = 0.
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    score += precision(ranking[:i]) * ranking[pos]

  return score / sum(ranking) if sum(ranking) > 0 else 0


######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels) 
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

######## >>>>> EVALUASI !

def eval(qrels, retrieve_fn, label, query_file = "queries.txt", k = 1000):
  """
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents

    Parameters
    ----------
    qrels: dict
        Relevance judgments
    retrieve_fn: callable
        Fungsi retrieval yang menerima (query, k) dan mengembalikan list of (score, doc)
    label: str
        Label untuk ditampilkan di output (misal "TF-IDF" atau "BM25")
  """
  with open(query_file) as file:
    rbp_scores = []
    dcg_scores = []
    ndcg_scores = []
    precision_scores = []
    ap_scores = []

    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
      # yang tertera di qrels
      ranking = []
      for (score, doc) in retrieve_fn(query, k = k):
          did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
          ranking.append(qrels[qid][did])
      rbp_scores.append(rbp(ranking))
      dcg_scores.append(dcg(ranking))
      ndcg_scores.append(ndcg(ranking))
      precision_scores.append(precision(ranking))
      ap_scores.append(ap(ranking))

  def mean(scores):
    return sum(scores) / len(scores) if len(scores) > 0 else 0

  print(f"Hasil evaluasi {label} terhadap 30 queries")
  print("RBP score =", mean(rbp_scores))
  print("DCG score =", mean(dcg_scores))
  print("NDCG score =", mean(ndcg_scores))
  print("Precision score =", mean(precision_scores))
  print("AP score =", mean(ap_scores))

if __name__ == '__main__':
  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

  eval(qrels, BSBI_instance.retrieve_tfidf, "TF-IDF")
  print()
  eval(qrels, BSBI_instance.retrieve_bm25, "BM25")