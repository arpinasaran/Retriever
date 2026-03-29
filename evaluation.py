import re
import math
import time
from bsbi import BSBIIndex
from spimi import SPIMIIndex
from compression import VBEPostings


def rbp(ranking, p=0.8):
    """Compute Rank Biased Precision (RBP).

    Models a user who reads rank-by-rank with probability p of continuing
    to the next result. Higher p = more patient user.

    Parameters
    ----------
    ranking : List[int]
        Binary relevance vector, e.g. [1, 0, 1, 1].
        ranking[i] == 1 means the document at rank i+1 is relevant.
    p : float
        Persistence parameter (default 0.8).

    Returns
    -------
    float
        RBP score in [0, 1].
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (p ** (i - 1))
    return (1 - p) * score


def dcg(ranking):
    """Compute Discounted Cumulative Gain (DCG).

    Measures the total gain collected by a user who reads results top-down,
    discounting gains at lower ranks by log2(rank + 1).

    Parameters
    ----------
    ranking : List[int]
        Binary relevance vector.

    Returns
    -------
    float
        DCG score.
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (1 / math.log2(i + 1))
    return score


def ndcg(ranking):
    """Compute Normalized Discounted Cumulative Gain (NDCG).

    Normalizes DCG by the ideal DCG (all relevant documents ranked first),
    yielding a score in [0, 1].

    Parameters
    ----------
    ranking : List[int]
        Binary relevance vector.

    Returns
    -------
    float
        NDCG score in [0, 1].
    """
    relevant_count = sum(ranking)
    ideal_ranking = [1] * relevant_count + [0] * (len(ranking) - relevant_count)
    ideal_dcg = dcg(ideal_ranking)

    this_dcg = dcg(ranking)
    score = this_dcg / ideal_dcg if ideal_dcg else 0

    return score


def precision(ranking):
    """Compute Precision.

    The fraction of returned documents that are relevant.

    Parameters
    ----------
    ranking : List[int]
        Binary relevance vector.

    Returns
    -------
    float
        Precision score in [0, 1].
    """
    score = sum(ranking) / len(ranking) if len(ranking) > 0 else 0
    return score


def ap(ranking):
    """Compute Average Precision (AP).

    The mean of precision values computed at each rank position where a
    relevant document appears.

    Parameters
    ----------
    ranking : List[int]
        Binary relevance vector.

    Returns
    -------
    float
        AP score in [0, 1].
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += precision(ranking[:i]) * ranking[pos]

    return score / sum(ranking) if sum(ranking) > 0 else 0


def load_qrels(qrel_file="qrels.txt", max_q_id=30, max_doc_id=1033):
    """Load relevance judgments (qrels) from a file.

    Returns a nested dict: qrels[query_id][doc_id] = 1 if relevant, 0 otherwise.

    Parameters
    ----------
    qrel_file : str
        Path to the qrels file (default "qrels.txt").
    max_q_id : int
        Number of queries (IDs 1 .. max_q_id).
    max_doc_id : int
        Number of documents (IDs 1 .. max_doc_id).

    Returns
    -------
    dict[str, dict[int, int]]
        qrels["Q3"][12] == 1 means doc 12 is relevant to query Q3.
    """
    qrels = {"Q" + str(i): {i: 0 for i in range(1, max_doc_id + 1)}
             for i in range(1, max_q_id + 1)}
    with open(qrel_file) as file:
        for line in file:
            parts = line.strip().split()
            qid = parts[0]
            did = int(parts[1])
            qrels[qid][did] = 1
    return qrels


def eval(qrels, retrieve_fn, label, query_file="queries.txt", k=1000):
    """Evaluate a retrieval function over all queries and report mean metrics.

    Runs retrieve_fn on each of the 30 queries (top-k=1000), builds a binary
    relevance vector from qrels, then computes and prints mean RBP, DCG,
    NDCG, Precision, and AP. Also reports total retrieval time.

    Parameters
    ----------
    qrels : dict
        Relevance judgments from load_qrels().
    retrieve_fn : callable
        Retrieval function with signature (query: str, k: int) -> List[(score, doc_path)].
    label : str
        Display label shown in the printed output (e.g. "BM25 (BSBI)").
    query_file : str
        Path to the queries file (default "queries.txt").
    k : int
        Number of documents to retrieve per query (default 1000).
    """
    with open(query_file) as file:
        rbp_scores = []
        dcg_scores = []
        ndcg_scores = []
        precision_scores = []
        ap_scores = []

        total_time = 0.0
        for qline in file:
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])

            # Note: doc IDs assigned during indexing may differ from the IDs
            # in qrels, so the numeric ID is extracted from the file path.
            t0 = time.time()
            results = retrieve_fn(query, k=k)
            t1 = time.time()
            total_time += (t1 - t0)

            ranking = []
            for (score, doc) in results:
                did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                ranking.append(qrels[qid][did])
            rbp_scores.append(rbp(ranking))
            dcg_scores.append(dcg(ranking))
            ndcg_scores.append(ndcg(ranking))
            precision_scores.append(precision(ranking))
            ap_scores.append(ap(ranking))

    def mean(scores):
        return sum(scores) / len(scores) if len(scores) > 0 else 0

    print(f"Evaluation results for {label} over 30 queries")
    print("RBP score       =", mean(rbp_scores))
    print("DCG score       =", mean(dcg_scores))
    print("NDCG score      =", mean(ndcg_scores))
    print("Precision score =", mean(precision_scores))
    print("AP score        =", mean(ap_scores))
    print(f"Total retrieval time = {total_time:.4f} seconds")


if __name__ == '__main__':
    qrels = load_qrels()

    assert qrels["Q1"][166] == 1, "qrels salah"
    assert qrels["Q1"][300] == 0, "qrels salah"

    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')

    SPIMI_instance = SPIMIIndex(data_dir='collection',
                                postings_encoding=VBEPostings,
                                output_dir='index_spimi')

    print("======== BSBI ========")
    eval(qrels, BSBI_instance.retrieve_tfidf, "TF-IDF (BSBI)")
    print()
    eval(qrels, BSBI_instance.retrieve_bm25, "BM25 (BSBI)")
    print()
    eval(qrels, BSBI_instance.retrieve_bm25_wand, "BM25 + WAND (BSBI)")

    print()
    print("======== SPIMI ========")
    eval(qrels, SPIMI_instance.retrieve_tfidf, "TF-IDF (SPIMI)")
    print()
    eval(qrels, SPIMI_instance.retrieve_bm25, "BM25 (SPIMI)")
    print()
    eval(qrels, SPIMI_instance.retrieve_bm25_wand, "BM25 + WAND (SPIMI)")
