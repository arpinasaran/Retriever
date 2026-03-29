# Search Engine from Scratch

A full-featured information retrieval system built from the ground up in Python. The engine supports two indexing algorithms (BSBI and SPIMI), multiple scoring models (TF-IDF, BM25, BM25 + WAND), two postings compression schemes (VBE and Elias-Gamma), and a suite of evaluation metrics (RBP, DCG, NDCG, Precision, AP).

## Features

| Feature | Details |
|---------|---------|
| **Indexing** | BSBI (Block Sort-Based) and SPIMI (Single-Pass In-Memory) |
| **Scoring** | TF-IDF, BM25, BM25 + WAND top-K optimization |
| **Compression** | Variable-Byte Encoding (VBE), Elias-Gamma coding |
| **NLP pipeline** | Tokenization, English stopword removal, Porter stemming |
| **Evaluation** | RBP, DCG, NDCG, Precision, Average Precision (AP) |

## Project Structure

```
.
├── bsbi.py          # BSBI indexer + TF-IDF / BM25 / WAND retrieval
├── spimi.py         # SPIMI indexer (inherits retrieval from BSBIIndex)
├── index.py         # Low-level inverted index reader / writer
├── compression.py   # StandardPostings, VBEPostings, EliasGammaPostings
├── util.py          # IdMap (string ↔ int mapping), postings merge helper
├── search.py        # Sample retrieval script (3 queries, BSBI + SPIMI)
├── evaluation.py    # Batch evaluation over 30 queries
├── qrels.txt        # Relevance judgments (query ID → relevant doc IDs)
├── queries.txt      # 30 evaluation queries
├── collection/      # Document collection (11 subdirectories, 1033 docs)
├── index/           # BSBI index output (generated, not committed)
└── index_spimi/     # SPIMI index output (generated, not committed)
```

## Setup

**Requirements:** Python 3.10+

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## How to Run

The three scripts must be run in order the first time. After indexing, search and evaluation can be re-run freely.

### Step 1 — Build the BSBI index

```bash
python bsbi.py
```

Processes each subdirectory in `collection/` as one block, writes intermediate indices to `index/`, and merges them into the final index. Also precomputes BM25 upper bounds for WAND.

### Step 2 (optional) — Build the SPIMI index

```bash
python spimi.py
```

Builds a second index in `index_spimi/` using Single-Pass In-Memory Indexing. Iterates over all documents in one pass, accumulating postings in a hashtable and flushing to disk when a memory threshold is reached. Produces an identical final index to BSBI.

### Step 3 — Run sample queries

```bash
python search.py
```

Runs three sample queries against both the BSBI and SPIMI indices and prints the top-10 results for each scoring method.

### Step 4 — Evaluate retrieval quality

```bash
python evaluation.py
```

Runs all 30 queries from `queries.txt` against both indices (top-1000 results per query) and reports mean RBP, DCG, NDCG, Precision, and AP scores along with total retrieval time.

## Indexing Algorithms

### BSBI (Block Sort-Based Indexing)
Documents are processed subdirectory-by-subdirectory. Each block produces a sorted list of `(termID, docID)` pairs, which is inverted and written as an intermediate index. All intermediate indices are then merged via an external merge sort.

### SPIMI (Single-Pass In-Memory Indexing)
Documents are processed one-by-one across the entire collection, regardless of subdirectory boundaries. Postings are accumulated directly in a hashtable `{termID: {docID: tf}}` — no sorting step is needed. When the number of accumulated postings reaches the memory threshold (default: 100,000 pairs), the in-memory block is flushed to disk. All flushed blocks are merged at the end.

**Key difference:** SPIMI avoids sorting `(termID, docID)` pairs entirely; postings lists grow incrementally in memory. Block boundaries are driven by memory usage rather than directory structure.

## Scoring Models

### TF-IDF
```
w(t, D) = 1 + log(tf(t, D))    if tf > 0, else 0
w(t, Q) = log(N / df(t))
score(D) = Σ w(t, Q) × w(t, D)
```

### BM25
```
score(D) = Σ IDF(t) × [tf(t,D) × (k1+1)] / [tf(t,D) + k1 × (1 - b + b × dl/avdl)]
```
Default parameters: `k1 = 1.2`, `b = 0.75`.

### BM25 + WAND
Uses precomputed per-term BM25 upper bounds to skip documents that cannot improve the current top-K threshold. Produces identical results to BM25 but evaluates far fewer documents.

## Compression Schemes

| Class | Method | Notes |
|-------|--------|-------|
| `StandardPostings` | Raw 4-byte integers | No compression; fastest encode/decode |
| `VBEPostings` | Variable-Byte Encoding + gap encoding | Good compression, byte-aligned |
| `EliasGammaPostings` | Elias-Gamma coding + gap encoding | Bit-level; better ratio for small gaps |

All three expose the same interface: `encode`, `decode`, `encode_tf`, `decode_tf`.
