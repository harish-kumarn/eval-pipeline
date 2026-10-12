# Retrieval Failure Analysis Report

Total queries: 3

## 1. Overall funnel

| Stage | Queries with answer present | Cumulative recall | Stage-over-stage loss |
|---|---|---|---|
| Total queries | 3 | 100.0% | — |
| After BM25 (lexical retrieval) | 3 | 100.0% | 0 lost |
| After Vector (semantic retrieval) | 3 | 100.0% | 0 lost |
| After Pre-Rerank (BM25 ∪ Vector union) | 3 | 100.0% | 0 lost |
| After Post-Rerank (reranked candidate list) | 3 | 100.0% | 0 lost |
| After Final Top-k (truncation) | 3 | 100.0% | 0 lost |

## 2. Breakdown by query type

| Tag | Total | Successful | Recall |
|---|---|---|---|
| direct | 3 | 3 | 100.0% |

## 3. Failure point breakdown

| Failure point | Count | % of total |
|---|---|---|
| Success | 3 | 100.0% |
