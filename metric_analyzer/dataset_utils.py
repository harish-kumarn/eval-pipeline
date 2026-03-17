"""Dataset utility functions for the metric analyzer."""

import pandas as pd


def extract_retrieval_dataset(categories: dict) -> pd.DataFrame:
    """Return a DataFrame of retrieval metrics for all queries."""
    rows = []
    for _cat, queries in categories.items():
        for q in queries:
            row = {"query_id": q.query_id or q.retrieval_metrics.get("qid", "")}
            row.update(q.retrieval_metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def extract_generation_dataset(categories: dict) -> pd.DataFrame:
    """Return a DataFrame of generation metrics for all queries."""
    rows = []
    for _cat, queries in categories.items():
        for q in queries:
            row = {"query_id": q.query_id or q.generation_metrics.get("qid", "")}
            row.update(q.generation_metrics)
            rows.append(row)
    return pd.DataFrame(rows)