import pandas as pd
def extract_retrieval_dataset(categories: dict) -> pd.DataFrame:
    """
    Extract retrieval metrics for all queries.
    """
    rows = []
    for cat, queries in categories.items():
        for q in queries:
            row = {"query_id": q.query_id or q.retrieval_metrics.get("qid", "")}
            row.update(q.retrieval_metrics)
            rows.append(row)
    return pd.DataFrame(rows)

def extract_generation_dataset(categories: dict) -> pd.DataFrame:
    """
    Extract generation metrics for all queries.
    """
    rows = []
    for cat, queries in categories.items():
        for q in queries:
            row = {"query_id": q.query_id or q.generation_metrics.get("qid", "")}
            row.update(q.generation_metrics)
            rows.append(row)
    return pd.DataFrame(rows)

def combine_metrics(categories: dict) -> pd.DataFrame:
    """
    Combine retrieval and generation metrics into a single DataFrame,
    including category, retrieval_score, and generation_score.
    """
    retrieval_df = extract_retrieval_dataset(categories)
    generation_df = extract_generation_dataset(categories)

    # Merge on query_id
    combined_df = pd.merge(retrieval_df, generation_df, on="query_id", suffixes=("_retrieval", "_generation"))

    # Add query-level info and category
    rows = []
    for cat, queries in categories.items():
        for q in queries:
            row = combined_df.loc[combined_df["query_id"] == q.query_id].iloc[0].to_dict()
            row["query"] = q.query
            row["category"] = cat.value
            row["retrieval_score"] = q.retrieval_score
            row["generation_score"] = q.generation_score
            rows.append(row)

    return pd.DataFrame(rows)