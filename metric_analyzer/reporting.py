import pandas as pd
import json
from typing import Dict, List
from metric_analyzer.models import QueryAnalysis, PerformanceCategory

def to_dataframe(categories: Dict[PerformanceCategory, List[QueryAnalysis]]) -> pd.DataFrame:
    """
    Standard summary dataframe with key fields and patterns.
    """
    rows = []
    for cat, queries in categories.items():
        for q in queries:
            rows.append({
                "query_id": q.query_id,
                "query": q.query,
                "tag": q.tag,
                "category": cat.value,
                "retrieval_score": q.retrieval_score,
                "generation_score": q.generation_score,
                "retrieval_status": q.retrieval_status,
                "hit_type": q.hit_type if q.hit_type else "N/A",
                "outcome": q.outcome,
            })
    return pd.DataFrame(rows)

def export_json(categories: Dict[PerformanceCategory, List[QueryAnalysis]], output_path: str):
    """
    Export full query-level analysis to JSON.
    """
    export_data = []

    for cat, queries in categories.items():
        for q in queries:
            item = {
                "query_id": q.query_id,
                "query": q.query,
                "tag": q.tag,
                "category": cat.value,
                "retrieval_score": q.retrieval_score,
                "generation_score": q.generation_score,
                "retrieval_status": q.retrieval_status,
                "hit_type": q.hit_type,
                "outcome": q.outcome,
                "retrieval_metrics": q.retrieval_metrics,
                "generation_metrics": q.generation_metrics,
            }
            export_data.append(item)

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"✅ Analysis exported to {output_path}")

def export_to_csv(categories: Dict[PerformanceCategory, List[QueryAnalysis]], output_path: str):
    """
    Export combined retrieval + generation metrics to CSV.
    """
    rows = []
    for cat, queries in categories.items():
        for q in queries:
            # Start with core fields
            row = {
                "query_id": q.query_id,
                "query": q.query,
                "tag": q.tag,
                "category": cat.value,
                "retrieval_score": q.retrieval_score,
                "generation_score": q.generation_score,
                "retrieval_status": q.retrieval_status,
                "hit_type": q.hit_type if q.hit_type else "N/A",
                "outcome": q.outcome,
            }
            
            if q.retrieval_metrics:
                for k, v in q.retrieval_metrics.items():
                    if k in ["qid", "query_id"]:
                        continue
                    row[f"ret_{k}"] = v
            
            if q.generation_metrics:
                for k, v in q.generation_metrics.items():
                    if k in ["qid", "query_id"]:
                        continue
                    row[f"gen_{k}"] = v
            
            rows.append(row)

    df = pd.DataFrame(rows)
    # Ensure key columns are first
    key_cols = ["query_id", "query", "tag", "category", "retrieval_status", "hit_type", "outcome", "retrieval_score", "generation_score"]
    # Reorder if they exist
    existing_key_cols = [c for c in key_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_key_cols]
    df = df[existing_key_cols + other_cols]
    
    df.to_csv(output_path, index=False)
    print(f"✅ CSV exported to {output_path}")