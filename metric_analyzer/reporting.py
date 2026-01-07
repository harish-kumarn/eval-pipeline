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
                "category": cat.value,
                "retrieval_score": q.retrieval_score,
                "generation_score": q.generation_score,
                "retrieval_pattern": q.retrieval_pattern.pattern_type if q.retrieval_pattern else "N/A",
                "retrieval_severity": q.retrieval_pattern.severity if q.retrieval_pattern else "N/A",
                "generation_pattern": q.generation_pattern.pattern_type if q.generation_pattern else "N/A",
                "generation_severity": q.generation_pattern.severity if q.generation_pattern else "N/A",
                "production_risk": q.generation_pattern.production_risk if q.generation_pattern else "N/A",
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
                "category": cat.value,
                "retrieval_score": q.retrieval_score,
                "generation_score": q.generation_score,
                "retrieval_metrics": q.retrieval_metrics,
                "generation_metrics": q.generation_metrics,
            }
            if q.retrieval_pattern:
                item["retrieval_pattern"] = {
                    "pattern_type": q.retrieval_pattern.pattern_type,
                    "diagnosis": q.retrieval_pattern.diagnosis,
                    "root_cause": q.retrieval_pattern.root_cause,
                    "action": q.retrieval_pattern.action,
                    "severity": q.retrieval_pattern.severity
                }
            if q.generation_pattern:
                item["generation_pattern"] = {
                    "pattern_type": q.generation_pattern.pattern_type,
                    "diagnosis": q.generation_pattern.diagnosis,
                    "inference": q.generation_pattern.inference,
                    "severity": q.generation_pattern.severity,
                    "production_risk": q.generation_pattern.production_risk
                }
            export_data.append(item)

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"✅ Analysis exported to {output_path}")

def export_to_csv(categories: Dict[PerformanceCategory, List[QueryAnalysis]], output_path: str):
    """
    Export combined retrieval + generation metrics to CSV.
    Includes: query_id, query, category, retrieval_score, generation_score, 
    and all fields from retrieval_metrics and generation_metrics.
    """
    rows = []
    for cat, queries in categories.items():
        for q in queries:
            # Start with core fields
            row = {
                "query_id": q.query_id,
                "query": q.query,
                "category": cat.value,
                "retrieval_score": q.retrieval_score,
                "generation_score": q.generation_score,
            }
            
            # Use prefixes for raw metrics to avoid collisions if any
            # and to clearly separate them.
            # However, the user said "all the rwtriver metrcis , generator metrics"
            # I will just merge them. If there are collisions, the latter wins.
            # To be safe and clean, I'll add them one by one.
            
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
    # Ensure query_id and query are the first columns
    cols = ["query_id", "category", "retrieval_score", "generation_score"]
    other_cols = [c for c in df.columns if c not in cols]
    df = df[cols + other_cols]
    
    df.to_csv(output_path, index=False)
    print(f"✅ CSV exported to {output_path}")