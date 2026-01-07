import json
from typing import List, Dict

from pathlib import Path
import json


def load_metrics(path: str):
    path = Path(path)

    if not path.is_absolute():
        base = Path(__file__).resolve().parent.parent
        path = base / path

    with open(path) as f:
        data = json.load(f)

    # If wrapped, return per-query metrics
    if isinstance(data, dict) and "per_query_metrics" in data:
        return data["per_query_metrics"]
        
    return data if isinstance(data, list) else data.get("metrics", [])



def export_json(data: Dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
