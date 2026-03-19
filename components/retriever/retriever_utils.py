import json
import re
from typing import List, Dict, Optional


def _parse_json(text: str) -> Optional[List[Dict]]:
    """Safely parse JSON text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def _extract_docs_from_stream(raw_chunks: List[str]) -> Optional[List[Dict]]:
    """Extract JSON from combined stream as fallback."""
    combined = "".join(raw_chunks)

    if "__RETRIEVED_DOCS__" in combined:
        raw = combined.split("__RETRIEVED_DOCS__", 1)[1]
        parsed = _parse_json(raw)
        if parsed:
            return parsed

    match = re.search(r"(\[\s*\{.*?\}\s*\])", combined, re.DOTALL)
    if match:
        parsed = _parse_json(match.group(1))
        if parsed:
            return parsed

    return None