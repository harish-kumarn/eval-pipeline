"""
retriever_utils.py
------------------
Utility helpers for parsing VisionRAG streaming responses.

Public API:
  - parse_json(text)              → safely deserialise a JSON string.
  - extract_docs_from_stream(...) → recover retrieved-doc list from raw stream chunks.
"""

import json
import re
from typing import Dict, List, Optional

__all__ = ["parse_json", "extract_docs_from_stream"]


def parse_json(text: str) -> Optional[List[Dict]]:
    """Safely parse a JSON string, returning None on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_docs_from_stream(raw_chunks: List[str]) -> Optional[List[Dict]]:
    """
    Recover retrieved-doc JSON from a combined stream as a fallback strategy.

    Tries two approaches in order:
      1. Split on the ``__RETRIEVED_DOCS__`` sentinel marker.
      2. Regex-match the first JSON array literal found in the stream.
    """
    combined = "".join(raw_chunks)

    if "__RETRIEVED_DOCS__" in combined:
        raw = combined.split("__RETRIEVED_DOCS__", 1)[1]
        parsed = parse_json(raw)
        if parsed is not None:
            return parsed

    match = re.search(r"(\[\s*\{.*?\}\s*\])", combined, re.DOTALL)
    if match:
        parsed = parse_json(match.group(1))
        if parsed is not None:
            return parsed

    return None


# ---------------------------------------------------------------------------
# Backward-compatible private aliases (kept so existing imports don't break)
# ---------------------------------------------------------------------------
_parse_json = parse_json
_extract_docs_from_stream = extract_docs_from_stream