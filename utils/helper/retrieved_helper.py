"""
utils/helper/retrieved_helper.py

Helper functions for evaluating retriever performance against a ground-truth
question/answer dataset.

Intended usage (import, not CLI):

    from utils.helper.retrieved_helper import run_evaluation

    evaluated_df = run_evaluation(
        dataset_path="retriver_res/merged_testing_dataset.json",
        results_path="retriver_res/04_08_dummy1_18_54.csv",
        output_path="retriver_res/04_08_dummy1_18_54_with_answers.csv",
    )

Or use the individual building blocks (load_dataset, load_results,
build_question_lookup, evaluate, ...) directly if you already have the
dataset/dataframe loaded elsewhere in your pipeline.
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Data model
# ---------------------------------------------------------
@dataclass(frozen=True)
class QuestionMeta:
    """Ground-truth metadata associated with a single question."""

    answer_chunk_id: list[str]
    tag: Any
    question_type: str


# Desired trailing column order for the columns this module adds.
ADDED_COLUMNS_ORDER: list[str] = [
    "question_type",
    "tag",
    "answer_chunk_id",
    "predicted",
    "retrieved_position",
]


# ---------------------------------------------------------
# Loading
# ---------------------------------------------------------
def load_dataset(path: Union[str, Path]) -> dict[str, Any]:
    """Load and lightly validate the merged ground-truth JSON dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset file is not valid JSON: {path}") from exc

    if not isinstance(data, (dict, list)):
        raise ValueError(
            f"Dataset must be a JSON object or array at the top level, got {type(data)}"
        )

    return data


def load_results(path: Union[str, Path]) -> pd.DataFrame:
    """Load the retriever results CSV and validate required columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError as exc:
        raise ValueError(f"Results file is not valid CSV: {path}") from exc

    required_columns = {"query", "final_retrieved_chunks"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Results CSV is missing required column(s): {sorted(missing)}"
        )

    return df


# ---------------------------------------------------------
# Lookup construction
# ---------------------------------------------------------
def build_question_lookup(data: Union[dict, list]) -> dict[str, QuestionMeta]:
    """
    Build a mapping of question text -> QuestionMeta from the merged dataset.

    Skips (with a warning) any question entry that is missing required fields,
    rather than raising, so one malformed record doesn't abort the whole run.
    """
    lookup: dict[str, QuestionMeta] = {}
    chunks_iter = data.values() if isinstance(data, dict) else data

    skipped = 0
    for chunk in chunks_iter:
        if not isinstance(chunk, dict):
            skipped += 1
            continue

        questions = chunk.get("questions", {})
        if not isinstance(questions, dict):
            skipped += 1
            continue

        for question_type in ("direct", "indirect"):
            for q_item in questions.get(question_type, []):
                try:
                    question_text = str(q_item["question"]).strip()
                    answer_chunk_id = q_item["answer_chunk_id"]
                    tag = q_item.get("tag")
                except (KeyError, TypeError):
                    skipped += 1
                    continue

                if not question_text:
                    skipped += 1
                    continue

                # Normalize to a list for consistent downstream handling.
                if not isinstance(answer_chunk_id, list):
                    answer_chunk_id = [answer_chunk_id]

                if question_text in lookup:
                    logger.debug(
                        "Duplicate question text encountered; overwriting "
                        "previous entry: %r",
                        question_text,
                    )

                lookup[question_text] = QuestionMeta(
                    answer_chunk_id=answer_chunk_id,
                    tag=tag,
                    question_type=question_type,
                )

    if skipped:
        logger.warning("Skipped %d malformed question entr(y/ies) in dataset.", skipped)

    logger.info("Built question lookup with %d entries.", len(lookup))
    return lookup


# ---------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------
def parse_retrieved_chunks(retrieved_chunks: Any) -> Optional[list[Any]]:
    """
    Normalize the 'final_retrieved_chunks' cell into a Python list.

    Returns None if the value is missing or cannot be interpreted as a list.
    """
    if retrieved_chunks is None or (
        isinstance(retrieved_chunks, float) and pd.isna(retrieved_chunks)
    ):
        return None

    if isinstance(retrieved_chunks, list):
        return retrieved_chunks

    if isinstance(retrieved_chunks, str):
        try:
            parsed = ast.literal_eval(retrieved_chunks)
        except (ValueError, SyntaxError):
            # Fall back to treating the raw string as a single chunk id.
            return [retrieved_chunks]
        return parsed if isinstance(parsed, list) else [parsed]

    # Unexpected type (e.g. a bare number) — wrap as a single-element list.
    return [retrieved_chunks]


# ---------------------------------------------------------
# Row-level evaluation
# ---------------------------------------------------------
def is_predicted(answer_chunk_ids: Optional[list[Any]], retrieved_chunks: Any) -> bool:
    """True if any ground-truth chunk id appears in the retrieved chunks."""
    if not answer_chunk_ids:
        return False

    parsed = parse_retrieved_chunks(retrieved_chunks)
    if not parsed:
        return False

    return any(chunk in parsed for chunk in answer_chunk_ids)


def get_chunk_positions(
    answer_chunk_ids: Optional[list[Any]], retrieved_chunks: Any
) -> Optional[list[Optional[int]]]:
    """
    Return the 1-based position of each ground-truth chunk within the
    retrieved list, or None per chunk if it wasn't retrieved.
    """
    if not answer_chunk_ids:
        return None

    parsed = parse_retrieved_chunks(retrieved_chunks)
    if parsed is None:
        return None

    positions: list[Optional[int]] = []
    for chunk in answer_chunk_ids:
        positions.append(parsed.index(chunk) + 1 if chunk in parsed else None)
    return positions


# ---------------------------------------------------------
# Column ordering
# ---------------------------------------------------------
def reorder_added_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns so the ground-truth/prediction columns this module adds
    appear last, in ADDED_COLUMNS_ORDER, regardless of the order they were
    assigned in. All other (original) columns keep their existing order.
    """
    added_present = [c for c in ADDED_COLUMNS_ORDER if c in df.columns]
    other_columns = [c for c in df.columns if c not in added_present]
    return df[other_columns + added_present]


# ---------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------
def evaluate(df: pd.DataFrame, question_lookup: dict[str, QuestionMeta]) -> pd.DataFrame:
    """
    Augment the results dataframe with ground-truth and prediction columns:
    'question_type', 'tag', 'answer_chunk_id', 'predicted', 'retrieved_position'
    (added as the trailing columns, in that order).
    """
    df = df.copy()

    def lookup_meta(query: Any) -> Optional[QuestionMeta]:
        return question_lookup.get(str(query).strip())

    metadata = df["query"].apply(lookup_meta)
    unmatched = metadata.isna().sum()
    if unmatched:
        logger.warning(
            "%d of %d queries had no matching ground-truth question.",
            unmatched,
            len(df),
        )

    df["answer_chunk_id"] = metadata.apply(lambda m: m.answer_chunk_id if m else None)
    df["tag"] = metadata.apply(lambda m: m.tag if m else None)
    df["question_type"] = metadata.apply(lambda m: m.question_type if m else None)

    df["predicted"] = df.apply(
        lambda row: is_predicted(row["answer_chunk_id"], row["final_retrieved_chunks"]),
        axis=1,
    )

    df["retrieved_position"] = df.apply(
        lambda row: get_chunk_positions(
            row["answer_chunk_id"], row["final_retrieved_chunks"]
        ),
        axis=1,
    )

    return reorder_added_columns(df)


# ---------------------------------------------------------
# High-level convenience wrapper
# ---------------------------------------------------------
def run_evaluation(
    dataset_path: Union[str, Path],
    results_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    End-to-end helper: load the dataset + results, evaluate, optionally save,
    and return the augmented dataframe.

    Args:
        dataset_path: Path to the merged ground-truth JSON dataset.
        results_path: Path to the retriever results CSV (must contain
            'query' and 'final_retrieved_chunks' columns).
        output_path: If provided, the augmented dataframe is written here
            as a CSV. If omitted, nothing is written to disk.

    Returns:
        The augmented pandas DataFrame with ground-truth and prediction
        columns appended.
    """
    dataset = load_dataset(dataset_path)
    results_df = load_results(results_path)

    question_lookup = build_question_lookup(dataset)
    evaluated_df = evaluate(results_df, question_lookup)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        evaluated_df.to_csv(output_path, index=False)
        logger.info("Saved results to: %s", output_path)

    total = len(evaluated_df)
    correct = int(evaluated_df["predicted"].sum())
    logger.info(
        "Prediction accuracy: %d/%d (%.1f%%)",
        correct,
        total,
        (correct / total * 100) if total else 0.0,
    )

    return evaluated_df