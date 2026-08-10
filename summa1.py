"""
evaluate_retriever.py

Evaluates retriever performance against a ground-truth question/answer dataset.

For each query in a retriever-results CSV, this script:
  1. Looks up the ground-truth answer chunk ids, tag, and question type from a
     merged JSON dataset (matched by exact question text).
  2. Determines whether any ground-truth chunk was retrieved ("predicted").
  3. Computes the 1-based position of each ground-truth chunk within the
     retrieved list (if present).
  4. Writes an augmented CSV and prints a short preview / summary.

Usage:
    python evaluate_retriever.py \
        --dataset retriver_res/merged_testing_dataset.json \
        --results retriver_res/04_08_dummy1_18_54.csv \
        --output  retriver_res/04_08_dummy1_18_54_with_answers.csv

Exit codes:
    0  success
    1  input file missing / unreadable
    2  input file malformed (bad JSON / CSV / schema)
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

logger = logging.getLogger("evaluate_retriever")


# ---------------------------------------------------------
# Data model
# ---------------------------------------------------------
@dataclass(frozen=True)
class QuestionMeta:
    """Ground-truth metadata associated with a single question."""

    answer_chunk_id: list[str]
    tag: Any
    question_type: str


# ---------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retriever results against ground-truth answer chunks."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("retriver_res/merged_testing_dataset.json"),
        help="Path to the merged ground-truth JSON dataset.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("retriver_res/04_08_dummy1_18_54_with_answers.csv"),
        help="Path to the retriever results CSV (must contain 'query' and "
        "'final_retrieved_chunks' columns).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the augmented CSV. Defaults to "
        "'<results-stem>_with_answers.csv' next to the results file.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Number of rows to print as a preview after processing (0 to disable).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------
# Loading
# ---------------------------------------------------------
def load_dataset(path: Path) -> dict[str, Any]:
    """Load and lightly validate the merged ground-truth JSON dataset."""
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


def load_results(path: Path) -> pd.DataFrame:
    """Load the retriever results CSV and validate required columns."""
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
def is_predicted(
    answer_chunk_ids: Optional[list[Any]], retrieved_chunks: Any
) -> bool:
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
# Main evaluation pipeline
# ---------------------------------------------------------
def evaluate(df: pd.DataFrame, question_lookup: dict[str, QuestionMeta]) -> pd.DataFrame:
    """Augment the results dataframe with ground-truth and prediction columns."""
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

    return df


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_path = args.output or args.results.with_name(
        f"{args.results.stem}_with_answers.csv"
    )

    try:
        dataset = load_dataset(args.dataset)
        results_df = load_results(args.results)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    except ValueError as exc:
        logger.error(str(exc))
        return 2

    question_lookup = build_question_lookup(dataset)
    evaluated_df = evaluate(results_df, question_lookup)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluated_df.to_csv(output_path, index=False)
    logger.info("Saved results to: %s", output_path)

    if args.preview_rows > 0:
        preview_columns = [
            "query",
            "answer_chunk_id",
            "tag",
            "question_type",
            "final_retrieved_chunks",
            "predicted",
            "retrieved_position",
        ]
        available_columns = [c for c in preview_columns if c in evaluated_df.columns]
        print(evaluated_df[available_columns].head(args.preview_rows).to_string())

        total = len(evaluated_df)
        correct = int(evaluated_df["predicted"].sum())
        logger.info(
            "Prediction accuracy: %d/%d (%.1f%%)",
            correct,
            total,
            (correct / total * 100) if total else 0.0,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())