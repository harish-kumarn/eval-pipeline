"""Token usage tracker for the evaluation pipeline.

Tracks LLM and embedding token consumption per question and per metric,
then produces a structured summary.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Any


class TokenTracker:
    """
    Context-aware token tracker.

    Usage pattern:
        tracker.set_context(metric_name="faithfulness", qid="q1_chunk1")
        # ... RAGAS/Gemini calls happen here ...
        summary = tracker.get_summary()
    """

    def __init__(self):
        # raw event log: list of {"metric", "qid", "prompt", "completion", "embedding"}
        self._events: list[Dict[str, Any]] = []

        # current evaluation context
        self._current_metric: Optional[str] = None
        self._current_qid:    Optional[str] = None

    # --------------------------------------------------
    # Context management
    # --------------------------------------------------

    def set_context(self, metric: Optional[str], qid: Optional[str]) -> None:
        """Set the current (metric, question) being evaluated."""
        self._current_metric = metric
        self._current_qid    = qid

    def clear_context(self) -> None:
        self._current_metric = None
        self._current_qid    = None

    # --------------------------------------------------
    # Recording
    # --------------------------------------------------

    def record_llm(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record one LLM generate_content call."""
        self._events.append({
            "kind":       "llm",
            "metric":     self._current_metric,
            "qid":        self._current_qid,
            "prompt":     int(prompt_tokens),
            "completion": int(completion_tokens),
            "embedding":  0,
        })

    def record_embedding(self, estimated_tokens: int) -> None:
        """Record one embedding call (token count is estimated by word count)."""
        self._events.append({
            "kind":       "embedding",
            "metric":     self._current_metric,
            "qid":        self._current_qid,
            "prompt":     0,
            "completion": 0,
            "embedding":  int(estimated_tokens),
        })

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a structured token-usage report.

        Schema:
        {
          "total": {
            "prompt_tokens", "completion_tokens", "embedding_tokens", "total_tokens"
          },
          "per_question": {
            "<qid>": {
              "prompt_tokens", "completion_tokens", "embedding_tokens", "total_tokens",
              "per_metric": {
                "<metric>": {
                  "prompt_tokens", "completion_tokens", "embedding_tokens", "total_tokens"
                }
              }
            }
          }
        }
        """
        # Accumulators: totals[qid][metric] = {prompt, completion, embedding}
        totals_by_q_m: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"prompt": 0, "completion": 0, "embedding": 0})
        )
        grand = {"prompt": 0, "completion": 0, "embedding": 0}

        for ev in self._events:
            qid    = ev["qid"]    or "_unknown_"
            metricanswer_correctness = ev["metric"] or "_unknown_"

            totals_by_q_m[qid][metric]["prompt"]     += ev["prompt"]
            totals_by_q_m[qid][metric]["completion"] += ev["completion"]
            totals_by_q_m[qid][metric]["embedding"]  += ev["embedding"]

            grand["prompt"]     += ev["prompt"]
            grand["completion"] += ev["completion"]
            grand["embedding"]  += ev["embedding"]

        # Build per_question
        per_question: Dict[str, Any] = {}
        for qid, metrics in sorted(totals_by_q_m.items()):
            q_p = q_c = q_e = 0
            per_metric: Dict[str, Any] = {}

            for metric, counts in sorted(metrics.items()):
                p, c, e = counts["prompt"], counts["completion"], counts["embedding"]
                per_metric[metric] = {
                    "prompt_tokens":     p,
                    "completion_tokens": c,
                    "embedding_tokens":  e,
                    "total_tokens":      p + c + e,
                }
                q_p += p; q_c += c; q_e += e

            per_question[qid] = {
                "prompt_tokens":     q_p,
                "completion_tokens": q_c,
                "embedding_tokens":  q_e,
                "total_tokens":      q_p + q_c + q_e,
                "per_metric":        per_metric,
            }

        gp, gc, ge = grand["prompt"], grand["completion"], grand["embedding"]
        return {
            "total": {
                "prompt_tokens":     gp,
                "completion_tokens": gc,
                "embedding_tokens":  ge,
                "total_tokens":      gp + gc + ge,
            },
            "per_question": per_question,
        }

    # --------------------------------------------------
    # Convenience
    # --------------------------------------------------

    def total_calls(self) -> int:
        return len(self._events)

    def reset(self) -> None:
        self._events.clear()
        self.clear_context()
