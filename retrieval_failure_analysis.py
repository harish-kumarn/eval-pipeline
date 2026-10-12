"""
retrieval_failure_analysis.py

Standalone script to analyze per-query retrieval CSV output and produce:
  1. retriver_res/failure_analysis.csv   - per-query stage-by-stage presence + failure point
  2. retriver_res/failure_report.md      - overall funnel + tag breakdown + failure-point summary

Usage:
    python retrieval_failure_analysis.py --input path/to/retrieval_details.csv
    python retrieval_failure_analysis.py --input path/to/retrieval_details.csv --outdir retriver_res
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _split_field(value: str) -> List[str]:
    """Split a '; '-joined CSV field back into a list of chunk ids."""
    if not value or not isinstance(value, str):
        return []
    return [v.strip() for v in value.split(";") if v.strip()]


class RetrievalFailureAnalyzer:
    """Builds a per-query failure-point report and pipeline funnel from
    the per-query retrieval details CSV."""

    def generate_failure_analysis_report(
        self,
        csv_path: Path,
        outdir: Path,
    ) -> Dict[str, Any]:
        outdir.mkdir(parents=True, exist_ok=True)
        output_csv_path = outdir / "failure_analysis.csv"
        report_path = outdir / "failure_report.md"

        rows = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            logger.warning("No rows found in %s — skipping failure analysis", csv_path)
            return {"funnel": {}, "per_query": []}

        per_query = [self._classify_row(row) for row in rows]

        self._write_per_query_csv(per_query, output_csv_path)

        funnel = self._build_funnel(per_query)
        by_tag = self._build_tag_breakdown(per_query)
        failure_counts = self._build_failure_counts(per_query)

        report_text = self._format_report(funnel, by_tag, failure_counts, total=len(per_query))
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info("Per-query failure analysis saved → %s", output_csv_path)
        logger.info("Failure analysis report saved → %s", report_path)

        return {
            "funnel": funnel,
            "by_tag": by_tag,
            "failure_point_counts": failure_counts,
            "output_csv": str(output_csv_path),
            "report_path": str(report_path),
        }

    # ── row-level classification ────────────────────────────────────────

    def _classify_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        gt_ids = set(_split_field(row.get("ground_truth_chunk_id", "")))
        bm25_ids = set(_split_field(row.get("bm25_chunk_ids", "")))
        vector_ids = set(_split_field(row.get("vector_chunk_ids", "")))
        pre_rerank_ids = set(_split_field(row.get("pre_rerank_chunk_ids", "")))
        post_rerank_ids = set(_split_field(row.get("post_rerank_chunk_ids", "")))
        final_ids = set(_split_field(row.get("retrieved_chunk_ids", "")))

        in_bm25 = bool(gt_ids & bm25_ids)
        in_vector = bool(gt_ids & vector_ids)
        in_pre_rerank = bool(gt_ids & pre_rerank_ids)
        in_post_rerank = bool(gt_ids & post_rerank_ids)
        in_final = bool(gt_ids & final_ids)

        # Walk the ACTUAL pipeline order: BM25/Vector feed Pre-Rerank (their
        # union), which feeds Post-Rerank, which is truncated to Final.
        if not in_bm25 and not in_vector:
            failure_point = "Lost before retrieval (missed by BM25 and Vector)"
        elif not in_pre_rerank:
            # found by a retriever but dropped at the union/merge step — flags a merge bug
            failure_point = "Lost at merge (BM25∪Vector → Pre-Rerank)"
        elif not in_post_rerank:
            failure_point = "Lost at rerank"
        elif not in_final:
            failure_point = "Lost at truncation (outside final top-k)"
        else:
            failure_point = "Success"

        return {
            "qid": row.get("qid"),
            "query": row.get("query"),
            "tag": row.get("tag") or "unknown",
            "answer_chunk_id": "; ".join(sorted(gt_ids)),
            "in_bm25": in_bm25,
            "in_vector": in_vector,
            "in_pre_rerank": in_pre_rerank,
            "in_post_rerank": in_post_rerank,
            "in_final": in_final,
            "failure_point": failure_point,
            "rank": row.get("rank"),
        }

    def _write_per_query_csv(self, per_query: List[Dict[str, Any]], output_csv_path: Path) -> None:
        fieldnames = [
            "qid", "query", "tag", "answer_chunk_id",
            "in_bm25", "in_vector", "in_pre_rerank",
            "in_post_rerank", "in_final", "failure_point", "rank",
        ]
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_query)

    # ── aggregation ──────────────────────────────────────────────────────

    def _build_funnel(self, per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(per_query)

        def _pct(n: int) -> float:
            return round(100.0 * n / total, 1) if total else 0.0

        n_bm25 = sum(r["in_bm25"] for r in per_query)
        n_vector = sum(r["in_vector"] for r in per_query)
        n_pre_rerank = sum(r["in_pre_rerank"] for r in per_query)
        n_post_rerank = sum(r["in_post_rerank"] for r in per_query)
        n_final = sum(r["in_final"] for r in per_query)

        return {
            "total_queries": total,
            "stages": [
                {"stage": "Total queries", "present": total,
                 "cumulative_recall_pct": 100.0, "stage_loss": None},
                {"stage": "After BM25 (lexical retrieval)", "present": n_bm25,
                 "cumulative_recall_pct": _pct(n_bm25), "stage_loss": total - n_bm25},
                {"stage": "After Vector (semantic retrieval)", "present": n_vector,
                 "cumulative_recall_pct": _pct(n_vector), "stage_loss": total - n_vector},
                {"stage": "After Pre-Rerank (BM25 ∪ Vector union)", "present": n_pre_rerank,
                 "cumulative_recall_pct": _pct(n_pre_rerank), "stage_loss": total - n_pre_rerank},
                {"stage": "After Post-Rerank (reranked candidate list)", "present": n_post_rerank,
                 "cumulative_recall_pct": _pct(n_post_rerank), "stage_loss": n_pre_rerank - n_post_rerank},
                {"stage": "After Final Top-k (truncation)", "present": n_final,
                 "cumulative_recall_pct": _pct(n_final), "stage_loss": n_post_rerank - n_final},
            ],
        }

    def _build_tag_breakdown(self, per_query: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        by_tag: Dict[str, Dict[str, Any]] = {}
        for r in per_query:
            tag = r["tag"]
            by_tag.setdefault(tag, {"total": 0, "success": 0})
            by_tag[tag]["total"] += 1
            if r["failure_point"] == "Success":
                by_tag[tag]["success"] += 1
        for counts in by_tag.values():
            counts["recall_pct"] = round(100.0 * counts["success"] / counts["total"], 1) if counts["total"] else 0.0
        return by_tag

    def _build_failure_counts(self, per_query: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in per_query:
            counts[r["failure_point"]] = counts.get(r["failure_point"], 0) + 1
        return counts

    # ── report formatting ───────────────────────────────────────────────

    def _format_report(
        self,
        funnel: Dict[str, Any],
        by_tag: Dict[str, Dict[str, Any]],
        failure_counts: Dict[str, int],
        total: int,
    ) -> str:
        lines = ["# Retrieval Failure Analysis Report", ""]
        lines.append(f"Total queries: {total}")
        lines.append("")
        lines.append("## 1. Overall funnel")
        lines.append("")
        lines.append("| Stage | Queries with answer present | Cumulative recall | Stage-over-stage loss |")
        lines.append("|---|---|---|---|")
        for s in funnel["stages"]:
            loss = "—" if s["stage_loss"] is None else f"{s['stage_loss']} lost"
            lines.append(f"| {s['stage']} | {s['present']} | {s['cumulative_recall_pct']}% | {loss} |")
        lines.append("")

        lines.append("## 2. Breakdown by query type")
        lines.append("")
        lines.append("| Tag | Total | Successful | Recall |")
        lines.append("|---|---|---|---|")
        for tag, counts in sorted(by_tag.items()):
            lines.append(f"| {tag} | {counts['total']} | {counts['success']} | {counts['recall_pct']}% |")
        lines.append("")

        lines.append("## 3. Failure point breakdown")
        lines.append("")
        lines.append("| Failure point | Count | % of total |")
        lines.append("|---|---|---|")
        for point, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            pct = round(100.0 * count / total, 1) if total else 0.0
            lines.append(f"| {point} | {count} | {pct}% |")
        lines.append("")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze retrieval pipeline failures from a per-query CSV.")
    parser.add_argument("--input", required=True, help="Path to the per-query retrieval details CSV")
    parser.add_argument("--outdir", default="retriver_res", help="Output directory (default: retriver_res)")
    args = parser.parse_args()

    csv_path = Path(args.input)
    outdir = Path(args.outdir)

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    analyzer = RetrievalFailureAnalyzer()
    result = analyzer.generate_failure_analysis_report(csv_path=csv_path, outdir=outdir)

    print("\nFunnel summary:")
    for s in result["funnel"]["stages"]:
        loss = "—" if s["stage_loss"] is None else f"{s['stage_loss']} lost"
        print(f"  {s['stage']:<45} present={s['present']:<4} recall={s['cumulative_recall_pct']}%  loss={loss}")


if __name__ == "__main__":
    main()