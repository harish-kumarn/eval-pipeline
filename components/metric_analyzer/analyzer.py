import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Iterable

from components.metric_analyzer.models import PerformanceCategory, QueryAnalysis
from components.metric_analyzer.patterns import analyze_rag_flow
from components.metric_analyzer.io_utils import load_metrics
from components.metric_analyzer.reporting import to_dataframe, export_json, export_to_csv
from config.logger import setup_logger

logger = setup_logger(name="eval_pipeline", tag="ANALYZER")


class RAGMetricAnalyzer:
    """
    Config-driven RAG Metric Analyzer.
    Loads thresholds from metrics_config.json
    """

    def __init__(self, metrics_config_path: str):
        logger.info("Loading metrics config from %s", metrics_config_path)
        with open(metrics_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        metric_thresholds = config.get("metric_thresholds")
        if not metric_thresholds:
            raise ValueError("metric_thresholds missing in metrics_config.json")

        try:
            self.retrieval_threshold = float(metric_thresholds["retrieval_threshold"])
            self.generation_threshold = float(metric_thresholds["generation_threshold"])
        except KeyError as e:
            raise KeyError(f"Missing required threshold: {e.args[0]}")

        # Per-metric thresholds only
        self.metric_thresholds = {
            k: float(v)
            for k, v in metric_thresholds.items()
            if k not in {"retrieval_threshold", "generation_threshold"}
        }

        logger.info(
            "Thresholds loaded — retrieval=%.2f, generation=%.2f, per-metric=%s",
            self.retrieval_threshold, self.generation_threshold, self.metric_thresholds,
        )
        self.analysis_results = None

    # -----------------------------------
    # Score aggregation (robust)
    # -----------------------------------
    def extract_score(self, metrics: Dict, kind: str) -> float:
        if not metrics:
            return 0.0

        if kind == "retrieval":
            keys = ("recall", "precision", "mrr")
        else:
            keys = ("faithfulness", "answer_correctness", "context_precision")

        values = [metrics[k] for k in keys if k in metrics]

        return float(np.mean(values)) if values else 0.0

    # -----------------------------------
    # Threshold gating (strict)
    # -----------------------------------
    def pass_thresholds(self, metrics: Dict, keys: Iterable[str]) -> bool:
        for k in keys:
            if k not in metrics:
                return False  # strict mode
            if metrics[k] < self.metric_thresholds.get(k, 0.0):
                return False
        return True

    # -----------------------------------
    # Categorization
    # -----------------------------------
    def categorize_queries(self, retrieval, generation):
        categories = {c: [] for c in PerformanceCategory}

        gen_map = {
            g.get("qid") or g.get("query_id") or g.get("id") or g.get("query", ""): g
            for g in generation or []
        }

        skipped = 0
        for r in retrieval or []:
            qid = r.get("qid") or r.get("query_id") or r.get("id") or r.get("query", "")
            if not qid or qid not in gen_map:
                logger.info("Skipping qid=%s — not found in generation results", qid)
                skipped += 1
                continue

            g = gen_map[qid]

            r_score = self.extract_score(r, "retrieval")
            g_score = self.extract_score(g, "generation")

            r_ok = r_score >= self.retrieval_threshold
            g_ok = g_score >= self.generation_threshold

            if r_ok and g_ok:
                category = PerformanceCategory.BOTH_SUCCESS
            elif not r_ok and g_ok:
                category = PerformanceCategory.RETRIEVAL_FAIL_GEN_SUCCESS
            elif r_ok and not g_ok:
                category = PerformanceCategory.RETRIEVAL_SUCCESS_GEN_FAIL
            else:
                category = PerformanceCategory.BOTH_FAIL

            flow = analyze_rag_flow(r, g, self.metric_thresholds)

            categories[category].append(
                QueryAnalysis(
                    query_id=qid,
                    query=r.get("query") or g.get("query", ""),
                    retrieval_score=r_score,
                    generation_score=g_score,
                    category=category,
                    retrieval_metrics=r,
                    generation_metrics=g,
                    retrieval_status=flow.retrieval_status,
                    tag=r.get("tag") or g.get("tag") or "",
                    hit_type=flow.hit_type,
                    outcome=flow.outcome,
                )
            )

        total = sum(len(v) for v in categories.values())
        if skipped:
            logger.info("Skipped %d queries not matched between retrieval and generation results", skipped)
        for cat, queries in categories.items():
            logger.info("  %-35s : %d queries", cat.value, len(queries))
        logger.info("Categorization complete — %d queries categorized", total)

        return categories

    # -----------------------------------
    # Public API
    # -----------------------------------
    def analyze(self, retrieval_path: Optional[str], generation_path: Optional[str]):
        logger.info(
            "Analyzing — retrieval_path=%s, generation_path=%s",
            retrieval_path or "(none)", generation_path or "(none)",
        )

        retrieval = load_metrics(retrieval_path) if retrieval_path else []
        generation = load_metrics(generation_path) if generation_path else []
        logger.info(
            "Loaded %d retrieval records, %d generation records",
            len(retrieval), len(generation),
        )

        categories = self.categorize_queries(retrieval, generation)

        self.analysis_results = {
            "categories": categories,
            "total_queries": sum(len(v) for v in categories.values()),
        }

        logger.info("Analysis complete — total_queries=%d", self.analysis_results["total_queries"])
        return self.analysis_results


