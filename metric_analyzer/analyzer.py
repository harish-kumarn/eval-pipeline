import numpy as np
from typing import Dict
from metric_analyzer.models import PerformanceCategory, QueryAnalysis
from metric_analyzer.patterns import analyze_retrieval_pattern, analyze_generation_pattern
from metric_analyzer.io_utils import load_metrics
from metric_analyzer.reporting import to_dataframe, export_json, export_to_csv


class RAGMetricAnalyzer:

    def __init__(self, retrieval_threshold=0.7, generation_threshold=0.7, metric_thresholds=None):
        self.retrieval_threshold = retrieval_threshold
        self.generation_threshold = generation_threshold
        self.metric_thresholds = metric_thresholds or {
            'recall': 0.7,
            'precision': 0.7,
            'mrr': 0.7,
            'faithfulness': 0.7,
            'context_precision': 0.7,
            'context_recall': 0.7,
            'answer_correctness': 0.7,
            'answer_relevancy': 0.7
        }
        self.analysis_results = None

    def extract_score(self, metrics: Dict, kind: str) -> float:
        keys = (
            ['score', 'mrr', 'recall', 'recall_at_k', 'precision', 'ndcg', 'map']
            if kind == 'retrieval'
            else ['score', 'faithfulness', 'answer_correctness', 'relevance', 'answer_relevancy']
        )
        for k in keys:
            if k in metrics:
                return float(metrics[k])
        return 0.0

    def categorize_queries(self, retrieval, generation):
        categories = {c: [] for c in PerformanceCategory}
        gen_map = {g.get('qid') or g.get('query_id') or g.get('id') or g.get('query', ''): g for g in generation if g}

        for r in retrieval:
            qid = r.get('qid') or r.get('query_id') or r.get('id') or r.get('query', '')
            if qid not in gen_map:
                continue

            g = gen_map[qid]
            r_score = self.extract_score(r, 'retrieval')
            g_score = self.extract_score(g, 'generation')

            r_ok = r_score >= self.retrieval_threshold
            g_ok = g_score >= self.generation_threshold

            if r_ok and g_ok:
                cat = PerformanceCategory.BOTH_SUCCESS
            elif not r_ok and g_ok:
                cat = PerformanceCategory.RETRIEVAL_FAIL_GEN_SUCCESS
            elif r_ok and not g_ok:
                cat = PerformanceCategory.RETRIEVAL_SUCCESS_GEN_FAIL
            else:
                cat = PerformanceCategory.BOTH_FAIL

            categories[cat].append(QueryAnalysis(
                query_id=qid,
                query=r.get('query', '') or g.get('query', ''),
                retrieval_score=r_score,
                generation_score=g_score,
                category=cat,
                retrieval_metrics=r,
                generation_metrics=g,
                retrieval_pattern=analyze_retrieval_pattern(r, self.metric_thresholds),
                generation_pattern=analyze_generation_pattern(g, self.metric_thresholds)
            ))

        return categories

    def analyze(self, retrieval_path, generation_path):
        retrieval = load_metrics(retrieval_path)
        generation = load_metrics(generation_path)

        categories = self.categorize_queries(retrieval, generation)
        self.analysis_results = {
            "categories": categories,
            "total_queries": sum(len(v) for v in categories.values())
        }
        return self.analysis_results

if __name__ == "__main__":
    analyzer = RAGMetricAnalyzer(
        retrieval_threshold=0.7,
        generation_threshold=0.7,
        metric_thresholds={
            'recall': 0.7,
            'precision': 0.7,
            'mrr': 0.7,
            'faithfulness': 0.7,
            'context_precision': 0.7,
            'context_recall': 0.7,
            'answer_correctness': 0.7,
            'answer_relevancy': 0.7
        }
    )

    results = analyzer.analyze(
        retrieval_path="retriever_metrics.json",
        generation_path="generator_metrics.json"
    )

    print(f"\nTotal queries analyzed: {results['total_queries']}")

    df = to_dataframe(results["categories"])
    print(df)

    export_json(results["categories"], "analysis_report.json")
    export_to_csv(results["categories"], "analysis_report.csv")

