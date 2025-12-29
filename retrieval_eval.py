import json
import random
from typing import List, Dict, Any
from pathlib import Path


class RetrieverEvaluationPipeline:
    """Production-ready pipeline for retriever evaluation datasets and metrics."""
    
    def __init__(self, 
                 input_file: str = 'retriever_input.json',
                 output_dir: str = '.'):
        """Initialize pipeline with input file and output directory."""
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.data = None
        self.eval_dataset = []
        self.metrics = {}
    
    # ==================== DATA LOADING ====================
    
    def load_data(self) -> Dict[str, Any]:
        """Load retriever input data."""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✓ Data loaded: {len(self.data)} chunks")
            return self.data
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.input_file} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in {self.input_file}")
    
    # ==================== DATASET CREATION ====================
    
    def create_eval_dataset(self) -> List[Dict[str, Any]]:
        """Create retriever evaluation dataset with random retrieved chunks."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.eval_dataset = []
        all_chunks = list(self.data.keys())
        
        for chunk_id, chunk_info in self.data.items():
            chunk_num = int(chunk_id.replace('chunk', ''))
            
            for idx, question in enumerate(chunk_info['questions']):
                retrieved = random.sample(all_chunks, min(2, len(all_chunks)))
                
                self.eval_dataset.append({
                    'qid': f"q{idx + 1}_c{chunk_num}",
                    'question': question,
                    'ground_truth_chunk': chunk_id,
                    'retrieved_chunks': retrieved,
                    'is_relevant': [c == chunk_id for c in retrieved]
                })
        
        print(f"✓ Created eval dataset: {len(self.eval_dataset)} question-chunk pairs")
        return self.eval_dataset
    
    # ==================== METRICS CALCULATION ====================
    
    def calculate_mrr(self, k: int = 5) -> float:
        """Mean Reciprocal Rank: average of 1/rank of first relevant item."""
        reciprocal_ranks = []
        
        for entry in self.eval_dataset:
            is_relevant = entry['is_relevant'][:k]
            rank = next((i + 1 for i, rel in enumerate(is_relevant) if rel), None)
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def calculate_recall_at_k(self, k: int = 5) -> float:
        """Recall@k: proportion of queries with at least one relevant item in top-k."""
        recalls = [1.0 if any(entry['is_relevant'][:k]) else 0.0 
                   for entry in self.eval_dataset]
        return sum(recalls) / len(recalls) if recalls else 0.0
    
    def calculate_precision_at_k(self, k: int = 5) -> float:
        """Precision@k: average number of relevant items in top-k divided by k."""
        precisions = [sum(entry['is_relevant'][:k]) / k 
                      for entry in self.eval_dataset]
        return sum(precisions) / len(precisions) if precisions else 0.0
    
    def calculate_ndcg(self, k: int = 5) -> float:
        """NDCG@k: normalized discounted cumulative gain considering position."""
        def dcg_at_k(is_relevant, k):
            return sum(1.0 / (i + 2) for i, rel in enumerate(is_relevant[:k]) if rel)
        
        ndcg_scores = []
        for entry in self.eval_dataset:
            is_relevant = entry['is_relevant']
            dcg = dcg_at_k(is_relevant, k)
            ideal = [True] * sum(is_relevant)
            idcg = dcg_at_k(ideal, k)
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
        
        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    
    def calculate_mean_rank(self) -> float:
        """Average position of relevant item across all queries."""
        ranks = [next((i + 1 for i, rel in enumerate(entry['is_relevant']) if rel), None)
                 for entry in self.eval_dataset]
        ranks = [r for r in ranks if r]
        return sum(ranks) / len(ranks) if ranks else float('inf')
    
    def calculate_per_query_metrics(self, k: int = 5) -> List[Dict[str, Any]]:
        """Calculate metrics for each individual query."""
        def dcg_at_k(is_relevant, k):
            return sum(1.0 / (i + 2) for i, rel in enumerate(is_relevant[:k]) if rel)
        
        per_query = []
        for entry in self.eval_dataset:
            is_relevant = entry['is_relevant'][:k]
            rank = next((i + 1 for i, rel in enumerate(is_relevant) if rel), None)
            num_relevant = sum(is_relevant)
            
            dcg = dcg_at_k(is_relevant, k)
            ideal = [True] * num_relevant
            idcg = dcg_at_k(ideal, k)
            
            per_query.append({
                'qid': entry['qid'],
                'question': entry['question'],
                'ground_truth_chunk': entry['ground_truth_chunk'],
                'retrieved_chunks': entry['retrieved_chunks'][:k],
                'rank_of_relevant': rank,
                'mrr': 1.0 / rank if rank else 0.0,
                'recall': 1.0 if any(is_relevant) else 0.0,
                'precision': num_relevant / k if k > 0 else 0.0,
                'ndcg': dcg / idcg if idcg > 0 else 0.0,
                'is_relevant': is_relevant
            })
        
        return per_query
    
    def calculate_all_metrics(self, k: int = 5) -> Dict[str, Any]:
        """Calculate all metrics."""
        per_query = self.calculate_per_query_metrics(k)
        
        self.metrics = {
            'mrr@k': self.calculate_mrr(k),
            'recall@k': self.calculate_recall_at_k(k),
            'precision@k': self.calculate_precision_at_k(k),
            'ndcg@k': self.calculate_ndcg(k),
            'hit_rate': self.calculate_recall_at_k(k),
            'mean_rank': self.calculate_mean_rank(),
            'k': k,
            'total_queries': len(self.eval_dataset),
            'per_query_metrics': per_query
        }
        return self.metrics
    
    # ==================== REPORTING & SAVING ====================
    
    def print_metrics_report(self) -> None:
        """Print formatted metrics report."""
        if not self.metrics:
            raise ValueError("Metrics not calculated. Call calculate_all_metrics() first.")
        
        print(f"\n{'='*70}")
        print("RETRIEVER EVALUATION METRICS REPORT")
        print('='*70)
        print(f"\nTotal Queries: {self.metrics['total_queries']} | Cutoff (k): {self.metrics['k']}")
        print(f"\n{'Metric':<20} {'Score':<12} {'Interpretation':<35}")
        print('-'*70)
        
        metrics_info = [
            ('MRR@k', self.metrics['mrr@k'], 'Rank of first match'),
            ('Recall@k', self.metrics['recall@k'], '% queries with relevant in top-k'),
            ('Precision@k', self.metrics['precision@k'], 'Avg relevance in top-k'),
            ('NDCG@k', self.metrics['ndcg@k'], 'Ranking quality (0-1)'),
            ('Hit Rate', self.metrics['hit_rate'], '% with any match'),
            ('Mean Rank', self.metrics['mean_rank'], 'Avg position'),
        ]
        
        for metric, value, interp in metrics_info:
            if metric == 'Mean Rank':
                print(f"{metric:<20} {value:<12.2f} {interp:<35}")
            else:
                print(f"{metric:<20} {value:<12.4f} {interp:<35}")
        
        print(f"\n{'='*70}")
        print("PER-QUERY METRICS (First 5)")
        print('='*70)
        
        for i, qm in enumerate(self.metrics['per_query_metrics'][:5], 1):
            print(f"\n{i}. {qm['qid']} | Question: {qm['question'][:60]}...")
            print(f"   Ground Truth: {qm['ground_truth_chunk']} | Retrieved: {', '.join(qm['retrieved_chunks'])}")
            print(f"   Rank: {qm['rank_of_relevant']} | MRR: {qm['mrr']:.4f} | Recall: {qm['recall']:.1f} | NDCG: {qm['ndcg']:.4f}")
        
        remaining = len(self.metrics['per_query_metrics']) - 5
        if remaining > 0:
            print(f"\n... and {remaining} more queries (see retriever_metrics.json)")
        
        print(f"\n{'='*70}\n")
    
    def save_outputs(self, 
                     eval_file: str = 'retriever_eval_dataset.json',
                     metrics_file: str = 'retriever_metrics.json') -> None:
        """Save evaluation dataset and metrics to JSON files."""
        if not self.eval_dataset:
            raise ValueError("Evaluation dataset not created.")
        if not self.metrics:
            raise ValueError("Metrics not calculated.")
        
        eval_path = self.output_dir / eval_file
        metrics_path = self.output_dir / metrics_file
        
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(self.eval_dataset, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved eval dataset: {eval_path}")
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved metrics: {metrics_path}")
    
    # ==================== MAIN PIPELINE ====================
    
    def run(self, k: int = 5, save: bool = True) -> Dict[str, Any]:
        """Execute complete pipeline: load data -> create dataset -> calculate metrics."""
        print(f"\n{'='*70}")
        print("RETRIEVER EVALUATION PIPELINE")
        print('='*70)
        
        self.load_data()
        self.create_eval_dataset()
        self.calculate_all_metrics(k=k)
        self.print_metrics_report()
        
        if save:
            self.save_outputs()
        
        print("✓ Pipeline completed successfully")
        return self.metrics


if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = RetrieverEvaluationPipeline(
        input_file='retriever_input.json',
        output_dir='.'
    )
    
    # Execute complete pipeline
    metrics = pipeline.run(k=5, save=True)