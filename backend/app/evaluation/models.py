from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


@dataclass
class BenchmarkSample:
    """A single evaluation question in the golden testset."""
    id: str
    question: str
    ground_truth_answer: str
    ground_truth_contexts: List[str]
    expected_document_title: Optional[str] = None
    question_type: Literal["factoid", "reasoning", "multihop", "unanswerable"] = "factoid"
    is_unanswerable: bool = False


@dataclass
class SampleMetricResult:
    """Evaluation metrics for a single sample."""
    sample_id: str
    question: str
    question_type: str
    is_unanswerable: bool
    retrieved_contexts: List[str]
    generated_answer: str
    is_grounded: bool

    # Retrieval metrics
    hit_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    # Generation metrics
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    refusal_correctness: float = 0.0

    # Operational metrics
    latency_ms: float = 0.0


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report over the evaluation testset."""
    total_samples: int
    mean_hit_at_1: float
    mean_hit_at_3: float
    mean_hit_at_5: float
    mean_mrr: float
    mean_context_precision: float
    mean_context_recall: float
    mean_faithfulness: float
    mean_answer_relevance: float
    refusal_accuracy: float
    # Hardware & Telemetry
    mean_latency_ms: float
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    peak_ram_mb: float = 0.0
    gpu_vram_gb: float = 0.0
    sample_results: List[SampleMetricResult] = field(default_factory=list)

    def summary_dict(self) -> dict:
        return {
            "Total Samples": self.total_samples,
            "Hit Rate @ 1": f"{self.mean_hit_at_1 * 100:.2f}%",
            "Hit Rate @ 3": f"{self.mean_hit_at_3 * 100:.2f}%",
            "Hit Rate @ 5": f"{self.mean_hit_at_5 * 100:.2f}%",
            "MRR @ 3": f"{self.mean_mrr:.4f}",
            "Context Precision @ 3": f"{self.mean_context_precision:.4f}",
            "Context Recall": f"{self.mean_context_recall * 100:.2f}%",
            "Faithfulness": f"{self.mean_faithfulness * 100:.2f}%",
            "Answer Relevancy": f"{self.mean_answer_relevance * 100:.2f}%",
            "Refusal Accuracy (Zero-Hallucination)": f"{self.refusal_accuracy * 100:.2f}%",
            "End-to-End Latency": f"{self.mean_latency_ms:.2f} ms",
            "Retrieval Latency (Stage 1)": f"{self.retrieval_latency_ms:.2f} ms",
            "Rerank Latency (Stage 2)": f"{self.rerank_latency_ms:.2f} ms",
            "Generation Latency (Stage 3)": f"{self.generation_latency_ms:.2f} ms",
            "Peak Host RAM": f"{self.peak_ram_mb:.1f} MB",
            "Peak GPU VRAM": f"{self.gpu_vram_gb:.2f} GB (Zero GPU Required / CPU-Native)",
        }
