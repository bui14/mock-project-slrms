import unittest

from app.evaluation.golden_dataset import get_default_golden_dataset
from app.evaluation.metrics import (
    compute_answer_relevance,
    compute_context_precision,
    compute_context_recall,
    compute_faithfulness,
    compute_hit_at_k,
    compute_mrr,
    compute_refusal_correctness,
    is_context_relevant,
)
from app.evaluation.models import BenchmarkReport, BenchmarkSample
from app.evaluation.runner import BenchmarkRunner


class RAGBenchmarkTestCase(unittest.TestCase):
    def test_golden_dataset_structure(self):
        dataset = get_default_golden_dataset()
        self.assertGreaterEqual(len(dataset), 5)
        for sample in dataset:
            self.assertTrue(sample.id)
            self.assertTrue(sample.question)
            self.assertTrue(sample.ground_truth_answer)
            self.assertIn(sample.question_type, ["factoid", "reasoning", "multihop", "unanswerable"])

    def test_retrieval_metrics(self):
        # relevance vector: [False, True, False, True]
        vec = [False, True, False, True]

        # Hit@1 is 0, Hit@2 is 1, Hit@4 is 1
        self.assertEqual(compute_hit_at_k(vec, 1), 0.0)
        self.assertEqual(compute_hit_at_k(vec, 2), 1.0)
        self.assertEqual(compute_hit_at_k(vec, 4), 1.0)

        # MRR: first True at index 1 -> 1 / (1 + 1) = 0.5
        self.assertEqual(compute_mrr(vec), 0.5)

        # Context Precision:
        # At rank 2: 1/2 = 0.5
        # At rank 4: 2/4 = 0.5
        # Total relevant = 2 -> (0.5 + 0.5) / 2 = 0.5
        self.assertAlmostEqual(compute_context_precision(vec), 0.5)

    def test_context_recall_computation(self):
        chunks = [
            "Thuật toán Tối ưu đàn kiến ACO dựa trên pheromone",
            "Bài toán cái túi được giải bằng quy hoạch động",
        ]
        ground_truth = [
            "Tối ưu đàn kiến ACO dựa trên pheromone",
            "Không liên quan chút nào cả",
        ]
        recall = compute_context_recall(chunks, ground_truth)
        self.assertEqual(recall, 0.5)

    def test_faithfulness_and_refusal(self):
        # Unanswerable question with correct refusal
        self.assertEqual(
            compute_faithfulness(
                generated_answer="Tài liệu bài giảng không đề cập đến thông tin này.",
                retrieved_chunks=[],
                is_grounded=False,
                is_unanswerable=True,
            ),
            1.0,
        )
        self.assertEqual(
            compute_refusal_correctness(
                generated_answer="Tài liệu bài giảng không đề cập đến thông tin này.",
                is_grounded=False,
                is_unanswerable=True,
            ),
            1.0,
        )

        # Unanswerable question where model hallucinated
        self.assertEqual(
            compute_faithfulness(
                generated_answer="Bánh pizza Napoli cần 400 độ C.",
                retrieved_chunks=[],
                is_grounded=True,
                is_unanswerable=True,
            ),
            0.0,
        )
        self.assertEqual(
            compute_refusal_correctness(
                generated_answer="Bánh pizza Napoli cần 400 độ C.",
                is_grounded=True,
                is_unanswerable=True,
            ),
            0.0,
        )

    def test_benchmark_runner_full_cycle(self):
        def fake_retriever(q, doc, top_k):
            return ["Thuật toán ACO dựa trên nồng độ vết mùi pheromone."]

        def fake_generator(q, ctxs):
            if "pizza" in q.lower() or "einstein" in q.lower():
                return "Tài liệu không có thông tin.", False
            return "Thuật toán ACO dựa trên nồng độ vết mùi pheromone.", True

        runner = BenchmarkRunner(
            retriever_fn=fake_retriever,
            generator_fn=fake_generator,
        )
        report = runner.run(top_k=3)

        self.assertIsInstance(report, BenchmarkReport)
        self.assertGreater(report.total_samples, 0)
        self.assertGreaterEqual(report.mean_hit_at_1, 0.0)
        self.assertGreaterEqual(report.refusal_accuracy, 0.5)

        # Format markdown check
        md = runner.format_markdown_report(report)
        self.assertIn("SLRMS Evaluation Report", md)
        self.assertIn("Hit Rate @ 1", md)


if __name__ == "__main__":
    unittest.main()
