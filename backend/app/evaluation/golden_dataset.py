from typing import List
from app.evaluation.models import BenchmarkSample


def get_default_golden_dataset() -> List[BenchmarkSample]:
    """Returns a curriculum-aligned academic testset covering Factoid, Reasoning,
    Multi-hop, and Negative/Unanswerable queries.
    """
    return [
        BenchmarkSample(
            id="ACO-001",
            question="Thuật toán Tối ưu hóa đàn kiến (ACO) dựa trên cơ chế sinh học nào?",
            ground_truth_answer="Thuật toán ACO dựa trên tập tính tìm đường của đàn kiến thông qua nồng độ vết mùi pheromone trên đường đi.",
            ground_truth_contexts=[
                "vết mùi pheromone",
                "tập tính tìm đường của đàn kiến",
                "Ant Colony Optimization",
            ],
            expected_document_title="4 Tối ưu đàn kiến.pdf",
            question_type="factoid",
            is_unanswerable=False,
        ),
        BenchmarkSample(
            id="ACO-002",
            question="Quy tắc cập nhật vết mùi toàn cục trong thuật toán ACO được thực hiện như thế nào?",
            ground_truth_answer="Cập nhật vết mùi toàn cục dựa trên lượng mùi bay hơi theo hệ số rho và lượng mùi bổ sung tỷ lệ nghịch với độ dài lời giải tốt nhất.",
            ground_truth_contexts=[
                "(1-rho)*tau + Delta_tau",
                "bay hơi",
                "cập nhật vết mùi toàn cục",
            ],
            expected_document_title="4 Tối ưu đàn kiến.pdf",
            question_type="reasoning",
            is_unanswerable=False,
        ),
        BenchmarkSample(
            id="KNAP-001",
            question="Bài toán Cái túi (Knapsack Problem) 0-1 có thể giải tối ưu bằng phương pháp nào?",
            ground_truth_answer="Bài toán Cái túi 0-1 có thể giải tối ưu bằng Quy hoạch động (Dynamic Programming) hoặc Nhánh và Cận (Branch and Bound).",
            ground_truth_contexts=[
                "Quy hoạch động",
                "bài toán cái túi",
                "Dynamic Programming",
                "Knapsack",
            ],
            expected_document_title="knapsack.pdf",
            question_type="factoid",
            is_unanswerable=False,
        ),
        BenchmarkSample(
            id="OPT-001",
            question="So sánh sự khác biệt cơ bản giữa Giải thuật Di truyền (GA) và Tối ưu đàn kiến (ACO)?",
            ground_truth_answer="GA sử dụng các toán tử lai ghép và đột biến trên quần thể nhiễm sắc thể, trong khi ACO dựa trên dấu vết mùi pheromone phân tán và học tăng cường của các cá thể kiến.",
            ground_truth_contexts=[
                "Genetic Algorithm",
                "lai ghép và đột biến",
                "ACO",
                "pheromone",
            ],
            expected_document_title="De cuong Toi uu hoa.pdf",
            question_type="multihop",
            is_unanswerable=False,
        ),
        BenchmarkSample(
            id="NEG-001",
            question="Thuyết tương đối hẹp của Einstein giải thích vận tốc ánh sáng như thế nào?",
            ground_truth_answer="Tài liệu bài giảng không đề cập đến thông tin này.",
            ground_truth_contexts=[],
            expected_document_title=None,
            question_type="unanswerable",
            is_unanswerable=True,
        ),
        BenchmarkSample(
            id="NEG-002",
            question="Quy trình nướng bánh pizza Napoli theo tiêu chuẩn quốc tế?",
            ground_truth_answer="Tài liệu bài giảng không đề cập đến thông tin này.",
            ground_truth_contexts=[],
            expected_document_title=None,
            question_type="unanswerable",
            is_unanswerable=True,
        ),
    ]
