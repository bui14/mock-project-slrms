import random
from typing import List
from app.evaluation.models import BenchmarkSample


def generate_curriculum_benchmark_dataset(target_size: int = 100) -> List[BenchmarkSample]:
    """Generates a large-scale, curriculum-aligned academic testset (50 to 500 questions)
    covering Factoid, Reasoning, Multi-hop, and Negative Out-of-Domain controls.
    """
    templates = [
        # ACO Topic
        {
            "topic": "ACO",
            "doc": "4 Tối ưu đàn kiến.pdf",
            "questions": [
                ("Thuật toán Tối ưu hóa đàn kiến (ACO) dựa trên cơ chế sinh học nào?",
                 "Thuật toán ACO dựa trên tập tính tìm đường của đàn kiến thông qua nồng độ vết mùi pheromone trên đường đi.",
                 ["vết mùi pheromone", "tập tính tìm đường của đàn kiến", "ACO"], "factoid"),
                ("Quy tắc cập nhật vết mùi toàn cục trong ACO có vai trò gì?",
                 "Quy tắc cập nhật toàn cục giúp tăng lượng pheromone trên đường đi ngắn nhất và làm bay hơi pheromone trên đường xấu.",
                 ["cập nhật vết mùi toàn cục", "bay hơi", "(1-rho)*tau"], "reasoning"),
                ("Hệ số bay hơi vết mùi rho trong thuật toán đàn kiến ảnh hưởng thế nào đến tốc độ hội tụ?",
                 "Hệ số rho điều chỉnh tốc độ bay hơi; nếu rho quá lớn thuật toán dễ rơi vào cực trị địa phương, nếu quá nhỏ tốc độ hội tụ sẽ chậm.",
                 ["hệ số bay hơi", "hội tụ", "cực trị địa phương"], "reasoning"),
                ("Tại sao trong thuật toán MMAS (Max-Min Ant System) lại giới hạn nồng độ pheromone trong đoạn [tau_min, tau_max]?",
                 "Giới hạn trong đoạn [tau_min, tau_max] nhằm tránh hiện tượng ứ đọng nồng độ mùi tại một cung đường, giúp duy trì khả năng khám phá không gian tìm kiếm.",
                 ["MMAS", "tau_min", "tau_max", "ứ đọng"], "reasoning"),
            ]
        },
        # Knapsack Topic
        {
            "topic": "KNAP",
            "doc": "knapsack.pdf",
            "questions": [
                ("Bài toán Cái túi (Knapsack Problem) 0-1 có đặc điểm toán học gì?",
                 "Bài toán Cái túi 0-1 là bài toán tối ưu tổ hợp thuộc lớp NP-khó, trong đó mỗi đồ vật chỉ được chọn 0 hoặc 1 lần.",
                 ["NP-khó", "bài toán cái túi", "0-1"], "factoid"),
                ("Phương pháp Quy hoạch động (Dynamic Programming) giải bài toán Cái túi 0-1 có độ phức tạp thời gian là bao nhiêu?",
                 "Phương pháp Quy hoạch động giải bài toán Cái túi 0-1 có độ phức tạp thời gian là O(n*W), là độ phức tạp giả đa thức.",
                 ["O(n*W)", "giả đa thức", "Quy hoạch động"], "factoid"),
                ("Khi nào thuật toán tham lam (Greedy) không tìm được nghiệm tối ưu cho bài toán Cái túi 0-1?",
                 "Thuật toán tham lam dựa trên tỷ số giá trị/trọng lượng có thể bỏ sót các tổ hợp đồ vật lấp đầy tối ưu dung lượng túi trong bài toán 0-1.",
                 ["tham lam", "tỷ số giá trị", "nghiệm tối ưu"], "reasoning"),
            ]
        },
        # Optimization Syllabus Topic
        {
            "topic": "OPT",
            "doc": "De cuong Toi uu hoa.pdf",
            "questions": [
                ("Môn học Tối ưu hóa bao gồm những nội dung chương trình cốt lõi nào?",
                 "Đề cương môn học Tối ưu hóa bao gồm: Quy hoạch tuyến tính, Phương pháp đơn hình, Quy hoạch nguyên, Giải thuật di truyền (GA) và Tối ưu hóa đàn kiến (ACO).",
                 ["Quy hoạch tuyến tính", "Phương pháp đơn hình", "Giải thuật di truyền", "ACO"], "factoid"),
                ("Phương pháp Đơn hình (Simplex Method) tìm nghiệm tối ưu dựa trên nguyên lý hình học nào?",
                 "Phương pháp đơn hình di chuyển giữa các điểm cực biên (vertices) liền kề trên đa diện lồi của miền chấp nhận được theo hướng cải thiện hàm mục tiêu.",
                 ["điểm cực biên", "đa diện lồi", "đơn hình"], "reasoning"),
                ("So sánh sự khác biệt cơ bản giữa Giải thuật Di truyền (GA) và Tối ưu đàn kiến (ACO)?",
                 "GA sử dụng các toán tử lai ghép và đột biến trên quần thể cá thể, trong khi ACO dựa trên dấu vết mùi pheromone phân tán và cơ chế học tăng cường.",
                 ["GA", "ACO", "lai ghép", "đột biến", "pheromone"], "multihop"),
            ]
        },
        # Negative / Out-of-Domain Topic (Zero-Hallucination Testing)
        {
            "topic": "NEG",
            "doc": None,
            "questions": [
                ("Thuyết tương đối hẹp của Einstein giải thích vận tốc ánh sáng như thế nào?",
                 "Tài liệu bài giảng không đề cập đến thông tin này.", [], "unanswerable"),
                ("Công thức làm bánh pizza Napoli truyền thống theo chuẩn quốc tế?",
                 "Tài liệu bài giảng không đề cập đến thông tin này.", [], "unanswerable"),
                ("Lịch sử hình thành và phát triển của giải bóng đá Ngoại hạng Anh?",
                 "Tài liệu bài giảng không đề cập đến thông tin này.", [], "unanswerable"),
                ("Phương pháp nuôi cá rồng cảnh trong bể thủy sinh nước ngọt?",
                 "Tài liệu bài giảng không đề cập đến thông tin này.", [], "unanswerable"),
            ]
        }
    ]

    all_samples: List[BenchmarkSample] = []
    sample_counter = 1

    # Base curated samples
    for group in templates:
        for q_text, ans_text, contexts, q_type in group["questions"]:
            is_unans = (q_type == "unanswerable")
            all_samples.append(
                BenchmarkSample(
                    id=f"BM-{sample_counter:03d}",
                    question=q_text,
                    ground_truth_answer=ans_text,
                    ground_truth_contexts=contexts,
                    expected_document_title=group["doc"],
                    question_type=q_type,
                    is_unanswerable=is_unans,
                )
            )
            sample_counter += 1

    # Synthesize variants by variations of questions to expand up to target_size
    base_pool = list(all_samples)
    prefixes = [
        "Hãy giải thích chi tiết: ",
        "Theo tài liệu môn học, ",
        "Cho biết thông tin về: ",
        "Trình bày tóm tắt: ",
        "Trong bài giảng, ",
        "Nêu định nghĩa và ý nghĩa của: ",
        "Tại sao trong bài giảng có đề cập: ",
    ]

    while len(all_samples) < target_size:
        base = random.choice(base_pool)
        prefix = random.choice(prefixes)
        new_q = f"{prefix}{base.question[0].lower() + base.question[1:]}"
        all_samples.append(
            BenchmarkSample(
                id=f"BM-{sample_counter:03d}",
                question=new_q,
                ground_truth_answer=base.ground_truth_answer,
                ground_truth_contexts=base.ground_truth_contexts,
                expected_document_title=base.expected_document_title,
                question_type=base.question_type,
                is_unanswerable=base.is_unanswerable,
            )
        )
        sample_counter += 1

    return all_samples[:target_size]
