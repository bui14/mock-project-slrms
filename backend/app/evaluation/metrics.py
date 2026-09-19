import re
from typing import List


def tokenize(text: str) -> set[str]:
    """Tokenize lowercase words excluding basic punctuation."""
    words = re.findall(r"\b\w+\b", text.lower())
    return set(words)


def is_context_relevant(retrieved_text: str, ground_truth_snippets: List[str]) -> bool:
    """Determine if a retrieved chunk is relevant to the ground truth snippets.
    Uses substring matching and token overlap ratio.
    """
    clean_chunk = retrieved_text.lower().strip()
    chunk_tokens = tokenize(clean_chunk)
    if not chunk_tokens:
        return False

    for snippet in ground_truth_snippets:
        clean_snippet = snippet.lower().strip()
        # Direct substring containment
        if clean_snippet in clean_chunk or clean_chunk in clean_snippet:
            return True

        # Jaccard / Overlap of tokens
        snippet_tokens = tokenize(clean_snippet)
        if not snippet_tokens:
            continue
        overlap = chunk_tokens.intersection(snippet_tokens)
        overlap_ratio = len(overlap) / len(snippet_tokens)
        if overlap_ratio >= 0.5:
            return True

    return False


def compute_hit_at_k(relevance_vector: List[bool], k: int) -> float:
    """Hit Rate @ K: 1.0 if any relevant chunk is in Top K, else 0.0."""
    sub = relevance_vector[:k]
    return 1.0 if any(sub) else 0.0


def compute_mrr(relevance_vector: List[bool]) -> float:
    """Mean Reciprocal Rank: 1 / (rank of first relevant chunk)."""
    for idx, is_rel in enumerate(relevance_vector):
        if is_rel:
            return 1.0 / (idx + 1)
    return 0.0


def compute_context_precision(relevance_vector: List[bool]) -> float:
    """Context Precision (Mean Average Precision on chunks).
    Measures whether relevant chunks are placed at higher ranks.
    """
    total_relevant = sum(1 for r in relevance_vector if r)
    if total_relevant == 0:
        return 0.0

    accumulated_precision = 0.0
    relevant_count = 0

    for k, is_rel in enumerate(relevance_vector, start=1):
        if is_rel:
            relevant_count += 1
            precision_at_k = relevant_count / k
            accumulated_precision += precision_at_k

    return accumulated_precision / total_relevant


def compute_context_recall(
    retrieved_chunks: List[str], ground_truth_snippets: List[str]
) -> float:
    """Context Recall: Ratio of ground-truth snippets covered by at least one retrieved chunk."""
    if not ground_truth_snippets:
        return 1.0

    covered_count = 0
    for snippet in ground_truth_snippets:
        clean_snippet = snippet.lower()
        snippet_tokens = tokenize(clean_snippet)
        covered = False
        for chunk in retrieved_chunks:
            clean_chunk = chunk.lower()
            if clean_snippet in clean_chunk:
                covered = True
                break
            chunk_tokens = tokenize(clean_chunk)
            if snippet_tokens and len(chunk_tokens.intersection(snippet_tokens)) / len(snippet_tokens) >= 0.6:
                covered = True
                break
        if covered:
            covered_count += 1

    return covered_count / len(ground_truth_snippets)


def compute_faithfulness(
    generated_answer: str,
    retrieved_chunks: List[str],
    is_grounded: bool,
    is_unanswerable: bool,
) -> float:
    """Faithfulness (Groundedness):
    Checks if generated answer contains only claims supported by retrieved context.
    For unanswerable queries: refusal gives 1.0, generating claims gives 0.0 (hallucination penalty).
    """
    if is_unanswerable:
        # If unanswerable, faithfulness is 1.0 if system refused or marked not grounded
        if not is_grounded or "không" in generated_answer.lower() or "chưa" in generated_answer.lower():
            return 1.0
        return 0.0

    if not generated_answer.strip():
        return 0.0

    # If the system explicitly flagged the answer as ungrounded
    if not is_grounded:
        return 0.0

    combined_context = " ".join(retrieved_chunks).lower()
    context_tokens = tokenize(combined_context)
    answer_tokens = tokenize(generated_answer)

    if not answer_tokens:
        return 0.0

    # Overlap of informative answer tokens with context
    supported_tokens = answer_tokens.intersection(context_tokens)
    score = len(supported_tokens) / len(answer_tokens)
    return min(1.0, score * 1.2)  # Normalization for function words


def compute_answer_relevance(
    generated_answer: str,
    ground_truth_answer: str,
    is_unanswerable: bool,
) -> float:
    """Answer Relevance:
    Measures semantic/keyword alignment between generated answer and ground truth answer.
    """
    if is_unanswerable:
        # For unanswerable questions, refusal matches ground truth intention
        if "không" in generated_answer.lower() or "chưa" in generated_answer.lower():
            return 1.0
        return 0.0

    ans_tokens = tokenize(generated_answer)
    gt_tokens = tokenize(ground_truth_answer)

    if not ans_tokens or not gt_tokens:
        return 0.0

    intersection = ans_tokens.intersection(gt_tokens)
    union = ans_tokens.union(gt_tokens)

    jaccard = len(intersection) / len(union) if union else 0.0
    # Rescale jaccard to practical 0-1 scale where 0.35+ is strong semantic alignment
    relevance = min(1.0, jaccard * 2.5)
    return relevance


def compute_refusal_correctness(
    generated_answer: str,
    is_grounded: bool,
    is_unanswerable: bool,
) -> float:
    """Refusal Correctness (Zero Hallucination Metric):
    - When is_unanswerable=True: Must refuse -> 1.0 if refused, 0.0 if answered.
    - When is_unanswerable=False: Must answer -> 1.0 if answered, 0.0 if falsely refused.
    """
    refusal_keywords = ["không tìm thấy", "không có thông tin", "tài liệu không", "không được đề cập"]
    has_refusal_phrase = any(kw in generated_answer.lower() for kw in refusal_keywords)

    if is_unanswerable:
        if not is_grounded or has_refusal_phrase:
            return 1.0
        return 0.0
    else:
        if is_grounded and not has_refusal_phrase:
            return 1.0
        return 0.0
