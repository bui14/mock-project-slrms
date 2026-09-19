import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False


class CrossEncoderReranker:
    """Lightweight Cross-Encoder Reranker using FlashRank (ONNX, CPU-optimized, Zero GPU required).
    Falls back gracefully to lexical-semantic scoring if FlashRank model fails to load.
    """

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2", cache_dir: str = "backend/cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.ranker = None
        if FLASHRANK_AVAILABLE:
            try:
                self.ranker = Ranker(model_name=model_name, cache_dir=cache_dir)
                logger.info("FlashRank Cross-Encoder initialized with model %s", model_name)
            except Exception as exc:
                logger.warning("Failed to initialize FlashRank: %s. Using lexical fallback.", exc)

    def rerank(self, query: str, passages: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank candidate passages for a given query.
        :param query: user question string
        :param passages: list of dicts with at least 'id' and 'text'
        :param top_k: number of passages to return after reranking
        :return: sorted list of passages with 'score' key
        """
        if not passages:
            return []

        if self.ranker is not None:
            try:
                req = RerankRequest(query=query, passages=passages)
                results = self.ranker.rerank(req)
                return results[:top_k]
            except Exception as exc:
                logger.warning("Reranking failed: %s, falling back to lexical scoring", exc)

        # Fallback lexical scoring if ranker unavailable
        q_tokens = set(query.lower().split())
        scored = []
        for p in passages:
            text = p.get("text", "")
            t_tokens = set(text.lower().split())
            overlap = len(q_tokens.intersection(t_tokens))
            score = overlap / (len(q_tokens) or 1)
            scored.append({**p, "score": float(score)})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
