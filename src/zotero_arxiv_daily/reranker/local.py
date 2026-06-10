from .base import BaseReranker, register_reranker
import logging
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@register_reranker("local")
class LocalReranker(BaseReranker):
    def _get_encode_kwargs(self) -> dict:
        if self.config.reranker.local.encode_kwargs:
            return dict(self.config.reranker.local.encode_kwargs)
        return {}

    def _get_tfidf_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        corpus = [(text or "").strip() for text in [*s1, *s2]]
        if not any(corpus):
            return np.zeros((len(s1), len(s2)))
        try:
            matrix = TfidfVectorizer(stop_words="english").fit_transform(corpus)
        except ValueError:
            return np.zeros((len(s1), len(s2)))
        s1_matrix = matrix[: len(s1)]
        s2_matrix = matrix[len(s1) :]
        return cosine_similarity(s1_matrix, s2_matrix)

    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        if not self.config.executor.debug:
            from transformers.utils import logging as transformers_logging
            from huggingface_hub.utils import logging as hf_logging
    
            transformers_logging.set_verbosity_error()
            hf_logging.set_verbosity_error()
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

        try:
            encoder = SentenceTransformer(self.config.reranker.local.model, trust_remote_code=True)
            encode_kwargs = self._get_encode_kwargs()
            s1_feature = encoder.encode(s1, **encode_kwargs, show_progress_bar=True)
            s2_feature = encoder.encode(s2, **encode_kwargs, show_progress_bar=True)
            sim = encoder.similarity(s1_feature, s2_feature)
            return sim.numpy()
        except Exception as exc:
            logging.warning(
                "Failed to load local embedding model '%s'; falling back to TF-IDF reranking. Error: %s",
                self.config.reranker.local.model,
                exc,
            )
            return self._get_tfidf_similarity_score(s1, s2)
