from collections.abc import Mapping

from src.embedding import EmbeddingModel
from src.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        index_path,
        metadata_path,
        similarity_threshold=0.5,
        embedding_model=None,
        vector_store=None,
        reranker=None,
        rerank_multiplier=5,
    ):
        """Retrieve relevant chunks while preserving their metadata.

        ``embedding_model`` and ``vector_store`` are injectable so callers can
        test retrieval logic without loading large models or FAISS files.
        """
        if not isinstance(similarity_threshold, (int, float)):
            raise TypeError("similarity_threshold must be a number")
        if similarity_threshold < 0:
            raise ValueError("similarity_threshold must be non-negative")
        if (
            not isinstance(rerank_multiplier, int)
            or isinstance(rerank_multiplier, bool)
            or rerank_multiplier <= 0
        ):
            raise ValueError("rerank_multiplier must be a positive integer")

        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or VectorStore(1)
        self.vector_store.load(index_path, metadata_path)
        self.similarity_threshold = similarity_threshold
        self.reranker = reranker
        self.rerank_multiplier = rerank_multiplier

    @staticmethod
    def _validate_options(query, top_k, adjacent_window):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if (
            not isinstance(adjacent_window, int)
            or isinstance(adjacent_window, bool)
            or adjacent_window < 0
        ):
            raise ValueError("adjacent_window must be a non-negative integer")

    @staticmethod
    def _matches_filter(metadata, metadata_filter):
        if metadata_filter is None:
            return True
        if not isinstance(metadata_filter, Mapping):
            raise TypeError("metadata_filter must be a mapping")
        return all(metadata.get(key) == value for key, value in metadata_filter.items())

    @staticmethod
    def _identity(metadata):
        chunk_id = metadata.get("chunk_id")
        if chunk_id:
            return chunk_id

        source = metadata.get("source")
        chunk_index = metadata.get("chunk_index")
        if source is not None and chunk_index is not None:
            return source, chunk_index

        return metadata.get("text")

    def _adjacent_chunks(self, metadata, adjacent_window, metadata_filter):
        """Return nearby chunks from the same source in document order."""
        if adjacent_window == 0:
            return []

        source = metadata.get("source")
        chunk_index = metadata.get("chunk_index")
        if source is None or not isinstance(chunk_index, int):
            return []

        adjacent = []
        for candidate in self.vector_store.metadata:
            if candidate.get("source") != source:
                continue
            candidate_index = candidate.get("chunk_index")
            if not isinstance(candidate_index, int):
                continue
            if abs(candidate_index - chunk_index) > adjacent_window:
                continue
            if candidate_index == chunk_index:
                continue
            if not self._matches_filter(candidate, metadata_filter):
                continue
            adjacent.append(candidate)
        return sorted(adjacent, key=lambda item: item["chunk_index"])

    def retrieve(
        self,
        query,
        top_k=10,
        metadata_filter=None,
        adjacent_window=1,
        deduplicate=True,
    ):
        """Retrieve ranked chunks with optional metadata filtering and context.

        Args:
            query: Non-empty natural-language query.
            top_k: Maximum number of semantic matches before adjacent expansion.
            metadata_filter: Exact metadata predicates, e.g. ``{"source": ...}``.
            adjacent_window: Number of neighboring chunks per semantic match.
            deduplicate: Remove repeated chunks by stable identity.
        """
        self._validate_options(query, top_k, adjacent_window)
        if not isinstance(deduplicate, bool):
            raise TypeError("deduplicate must be a boolean")
        if metadata_filter is not None and not isinstance(metadata_filter, Mapping):
            raise TypeError("metadata_filter must be a mapping")

        query_embedding = self.embedding_model.encode([query])

        candidate_count = max(top_k, top_k * (2 * adjacent_window + 1))
        if self.reranker is not None:
            candidate_count = max(candidate_count, top_k * self.rerank_multiplier)
        raw_results = self.vector_store.search(
            query_embedding,
            top_k=candidate_count,
        )

        semantic_results = []
        seen = set()
        semantic_limit = candidate_count if self.reranker is not None else top_k
        for result in raw_results:
            score = result.get("score")
            metadata = result.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            if not isinstance(score, (int, float)):
                continue
            if score < self.similarity_threshold:
                continue
            if not self._matches_filter(metadata, metadata_filter):
                continue

            identity = self._identity(metadata)
            if deduplicate and identity in seen:
                continue
            seen.add(identity)
            semantic_results.append({
                **result,
                "metadata": dict(metadata),
                "retrieval_type": "semantic",
            })
            if len(semantic_results) >= semantic_limit:
                break

        if self.reranker is not None:
            semantic_results = self.reranker.rerank(query, semantic_results, top_k)

        results = list(semantic_results)
        if adjacent_window:
            for result in semantic_results:
                parent_id = self._identity(result["metadata"])
                for metadata in self._adjacent_chunks(
                    result["metadata"], adjacent_window, metadata_filter
                ):
                    identity = self._identity(metadata)
                    if deduplicate and identity in seen:
                        continue
                    seen.add(identity)
                    results.append({
                        "score": result["score"],
                        "metadata": dict(metadata),
                        "retrieval_type": "adjacent",
                        "parent_chunk_id": parent_id,
                    })

        for rank, result in enumerate(results, start=1):
            result["rank"] = rank
        return results

    def retriever(self, query, top_k=10):
        """Backward-compatible alias for the original public method."""
        return self.retrieve(query=query, top_k=top_k)