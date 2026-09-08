import os
import pickle
import tempfile

import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension):
        """
        Create FAISS index for vector storage and retrieval.

        Args:
            dimension: The dimension of the vectors to be stored.
        """
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []

    def add(self,embeddings,metadata):
        """
        Add embeddings and the corresponding matadata to FAISS index

        Args:
            embeddings:numpy array, shape (n_samples, dimension)
            metadata: list of metadata, length n_samples
        """
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"embeddings must have shape (n, {self.dimension}), got {embeddings.shape}"
            )
        if len(embeddings) != len(metadata):
            raise ValueError("embeddings and metadata must contain the same number of items")
        if len(metadata) == 0:
            return
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must contain only finite values")
        self.index.add(embeddings)
        self.metadata.extend(metadata)
    
    def search(self, query_embedding, top_k=3):
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        query_embedding = np.asarray(query_embedding, dtype="float32")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.shape != (1, self.dimension):
            raise ValueError(
                f"query_embedding must have shape (1, {self.dimension}), got {query_embedding.shape}"
            )
        if not np.isfinite(query_embedding).all():
            raise ValueError("query_embedding must contain only finite values")
        if self.index.ntotal == 0:
            return []

        scores, indices = self.index.search(query_embedding, top_k)

        results=[]

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue
            
            results.append({
                "score": float(score),
                "metadata": self.metadata[index],
            })
        
        return results
    
    def save(self, index_path, metadata_path):
        if self.index.ntotal != len(self.metadata):
            raise RuntimeError("FAISS index and metadata are out of sync")
        index_path = os.fspath(index_path)
        metadata_path = os.fspath(metadata_path)
        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)
        self._atomic_write_index(index_path)
        self._atomic_write_metadata(metadata_path)

    def _atomic_write_index(self, index_path):
        directory = os.path.dirname(os.path.abspath(index_path))
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".faiss", delete=False) as handle:
            temporary_path = handle.name
        try:
            faiss.write_index(self.index, temporary_path)
            os.replace(temporary_path, index_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _atomic_write_metadata(self, metadata_path):
        directory = os.path.dirname(os.path.abspath(metadata_path))
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".pkl", delete=False) as handle:
            temporary_path = handle.name
        try:
            with open(temporary_path, "wb") as file_handle:
                pickle.dump(self.metadata, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary_path, metadata_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
    
    def load(self, index_path, metadata_path):
        index_path = os.fspath(index_path)
        metadata_path = os.fspath(metadata_path)
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"FAISS index does not exist: {index_path}")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"metadata file does not exist: {metadata_path}")

        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as file_handle:
            metadata = pickle.load(file_handle)
        if not isinstance(metadata, list):
            raise ValueError("metadata must be a list")
        if index.ntotal != len(metadata):
            raise ValueError(
                f"index contains {index.ntotal} vectors but metadata contains {len(metadata)} items"
            )

        self.index = index
        self.dimension = index.d
        self.metadata = metadata