"""Inverted File (IVF) Index implementation for faster approximate nearest neighbor search."""

import numpy as np
from typing import Dict, List, Optional
from uuid import UUID
from sklearn.cluster import KMeans

from src.embeddings import get_embeddings_bulk
from src.models.collection import Index
from src.models.datarecord import Chunk, DataRecord
from src.models.search import SearchResult


class IVFIndex(Index):
    """
    An Inverted File Index (IVF) for approximate nearest neighbor search.

    The IVF index divides the vector space into k clusters and assigns each embedding
    to its nearest cluster centroid. During search, we:
    1. Find the nearest cluster(s) to the query vector
    2. Only compute distances to vectors in those clusters

    This provides a significant speedup over brute force search at the cost of
    some accuracy, since we may miss vectors that are close to the query but
    happen to be in different clusters.
    """

    def __init__(self, n_clusters: int = 100):
        """
        Initialize the IVF index.

        Args:
            n_clusters: Number of clusters to use. More clusters = faster search but less accurate
                       and more memory usage. Default 100.
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.kmeans: Optional[KMeans] = None
        self.cluster_assignments: Dict[int, List[UUID]] = (
            {}
        )  # cluster_id -> [vector_ids]
        self.embeddings: Dict[UUID, np.ndarray] = {}  # id -> embedding vector

    def rebuild(self, items: list[DataRecord]):
        """
        Rebuild the entire index from scratch using provided items.

        Args:
            items: List of DataRecord objects with embeddings
        """
        # Clear existing index
        self.cluster_assignments = {}
        self.embeddings = {}

        # Filter out items without embeddings
        valid_items = [
            (item.id, item.embedding)
            for item in items
            if isinstance(item, Chunk) and item.embedding is not None
        ]

        if not valid_items:
            return

        # Split into IDs and embeddings
        ids, embeddings = zip(*valid_items)
        embeddings_array = np.array(embeddings)

        # Train KMeans on embeddings
        n_clusters = min(
            self.n_clusters, len(embeddings)
        )  # Can't have more clusters than points
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_ids = self.kmeans.fit_predict(embeddings_array)

        # Build cluster assignments and embeddings lookup
        for i, (item_id, embedding) in enumerate(zip(ids, embeddings_array)):
            cluster_id = cluster_ids[i]
            if cluster_id not in self.cluster_assignments:
                self.cluster_assignments[cluster_id] = []
            self.cluster_assignments[cluster_id].append(item_id)
            self.embeddings[item_id] = embedding

    def add(self, item: DataRecord):
        """
        Add a single item to the index.

        Args:
            item: DataRecord object with an embedding to add
        """
        if not isinstance(item, Chunk) or item.embedding is None:
            return

        embedding = np.array(item.embedding).reshape(1, -1)

        # Initialize KMeans if needed
        if self.kmeans is None:
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            self.kmeans.fit(embedding)  # Start with just this point

        # Get cluster assignment for new point
        cluster_id = self.kmeans.predict(embedding)[0]

        # Add to cluster assignments
        if cluster_id not in self.cluster_assignments:
            self.cluster_assignments[cluster_id] = []
        self.cluster_assignments[cluster_id].append(item.id)

        # Store embedding
        self.embeddings[item.id] = embedding.reshape(-1)

    def remove(self, item_id: UUID):
        """
        Remove a single item from the index.

        Args:
            item_id: UUID of item to remove
        """
        if item_id not in self.embeddings:
            return

        # Find and remove from cluster assignments
        for cluster_ids in self.cluster_assignments.values():
            if item_id in cluster_ids:
                cluster_ids.remove(item_id)
                break

        # Remove from embeddings
        del self.embeddings[item_id]

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """
        Search the index with a query string.
        Returns list of SearchResult objects sorted by relevance.

        Args:
            query: Query string to search for
            limit: Maximum number of results to return

        Returns:
            List of SearchResult objects sorted by cosine similarity (highest first)
        """
        if not query.strip() or not self.kmeans or not self.embeddings:
            return []

        # Get query embedding
        query_embedding = get_embeddings_bulk([query])[0]
        query_vector = np.array(query_embedding)

        # Find nearest cluster(s)
        n_probe = min(
            3, len(self.cluster_assignments)
        )  # Search 3 nearest clusters by default
        nearest_clusters = self.kmeans.predict(query_vector.reshape(1, -1))
        cluster_dists = np.linalg.norm(
            self.kmeans.cluster_centers_ - query_vector, axis=1
        )
        nearest_clusters = np.argsort(cluster_dists)[:n_probe]

        # Gather candidate vectors from nearest clusters
        candidates = []
        for cluster_id in nearest_clusters:
            if cluster_id in self.cluster_assignments:
                candidates.extend(self.cluster_assignments[cluster_id])

        if not candidates:
            return []

        # Compute similarities for candidates
        results = []
        for item_id in candidates:
            if item_id in self.embeddings:
                vector = self.embeddings[item_id]
                # Compute cosine similarity
                similarity = np.dot(query_vector, vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(vector)
                )
                results.append(SearchResult(id=item_id, confidence=float(similarity)))

        # Sort by similarity and return top k
        results.sort(key=lambda x: x.confidence or 0.0, reverse=True)
        return results[:limit]
