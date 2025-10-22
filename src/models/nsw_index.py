import numpy as np
from uuid import UUID

from src.embeddings import get_embeddings_bulk
from src.models.collection import Index
from src.models.datarecord import DataRecord
from src.models.search import SearchResult, SearchResults


class NSWIndex(Index):
    """
    A Navigable Small World (NSW) index for approximate nearest neighbor search.
    This implementation creates a graph where each node is connected to its k nearest
    neighbors, enabling efficient similarity search through graph traversal.
    """

    def __init__(self, n_neighbors: int = 5, ef_construction: int = 100):
        """Initialize the NSW index.

        Args:
            n_neighbors: Number of neighbors to connect for each node during graph construction
            ef_construction: Size of the dynamic candidate list during construction
        """
        self.embeddings: dict[UUID, np.ndarray] = {}  # Store embeddings
        self.graph: dict[UUID, set[UUID]] = {}  # Adjacency list representation
        self.n_neighbors = n_neighbors
        self.ef_construction = ef_construction
        super().__init__()

    def rebuild(self, items: list[DataRecord]):
        """Rebuild the entire index from scratch using provided items."""
        self.embeddings.clear()
        self.graph.clear()

        # First collect all embeddings
        for item in items:
            if hasattr(item, "embedding") and isinstance(item.embedding, list):
                self.embeddings[item.id] = np.array(item.embedding)

        # Then build the graph connections
        for item_id, embedding in self.embeddings.items():
            neighbors = self._find_nearest_neighbors(
                item_id, embedding, self.n_neighbors
            )
            self.graph[item_id] = set(n.id for n in neighbors)

    def add(self, item: DataRecord):
        """Add a single item's embedding to the index and connect it to neighbors."""
        if not hasattr(item, "embedding") or not isinstance(item.embedding, list):
            print(
                f"WARNING: Cannot add item {item.id} to index: no valid 'embedding' attribute found."
            )
            return

        embedding = np.array(item.embedding)
        self.embeddings[item.id] = embedding

        # Find and establish connections with nearest neighbors
        neighbors = self._find_nearest_neighbors(item.id, embedding, self.n_neighbors)
        self.graph[item.id] = set(n.id for n in neighbors)

        # Add bidirectional connections
        for neighbor in neighbors:
            if neighbor.id in self.graph:
                self.graph[neighbor.id].add(item.id)

    def remove(self, item_id: UUID):
        """Remove an item from the index and update graph connections."""
        if item_id in self.embeddings:
            del self.embeddings[item_id]

        # Remove connections to this item from all other items
        if item_id in self.graph:
            for neighbor_id in self.graph[item_id]:
                if neighbor_id in self.graph:
                    self.graph[neighbor_id].discard(item_id)
            del self.graph[item_id]

    def _find_nearest_neighbors(
        self, item_id: UUID, embedding: np.ndarray, k: int
    ) -> SearchResults:
        """Find k nearest neighbors for a given embedding."""
        similarities = []

        for other_id, other_vec in self.embeddings.items():
            if other_id == item_id:
                continue

            if np.linalg.norm(other_vec) == 0:
                continue

            # Calculate cosine similarity
            dot_product = np.dot(embedding, other_vec)
            norm_product = np.linalg.norm(embedding) * np.linalg.norm(other_vec)

            if norm_product == 0:
                confidence = 0.0
            else:
                confidence = dot_product / norm_product

            similarities.append(SearchResult(id=other_id, confidence=confidence))

        # Sort by confidence and return top k
        similarities.sort(key=lambda x: x.confidence or 0.0, reverse=True)
        return similarities[:k]

    def search(self, query: str, limit: int = 5) -> SearchResults:
        """Perform graph-based nearest neighbor search."""
        if not self.embeddings:
            return []

        # Get embedding for query string
        query_vec = np.array(get_embeddings_bulk([query])[0])
        if np.linalg.norm(query_vec) == 0:
            return []

        # Start from a random entry point
        entry_point = next(iter(self.graph.keys()))
        visited = {entry_point}
        candidates = [
            (
                self._cosine_similarity(query_vec, self.embeddings[entry_point]),
                entry_point,
            )
        ]
        best_candidates = []

        # Graph traversal search
        while candidates:
            similarity, current_id = max(candidates, key=lambda x: x[0])
            candidates.remove((similarity, current_id))
            best_candidates.append(SearchResult(id=current_id, confidence=similarity))

            # If we have enough results and the next candidate is worse, stop
            if (
                len(best_candidates) >= limit and similarity < candidates[0][0]
                if candidates
                else True
            ):
                break

            # Explore neighbors
            for neighbor_id in self.graph.get(current_id, set()):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    neighbor_sim = self._cosine_similarity(
                        query_vec, self.embeddings[neighbor_id]
                    )
                    candidates.append((neighbor_sim, neighbor_id))

        return sorted(best_candidates, key=lambda x: x.confidence or 0.0, reverse=True)[
            :limit
        ]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
