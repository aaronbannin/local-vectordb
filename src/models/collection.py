import json
import numpy as np
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Generic
from uuid import UUID

from src.embeddings import get_embeddings_bulk
from src.models.search import (
    SearchResult,
    SearchResults,
    FullSearchResult,
    FullSearchResults,
)
from src.models.datarecord import T, DataRecord


class IndexType(Enum):
    """Available index types for collections."""

    COSINE = "cosine"  # BruteForceCosineSimilarityIndex for vector similarity searches
    IVF = "ivf"  # IVFIndex for approximate nearest neighbor vector searches
    NSW = "nsw"  # NSWIndex for graph-based approximate nearest neighbor searches


class Collection(Generic[T]):
    """
    A persistent collection that stores each record as a separate JSON file.
    Provides CRUD operations and handles timestamp mutations.
    """

    def __init__(self, storage_path: Path, record_class: type[T]):
        """
        Initialize collection with storage path and record class type.

        Args:
            storage_path: Directory where records will be stored
            record_class: Class to use for instantiating records (e.g. Library, Document)
        """
        self.storage_path: Path = storage_path
        self.record_class: type[T] = record_class
        self.indexes: dict[IndexType, Index] = {}
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def add_index(self, index_type: IndexType, index: "Index") -> None:
        """Add an index to this collection.

        Args:
            index_type: The type of index being added (e.g. HASH, COSINE)
            index: The index instance to add
        """
        self.indexes[index_type] = index
        # Build index with all current items
        index.rebuild(self.list_all())

    def _get_file_path(self, item_id: UUID) -> Path:
        """Get the full file path for an item ID."""
        return self.storage_path / f"{str(item_id)}.json"

    def _load_record(self, file_path: Path) -> T | None:
        """Load and instantiate a record from a JSON file."""
        try:
            with file_path.open("r") as f:
                data = json.load(f)
            return self.record_class.model_validate(data)
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    def _save_record(self, item: T) -> None:
        """Save a record to its JSON file."""
        file_path = self._get_file_path(item.id)
        with file_path.open("w") as f:
            json.dump(item.model_dump(), f, default=str)

    def add(self, item: T) -> T:
        """
        Adds a new item to the collection.
        The item's created_at and updated_at are set by DataRecord's default factories.
        Also updates all indexes.
        """
        print(f"Adding item with ID {item.id} to {self.storage_path}")
        file_path = self._get_file_path(item.id)
        print(f"Writing to file: {file_path}")
        try:
            self._save_record(item)
            if not file_path.exists():
                print(f"WARNING: File {file_path} was not created!")
            else:
                print(f"Successfully created {file_path}")
                # Update all indexes with the new item
                for index in self.indexes.values():
                    index.add(item)
            return item
        except Exception as e:
            print(f"ERROR: Failed to add item {item.id}: {str(e)}")
            raise

    def get(self, item_id: UUID) -> T | None:
        """Retrieves an item by its ID."""
        file_path = self._get_file_path(item_id)
        return self._load_record(file_path)

    def exists(self, item_id: UUID) -> bool:
        """Check if item exists in collection."""
        return self._get_file_path(item_id).exists()

    def update(
        self, item_id: UUID, update_data: dict[str, Any], model: type[T]
    ) -> T | None:
        """
        Updates an existing item with partial data.
        Preserves original 'id' and 'created_at'. Updates 'updated_at'.
        'None' values in update_data are ignored.
        Also updates all indexes.
        """
        existing_item = self.get(item_id)
        if existing_item:
            item_dict = existing_item.model_dump()

            # Apply partial updates
            for key, value in update_data.items():
                if key in ["id", "created_at"]:
                    continue
                if value is not None:
                    item_dict[key] = value

            # Update timestamp
            item_dict["updated_at"] = datetime.utcnow()

            # Create and save updated record
            updated_item = model.model_validate(item_dict)
            self._save_record(updated_item)

            # Update all indexes - remove old item and add updated one
            for index in self.indexes.values():
                index.remove(item_id)
                index.add(updated_item)

            return updated_item
        return None

    def delete(self, item_id: UUID) -> bool:
        """
        Deletes an item by its ID.
        Also updates all indexes.
        """
        file_path = self._get_file_path(item_id)
        try:
            # Remove from all indexes before deleting
            for index in self.indexes.values():
                index.remove(item_id)
            file_path.unlink()
            return True
        except FileNotFoundError:
            return False

    def list_all(self) -> list[T]:
        """Returns a list of all items in the collection."""
        items = []
        for file_path in self.storage_path.glob("*.json"):
            if item := self._load_record(file_path):
                items.append(item)
        return items

    def search(
        self, index_type: IndexType, query: str, limit: int = 5
    ) -> FullSearchResults:
        """Search the collection using the specified index.

        Args:
            index_type: The type of index to search with
            query: The search query
            limit: Maximum number of results to return

        Returns:
            List of FullSearchResult objects containing the full record data, sorted by relevance

        Raises:
            KeyError: If the specified index type doesn't exist
        """
        if index_type not in self.indexes:
            raise KeyError(f"No index of type {index_type.value} exists")

        # Get the initial search results
        search_results = self.indexes[index_type].search(query, limit)

        # Convert to FullSearchResults by looking up each document
        full_results: FullSearchResults = []
        for result in search_results:
            if record := self.get(result.id):
                content = record.content if hasattr(record, "content") else ""
                full_results.append(
                    FullSearchResult(
                        id=result.id,
                        confidence=result.confidence,
                        content=content,
                        # metadata=metadata,
                    )
                )

        return full_results


class Index:
    """
    Base class for collection indexes. Provides fast lookup capabilities
    for various access patterns.
    """

    def __init__(self):
        """Initialize the index."""
        pass

    def rebuild(self, items: list[DataRecord]):
        """Rebuild the entire index from scratch using provided items."""
        raise NotImplementedError()

    def add(self, item: DataRecord):
        """Add a single item to the index."""
        raise NotImplementedError()

    def remove(self, item_id: UUID):
        """Remove a single item from the index."""
        raise NotImplementedError()

    def search(self, query: str, limit: int = 5) -> SearchResults:
        """
        Search the index with a query string.
        Returns list of SearchResult objects sorted by relevance.
        """
        raise NotImplementedError()


class BruteForceCosineSimilarityIndex(Index):
    """
    An index for performing brute-force k-Nearest Neighbor (kNN) searches
    using cosine similarity on item embeddings.
    """

    def __init__(self):
        self.embeddings: dict[UUID, np.ndarray] = {}
        super().__init__()

    def rebuild(self, items: list[DataRecord]):
        """Rebuild the entire index using embeddings from provided items."""
        self.embeddings.clear()
        for item in items:
            if hasattr(item, "embedding") and isinstance(item.embedding, list):
                self.embeddings[item.id] = np.array(item.embedding)
            else:
                print(
                    f"WARNING: Item {item.id} does not have a valid 'embedding' attribute."
                )

    def add(self, item: DataRecord):
        """Add a single item's embedding to the index."""
        if hasattr(item, "embedding") and isinstance(item.embedding, list):
            self.embeddings[item.id] = np.array(item.embedding)
        else:
            print(
                f"WARNING: Cannot add item {item.id} to index: no valid 'embedding' attribute found."
            )

    def remove(self, item_id: UUID):
        """Remove an item's embedding from the index by its ID."""
        if item_id in self.embeddings:
            del self.embeddings[item_id]

    def search(self, query: str, limit: int = 5) -> SearchResults:
        """
        Performs a k-Nearest Neighbor search using cosine similarity.

        Args:
            query: The query string to search for.
            limit: The number of nearest neighbors to return.

        Returns:
            A list of SearchResult objects sorted by confidence (cosine similarity)
            in descending order.
        """

        if not self.embeddings:
            print("no embeddings in index")
            return []

        # Get embedding for query string
        query_vec = np.array(get_embeddings_bulk([query])[0])
        if np.linalg.norm(query_vec) == 0:
            return []  # Avoid division by zero if query vector is zero

        similarities = []
        for item_id, item_vec in self.embeddings.items():
            if np.linalg.norm(item_vec) == 0:
                continue  # Skip items with zero-norm embeddings

            # Calculate cosine similarity
            # cos_sim = (A . B) / (||A|| * ||B||)
            dot_product = np.dot(query_vec, item_vec)
            norm_product = np.linalg.norm(query_vec) * np.linalg.norm(item_vec)

            if norm_product == 0:
                confidence = 0.0
            else:
                confidence = dot_product / norm_product

            similarities.append(SearchResult(id=item_id, confidence=confidence))

        # Sort by confidence in descending order and return top k
        similarities.sort(key=lambda x: x.confidence or 0.0, reverse=True)
        return similarities[:limit]
