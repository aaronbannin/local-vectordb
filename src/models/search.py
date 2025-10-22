from dataclasses import dataclass
from typing import TypeAlias
from uuid import UUID


@dataclass
class SearchResult:
    """
    Represents a basic search result with an ID and optional confidence score.
    Used by both HashIndex and BruteForceCosineSimilarityIndex.
    """

    id: UUID
    confidence: float | None = None


@dataclass
class FullSearchResult:
    """
    Represents a complete search result including the document content and metadata.
    Extends SearchResult to include the full document information.
    """

    id: UUID
    content: str
    confidence: float | None = None


# Type aliases for lists of search results
SearchResults: TypeAlias = list[SearchResult]
FullSearchResults: TypeAlias = list[FullSearchResult]
