from pydantic import BaseModel, Field
from typing import Literal
from uuid import UUID

from src.models.datarecord import Chunk
from src.models.collection import IndexType
from src.models.search import SearchResults, FullSearchResults


class CreateChunkRequest(BaseModel):
    """
    Schema for creating a Chunk, excluding auto-generated fields like ID and timestamps.
    """

    content: str
    embedding: list[float] | None = None
    metadata: dict[str, str | int] = Field(default_factory=dict)
    document_id: UUID


class UpdateChunkRequest(BaseModel):
    """
    Schema for updating a Chunk, allowing partial updates.
    """

    content: str | None = None
    embedding: list[float] | None = None
    metadata: dict[str, str | int] | None = None
    document_id: UUID | None = None


class UpdateLibraryRequest(BaseModel):
    """
    Schema for updating a Library, allowing partial updates to name and metadata.
    """

    name: str | None = None
    metadata: dict[str, str] | None = None


class QueryRequest(BaseModel):
    """
    Schema for querying chunks within a collection.
    """

    collection: Literal["chunks", "documents", "libraries"]
    index_type: IndexType
    text: str
    limit: int
    metadata: dict[str, str | int] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """
    Schema for query results containing matching chunks.
    """

    results: FullSearchResults
