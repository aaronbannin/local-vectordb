from datetime import datetime
from typing import TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

T = TypeVar("T", bound="DataRecord")


class DataRecord(BaseModel):
    """
    Base class for all data records, providing common fields
    like ID, metadata, creation and update timestamps.
    """

    id: UUID = Field(default_factory=uuid4)
    metadata: dict[str, str | int] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Document(DataRecord):
    """
    Represents a document with a name, and metadata, linked to a library.
    Inherits common fields from DataRecord.
    """

    name: str
    library_id: UUID


class Chunk(DataRecord):
    """
    Represents a piece of text with its content, embedding, and metadata, linked to a document.
    Inherits common fields from DataRecord.
    """

    content: str
    embedding: list[float] | None = None
    document_id: UUID


class Library(DataRecord):
    """
    Represents a collection of documents with a name and metadata.
    Inherits common fields from DataRecord.
    """

    name: str
