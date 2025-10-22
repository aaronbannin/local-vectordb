import os
import shutil
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException, status

from src.models.api import (
    CreateChunkRequest,
    UpdateChunkRequest,
    UpdateLibraryRequest,
    QueryRequest,
    QueryResponse,
)
from src.models.collection import (
    Collection,
    BruteForceCosineSimilarityIndex,
    IndexType,
)
from src.models.ivf_index import IVFIndex
from src.models.nsw_index import NSWIndex
from src.models.datarecord import Chunk, Document, Library
from src.embeddings import get_embeddings_bulk


app = FastAPI(
    title="Vector Database API",
    description="API for indexing and querying documents in a Vector Database",
    version="0.1.0",
)

# Use absolute path to project root's data directory
data_dir = Path(os.getenv("DATA_DIR", str(Path(__file__).parent.parent / "data")))
chunks = Collection(data_dir / "chunks", Chunk)
chunks_brute = BruteForceCosineSimilarityIndex()
chunks.add_index(IndexType.COSINE, chunks_brute)
chunks_ivf = IVFIndex()
chunks.add_index(IndexType.IVF, chunks_ivf)
chunks_nsw = NSWIndex()
chunks.add_index(IndexType.NSW, chunks_nsw)
documents = Collection(data_dir / "documents", Document)
libraries = Collection(data_dir / "libraries", Library)

collections = {"chunks": chunks, "documents": documents, "libraries": libraries}


def init_collections(
    clean: bool = True, data_dir: Path | None = None
) -> tuple[Collection[Chunk], Collection[Document], Collection[Library]]:
    """
    Initialize collections for seeding, optionally cleaning existing data.

    Args:
        clean: Whether to clean existing data before initializing
        data_dir: Base directory for storing collection data
    """
    _data_dir = data_dir or Path("data").absolute()
    if clean:
        print(f"cleaning data directory: {_data_dir}")
        # Force remove the entire data directory if it exists
        if _data_dir.exists():
            print(f"force removing {_data_dir}")
            try:
                shutil.rmtree(_data_dir, ignore_errors=True)
            except Exception as e:
                print(f"Error removing directory: {e}")

        # Create fresh collection directories
        for subdir in ["chunks", "documents", "libraries"]:
            path = _data_dir / subdir
            print(f"creating {path}")
            path.mkdir(parents=True, exist_ok=True)

    # Initialize collections
    chunks = Collection(_data_dir / "chunks", Chunk)
    documents = Collection(_data_dir / "documents", Document)
    libraries = Collection(_data_dir / "libraries", Library)

    return chunks, documents, libraries


def _reset():
    """Reset collections and optionally load seed data."""
    global chunks, documents, libraries

    # First clean the data directories
    chunks, documents, libraries = init_collections(clean=True, data_dir=data_dir)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Welcome to the Vector Database API!"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reset", status_code=status.HTTP_200_OK)
async def reset() -> dict[str, str]:
    """Reset all collections to a clean state."""
    _reset()
    return {"message": "Collections reset successfully"}


# Library Endpoints
@app.post("/libraries", response_model=Library, status_code=status.HTTP_201_CREATED)
async def create_library(library: Library) -> Library:
    if libraries.exists(library.id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Library with this ID already exists",
        )

    new_record = libraries.add(library)
    return new_record


@app.get("/libraries", response_model=list[Library])
async def get_all_libraries() -> list[Library]:
    return libraries.list_all()


@app.get("/libraries/{library_id}", response_model=Library)
async def get_library(library_id: UUID) -> Library:
    library = libraries.get(library_id)
    if not library:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Library not found"
        )
    return library


@app.put("/libraries/{library_id}", response_model=Library)
async def update_library(
    library_id: UUID, updated_library: UpdateLibraryRequest
) -> Library:
    update_data = updated_library.model_dump(exclude_unset=True)
    library = libraries.update(library_id, update_data, Library)
    if not library:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Library not found"
        )
    return library


@app.delete("/libraries/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_library(library_id: UUID) -> None:
    if not libraries.delete(library_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Library not found"
        )


# Document Endpoints
@app.post("/documents", response_model=Document, status_code=status.HTTP_201_CREATED)
async def create_document(document: Document) -> Document:
    if not libraries.exists(document.library_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Parent Library not found"
        )
    if documents.exists(document.id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document with this ID already exists",
        )

    new_record = documents.add(document)
    return new_record


@app.get("/documents", response_model=list[Document])
async def get_all_documents() -> list[Document]:
    return documents.list_all()


@app.get("/documents/{document_id}", response_model=Document)
async def get_document(document_id: UUID) -> Document:
    document = documents.get(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )
    return document


@app.put("/documents/{document_id}", response_model=Document)
async def update_document(document_id: UUID, updated_document: Document) -> Document:
    if not libraries.exists(updated_document.library_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Parent Library not found"
        )

    update_data = updated_document.model_dump(exclude_unset=True)
    document = documents.update(document_id, update_data, Document)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )
    return document


@app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: UUID) -> None:
    if not documents.delete(document_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )
    # Also delete all chunks associated with this document
    chunks_to_delete = [
        chunk.id for chunk in chunks.list_all() if chunk.document_id == document_id
    ]
    for chunk_id in chunks_to_delete:
        chunks.delete(chunk_id)


# Chunk Endpoints
@app.post("/chunks", response_model=Chunk, status_code=status.HTTP_201_CREATED)
async def create_chunk(chunk_request: CreateChunkRequest) -> Chunk:
    document = documents.get(chunk_request.document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Parent Document not found"
        )
    if not libraries.exists(
        document.library_id
    ):  # Check if parent library exists for integrity
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Parent Library not found for this document",
        )

    # Generate embedding if not provided
    embedding = chunk_request.embedding
    if embedding is None:
        embeddings_result = get_embeddings_bulk([chunk_request.content])
        embedding = list(embeddings_result[0])

    # Create the Chunk instance with all required fields
    chunk = Chunk(
        content=chunk_request.content,
        embedding=embedding,
        metadata=chunk_request.metadata,
        document_id=chunk_request.document_id,
    )

    new_record = chunks.add(chunk)
    return new_record


@app.get("/chunks", response_model=list[Chunk])
async def get_all_chunks() -> list[Chunk]:
    return chunks.list_all()


@app.get("/chunks/{chunk_id}", response_model=Chunk)
async def get_chunk(chunk_id: UUID) -> Chunk:
    chunk = chunks.get(chunk_id)
    if not chunk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found"
        )
    return chunk


@app.put("/chunks/{chunk_id}", response_model=Chunk)
async def update_chunk(chunk_id: UUID, update_request: UpdateChunkRequest) -> Chunk:
    existing_chunk = chunks.get(chunk_id)
    if not existing_chunk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found"
        )

    # Validate parent document and library existence if document_id is being changed
    document_id = (
        update_request.document_id
        if update_request.document_id is not None
        else existing_chunk.document_id
    )
    document = documents.get(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Parent Document not found",
        )
    if not libraries.exists(document.library_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Parent Library not found for this document",
        )

    update_data = update_request.model_dump(exclude_unset=True)

    # If content is being updated and no embedding is provided in the request,
    # generate a new embedding for the updated content.
    if "content" in update_data and update_request.embedding is None:
        embeddings_result = get_embeddings_bulk([update_data["content"]])
        update_data["embedding"] = list(embeddings_result[0])

    chunk = chunks.update(chunk_id, update_data, Chunk)
    if not chunk:  # This case should theoretically not be hit if existing_chunk was found, but good for robustness.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chunk not found after update attempt",
        )
    return chunk


@app.delete("/chunks/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chunk(chunk_id: UUID) -> None:
    if not chunks.delete(chunk_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found"
        )


@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query(query_request: QueryRequest) -> QueryResponse:
    if query_request.collection not in collections:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{query_request.collection}' not found",
        )

    collection = collections[query_request.collection]
    if query_request.index_type not in collection.indexes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{query_request.index_type} has not been configured for {query_request.collection}",
        )

    response = collection.search(
        index_type=query_request.index_type,
        query=query_request.text,
        limit=query_request.limit,
    )

    return QueryResponse(results=response)
