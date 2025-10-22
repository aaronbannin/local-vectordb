# Local Vector Database

A REST API that allows users to index and query documents within a Vector Database. Documents are chunked and stored with vector embeddings for efficient similarity search.

## Requirements
- Docker
- Built with Python 3.13

# Running With Makefile

## Building Environment
```bash
# Install dependencies
make install

# Activate virtual environment
source .venv/bin/activate
```

## Running the Server

`dev` target will start the server using Docker Compose. This will run in the foreground.

```bash
make dev
```

The server will be available at `http://localhost:8000`.

## Seeding the Database

The easiest way to get started is with `seed`. This will load sample data into the database by calling it's API endpoints.

```bash
make seed
```

## Making Queries

A simple interactive shell can be used for making queries.

```bash
make query
```

# API
- Documentation can be found at `http://127.0.0.1:8000/docs`.
- Lifecycle endpoints are provided for Libraries, Documents, and Chunks.
- You can query the database using the `/query` endpoint. Here's an example using Python requests:

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "collection": "default",    # Name of the collection to search
        "index_type": "cosine",    # Type of similarity search
        "text": "What is the capital of Germany?",  # Your query text
        "limit": 5                  # Number of results to return
    }
)

# Print results
results = response.json()["results"]
for result in results:
    print(f"Score: {result['score']}")
    print(f"Text: {result['text']}")
    print("---")
```

# Storage and Persistence

Docker Compose mounts the `./data` directory to provide persistent storage on the local file system. Each record is stored as a separate JSON file with the format `{record_id}.json`. This approach offers several benefits:

- **Human Readable**: Data is stored in plain JSON format, making it easy to inspect and modify
- **Natural Uniqueness**: The file system enforces record uniqueness through unique filenames
- **Granular Locking**: A foundation for fine-grained concurrency control - requests can lock only the specific files they need, reducing contention

# Defining and Extending Indexes

The vector database supports custom index implementations through the `Index` base class. To create a new index type:

1. Create a class that inherits from the `Index` class found in `src/models/collection.py`.

2. Implement the required methods. These will be used by the configured `Collection` for accessing the index.
   - `rebuild()`: Build the entire index from a list of items
   - `add()`: Add a single item to the index
   - `remove()`: Remove an item by ID
   - `search()`: Perform the search and return and array of object ids, which point to specific files.

3. To attach the `Index` to a `Collection`, simply use `add_index()` in `src/main.py`:

```python
# The Collection class acts as the context, delegating search to concrete strategies
chunks = Collection(data_dir / "chunks", Chunk)

# Each index type (IVFIndex, NSWIndex, etc.) is a concrete strategy
# that implements the Index interface
chunks.add_index(IndexType.IVF, IVFIndex())
```

The system uses the Strategy Pattern, which allows us to:
- Swap search algorithms at runtime
- Add new index types without modifying existing code
- Keep search algorithm details encapsulated from the Collection class

Note: All indexes are maintained in memory and are rebuilt when the application starts. This means any changes to the underlying data will not persist in the indexes until the application restarts and large datasets will require longer cold start times and scale memory requirements.

## Index Comparison

The system implements three different indexing strategies, each with different space and time complexity tradeoffs:

### 1. Brute Force Cosine Similarity (COSINE)

A simple but effective approach for smaller datasets:

- **Space Complexity**: O(n), where n is number of vectors
  - Stores full vectors in memory
  - No additional index structures needed

- **Time Complexity**:
  - Search: O(n) - compares query against every vector
  - Insert: O(1) - simple dictionary insertion
  - Delete: O(1) - dictionary key removal

- **Best For**:
  - Small to medium datasets
  - When accuracy is critical
  - Development and testing

### 2. Inverted File Index (IVF)

Approximate nearest neighbor search using clustering:

- **Space Complexity**: O(n + k), where:
  - n is number of vectors
  - k is number of centroids

- **Time Complexity**:
  - Search: O(k + m), where m is vectors in closest clusters
  - Insert: O(k) - find nearest cluster
  - Delete: O(k) - remove from cluster

- **Best For**:
  - Large datasets
  - When approximate results are acceptable
  - Trading accuracy for speed

### 3. Navigable Small World (NSW)

Graph-based approximate nearest neighbor search:

- **Space Complexity**: O(n * e), where:
  - n is number of vectors
  - e is edges per node (typically small constant)

- **Time Complexity**:
  - Search: O(log n) average case
  - Insert: O(log n) - find insertion point
  - Delete: O(e) - remove node and reconnect

- **Best For**:
  - Very large datasets
  - Dynamic insertions/deletions
  - When graph traversal matches data relationships

### Selection Guide

- Use **COSINE** for:
  - Development/testing
  - Small collections (<10K vectors)
  - When exact results required

- Use **IVF** for:
  - Medium to large collections
  - Static data
  - Balanced speed/accuracy needs

- Use **NSW** for:
  - Very large collections
  - Frequent updates
  - When approximate results acceptable
