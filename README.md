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

## Queries with the API
You can query the database using the `/query` endpoint. Here's an example using Python requests:

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

3. The system uses the Strategy Pattern to make search algorithms interchangeable:

```python
# The Collection class acts as the context, delegating search to concrete strategies
chunks = Collection(data_dir / "chunks", Chunk)

# Each index type (IVFIndex, NSWIndex, etc.) is a concrete strategy
# that implements the Index interface
chunks.add_index(IndexType.IVF, IVFIndex())
```

This pattern allows us to:
- Swap search algorithms at runtime
- Add new index types without modifying existing code
- Keep search algorithm details encapsulated from the Collection class

Note: All indexes are maintained in memory and are rebuilt when the application starts. This means any changes to the underlying data will not persist in the indexes until the application restarts and large datasets will require longer cold start times and scale memory requirements.
