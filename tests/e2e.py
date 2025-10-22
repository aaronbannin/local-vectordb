import pandas as pd
import requests
import uuid
from datasets import load_dataset

from src.models.collection import IndexType

# Base URL for the API
BASE_URL = "http://localhost:8000"


def reset_db():
    response = requests.post(f"{BASE_URL}/reset")
    response.raise_for_status()
    return response.json()


def create_library():
    """Create a new library for testing"""
    library_id = str(uuid.uuid4())
    library_data = {
        "id": library_id,
        "name": "TREC Dataset Library",
        "description": "Test library containing TREC dataset questions",
        "metadata": {"source": "TREC", "type": "questions"},
    }

    response = requests.post(f"{BASE_URL}/libraries", json=library_data)
    response.raise_for_status()
    return response.json()


def list_libraries():
    """
    Connects to the FastAPI application and lists all available libraries.
    """
    url = f"{BASE_URL}/libraries"
    print(f"Attempting to connect to {url} to list libraries...")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        libraries = response.json()

        if libraries:
            print("Successfully listed libraries:")
            for library in libraries:
                print(f"  - {library.get('id')} (Name: {library.get('name')})")
        else:
            print("No libraries found.")

    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to the API. Is it running at {BASE_URL}?")
        print("Please ensure your FastAPI application is running and accessible.")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error occurred: {e}")
        print(f"Response content: {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"An unexpected request error occurred: {e}")
    except ValueError:
        print(
            "Error: Could not decode JSON response. The API might not be returning valid JSON."
        )
    except Exception as e:
        print(f"An unhandled error occurred: {e}")


def populate_library(library_id, df):
    """Populate library with data from DataFrame"""
    for idx, row in df.iterrows():
        # Create a document for each question
        document_id = str(uuid.uuid4())
        document_data = {
            "id": document_id,
            "library_id": library_id,
            "name": f"Question {idx}",
            "metadata": {
                "coarse_label": row["coarse_label"],
                "fine_label": row["fine_label"],
            },
        }

        # Create document
        response = requests.post(f"{BASE_URL}/documents", json=document_data)
        response.raise_for_status()

        # Create chunk with the question text
        chunk_data = {
            "document_id": document_id,
            "content": row["text"],
            "metadata": {"type": "question"},
        }

        # Create chunk
        response = requests.post(f"{BASE_URL}/chunks", json=chunk_data)
        response.raise_for_status()


def load():
    # Load dataset
    dataset = load_dataset("trec", split="train", trust_remote_code=True)
    df = pd.DataFrame(dataset)[:100]
    print("Loading into database; sample:")
    print(df)

    # Create library and populate it
    library = create_library()
    populate_library(library["id"], df)
    print(f"Successfully created and populated library {library['id']}")


def query(query_text=None):
    """Query chunks in a library"""
    # Get the first available library
    libraries = requests.get(f"{BASE_URL}/libraries").json()
    if not libraries:
        print("No libraries found to query")
        return

    library_id = libraries[0]["id"]

    # If no query text provided, make it interactive
    if query_text is None:
        # List available index types from the enum
        print("\nAvailable index types:")
        index_choices = list(IndexType)
        for i, index_type in enumerate(index_choices, 1):
            print(f"{i}. {index_type.name} ({index_type.value})")

        while True:
            try:
                choice = int(
                    input(f"\nSelect index type (1-{len(index_choices)}): ").strip()
                )
                if 1 <= choice <= len(index_choices):
                    selected_index = index_choices[choice - 1]
                    break
                print(f"Please select a number between 1 and {len(index_choices)}")
            except ValueError:
                print("Please enter a valid number")

        # Get query text from user
        query_text = input("\nEnter your query: ").strip()
        while not query_text:
            print("Query cannot be empty.")
            query_text = input("Enter your query: ").strip()
    else:
        selected_index = (
            IndexType.IVF
        )  # Default to IVF when query is provided via command line

    # Example query request
    query_data = {
        "collection": "chunks",
        "index_type": selected_index.value,
        "text": query_text,
        "limit": 5,
        "metadata": {"type": "question"},
    }

    response = requests.post(f"{BASE_URL}/query", json=query_data)
    response.raise_for_status()
    results = response.json()
    print("\nQuery results:")
    for i, result in enumerate(results["results"], 1):
        print(f"\n{i}. Match (confidence: {result['confidence']:.2f})")
        print(f"   ID: {result['id']}")
        print(f"   Content: {result['content']}")
    return results
