# Take-at-Home Task – Backend (Vector DB)

Congrats on making it thus far in the interview process!
Here is a task for you to show us where you shine the most ✨

The purpose is not to see how fast you go or what magic tricks you know in Python — it’s mostly to understand how clearly you think and code. 🙂

If you think clearly and your code is clean, you’re already better than 90% of applicants!

> 💡 Feel free to use Cursor, but only where it makes sense. Don’t overuse it — it introduces bugs and is super verbose and not really Pythonic.

---

## ⚙️ Objective

The goal of this project is to develop a **REST API** that allows users to **index** and **query** their documents within a **Vector Database**.

A Vector Database specializes in storing and indexing vector embeddings, enabling fast retrieval and similarity searches.
This capability is crucial for applications involving NLP, recommendation systems, and more.

The REST API should be **containerized in a Docker container**.

---

## 📘 Definitions

To ensure a clear understanding, let’s define some key concepts:

1. **Chunk**: A piece of text with an associated embedding and metadata.
2. **Document**: A collection of multiple chunks and its own metadata.
3. **Library**: A collection of documents and metadata.

---

## 🧩 API Requirements

The API should:

1. Allow users to **create, read, update, and delete (CRUD)** libraries.
2. Allow users to CRUD documents and chunks within a library.
3. **Index** the contents of a library.
4. Perform **k-Nearest Neighbor (kNN)** vector searches over a selected library.

---

## 🐍 Guidelines

The code should be written in **Python**, since that’s our backend language.

Here’s a suggested implementation path:

1. **Define the models:** `Chunk`, `Document`, and `Library`.
   - Use a **fixed schema** for simplicity (fewer validation issues).
   - Optionally, let users define their own schemas for extra credit.
   - Use **Pydantic** for data modeling.

2. **Implement two or three indexing algorithms**.
   - Don’t use external libraries — code them yourself.
   - Explain **space and time complexity** for each.
   - Justify your design choice.

3. **Handle data concurrency** — ensure no data races between reads and writes.
   - Explain your synchronization design choices.

4. **Implement CRUD logic** for libraries, documents, and chunks.
   - Use a **Service layer** to decouple API endpoints from business logic.

5. **Implement the API layer** on top of your service logic.

6. **Containerize the project** using Docker.

---

## 🌟 Extra Points (Optional Enhancements)

1. **Metadata Filtering**
   - Add filters for kNN queries, e.g., search all chunks created after a date or containing a specific keyword.

2. **Persistence to Disk**
   - Persist the DB state to disk so the Docker container can restart and resume.
   - Explain trade-offs (performance, consistency, durability).

3. **Leader–Follower Architecture**
   - Implement leader-follower (master-slave) architecture for multi-node clusters.
   - Handle leader election, data replication, failover, scalability, and availability.

4. **Python SDK Client**
   - Develop a Python SDK for your API with proper documentation and usage examples.

5. **Durable Execution (Bonus with Temporal.io)**
   - Use **Temporal (Python SDK)** for orchestrating query execution workflows.
   - Set up a **local Temporal cluster** with Docker.
   - Design a workflow (`QueryWorkflow`) handling:
     - User query → preprocessing → retrieval → reranking → answer generation.
   - Include:
     - Client connection to Temporal.
     - Worker executing tasks.
     - Demonstrate **sync vs async**, **workflow vs activity**, and bonus **signals/queries**.

---

## 🚫 Constraints

- **Do NOT** use external vector DBs (e.g., ChromaDB, Pinecone, FAISS).
- You **can use** `numpy` for trigonometric functions like `cos`, `sin`, etc.
- You **do not** need to build a document processing pipeline (OCR, chunking, etc.).
  - Using manually created chunks is fine.

---

## 🧱 Tech Stack

- **Backend:** Python + FastAPI + Pydantic
- **Embedding API:** [Cohere Embeddings](https://cohere.com/embeddings)
  - Example API keys for testing:
    ```
    pa6sRhnVAedMVClPAwoCvC1MjHKEwjtcGSTjWRMd
    rQsWxQJOK89Gp87QHo6qnGtPiWerGJOxvdg59o5f
    ```

---

## 🧮 Evaluation Criteria

We’ll evaluate both **functionality** and **code quality**.

### Code Quality
- Static typing
- FastAPI best practices
- Pydantic schema validation
- Code modularity & reusability
- RESTful endpoint design
- Proper Docker containerization
- Testing
- Error handling
- Domain-driven design (DDD) structure:
  - Separate API endpoints from services
  - Separate services from repositories
- Pythonic style:
  - Early returns
  - Use composition over inheritance
  - Avoid hardcoding values (especially HTTP codes — use [FastAPI status codes](https://fastapi.tiangolo.com/reference/status/))

### Functionality
- Does everything work as expected?

---

## 📦 Deliverables

1. **Source Code:**
   - Link to a public GitHub repository with all code.

2. **Documentation:**
   - A `README` explaining:
     - Your approach and design choices
     - How to run the project locally
     - Any additional info

3. **Demo Video:**
   - Show how to install and use the project.
   - Walk through your design and key decisions.

---

## ⏱️ Timeline

- Expected duration: **4 days (96 hours)**.
- However, if you need more time to improve quality, feel free to take it! 🚀

At the end of the day, if it doesn’t impress the team — it won’t fly. So give it your best shot ✈️

---

## 💬 Questions

Feel free to reach out anytime if you encounter issues or have questions about the task.
