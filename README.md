# Adaptive RAG

Adaptive RAG is a deployable document-chat application with three answer paths:

1. **Index** — Qdrant retrieval, LLM reranking, grounded generation, and faithfulness verification.
2. **General** — direct LLM response for stable common knowledge and conversation.
3. **Search** — Tavily search followed by cited, grounded generation and verification.

Authentication is intentionally not enabled. A random workspace/session ID partitions Qdrant documents
and MongoDB history, but it is not a security boundary.

## Architecture

```text
Streamlit → FastAPI → query classifier
                         ├─ index: Qdrant → rerank → generate → verify
                         ├─ general: GPT-4o
                         └─ search: Tavily → generate → verify

MongoDB: bounded conversation history
Qdrant: persistent vectors, filtered by session_id
```

The index path retrieves 12 vector candidates by default, reranks them with structured LLM output,
keeps the best 5, and only generates when the best rerank score passes the configured threshold.
One rewritten-query retry is allowed. Failed retrieval then falls back to web search. An unsupported
generated answer is regenerated once and then replaced with a safe failure response.

## Quick start with Docker Compose

Requirements:

- Docker Engine with Docker Compose v2
- An OpenAI API key
- A Tavily API key

```bash
cp .env.example .env
# Edit .env and replace every replace-me/replace-with value.
docker compose up --build -d
docker compose ps
```

Open:

- UI: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`
- Readiness: `http://localhost:8000/health/ready`

MongoDB and Qdrant are intentionally reachable only through the internal Compose network. Their data
is stored in the `mongo_data` and `qdrant_data` named volumes.

## Environment variables

Copy [.env.example](.env.example). Required values are:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Chat, structured routing/reranking, verification, embeddings |
| `TAVILY_API_KEY` | Web search |
| `QDRANT_URL` | Qdrant REST endpoint |
| `QDRANT_API_KEY` | Qdrant service or cloud key |
| `QDRANT_COLLECTION` | Shared, session-partitioned collection |
| `MONGODB_URL` | MongoDB connection string |
| `MONGODB_DB_NAME` | Chat database |
| `MONGO_ROOT_USERNAME` / `MONGO_ROOT_PASSWORD` | Local Compose Mongo initialization |
| `RAG_API_URL` | Backend URL used by Streamlit |

`OPENAI_EMBEDDING_MODEL` and `EMBEDDING_DIMENSIONS` must match the existing Qdrant collection. If you
change either, use a new `QDRANT_COLLECTION` name or migrate/re-embed the existing vectors.

## Running services without Docker

Use Python 3.10 or newer and run MongoDB and Qdrant separately. For host-based services, change the
Docker hostnames in `.env` to `localhost`:

```env
QDRANT_URL=http://localhost:6333
MONGODB_URL=mongodb://localhost:27017
RAG_API_URL=http://127.0.0.1:8000
```

Then:

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload
```

In another terminal:

```bash
streamlit run streamlit_app/home.py
```

## Qdrant Cloud

Create a Qdrant Cloud cluster and set:

```env
QDRANT_URL=https://your-cluster-url
QDRANT_API_KEY=your-cloud-api-key
QDRANT_COLLECTION=adaptive_rag_documents
```

The API creates the collection and keyword payload indexes for `session_id` and `document_id` on
startup. Do not create one collection per user; all sessions share one collection and use payload
filtering.

## API

### Query

```http
POST /rag/query
Content-Type: application/json

{"query":"What does the handbook say?","session_id":"workspace-123"}
```

The response contains `content`, `route`, and `sources`.

### Upload

```http
POST /rag/documents/upload
X-Session-ID: workspace-123
X-Description: Product handbook
Content-Type: multipart/form-data
```

PDF and UTF-8 TXT are accepted. The default maximum is 20 MB. Uploads are additive and return a
`document_id`.

### Delete document

```http
DELETE /rag/documents/{document_id}
X-Session-ID: workspace-123
```

### Clear chat history

```http
DELETE /rag/history
X-Session-ID: workspace-123
```

## Tests and quality checks

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check src tests streamlit_app
ruff format --check src tests streamlit_app
```

## Production deployment

For a single server, install Docker, copy the repository and `.env`, then run `docker compose up
--build -d`. Put a TLS reverse proxy or managed load balancer in front of ports 8501 and optionally
8000. Do not expose MongoDB or Qdrant publicly.

For a managed deployment:

- Run API and Streamlit images on a container platform.
- Use Qdrant Cloud and MongoDB Atlas.
- Store all secrets in the platform secret manager, not in an image or repository.
- Point `QDRANT_URL`, `MONGODB_URL`, and `RAG_API_URL` at the managed services.
- Add authentication before storing sensitive or private documents.

The API is stateless apart from MongoDB and Qdrant, so replicas can share the same backing services.
