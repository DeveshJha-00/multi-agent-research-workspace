# Multi-Agent Research Workspace

An educational, demo-oriented AI workspace that combines adaptive chat routing, retrieval-augmented
generation, live web research, multi-agent orchestration, dataset analysis, evidence auditing, and
downloadable reports.

The application has a Streamlit UI and FastAPI backend. Groq supplies chat-model inference, while
embeddings and reranking run locally with ONNX models. Qdrant stores vectors and MongoDB stores chat
history, datasets, evidence, and generated artifacts.

Authentication is intentionally not implemented yet. Workspace IDs partition records, but they are
not access-control credentials. Do not upload confidential data to a public deployment.

## Features

### Adaptive chat

Each Chat question selects exactly one route:

- `index`: answer from uploaded PDF/TXT content.
- `general`: answer stable general-knowledge or conversational questions directly.
- `search`: search the current web with Tavily and produce a sourced answer.

The indexed-document route performs vector retrieval, local cross-encoder reranking, grounded answer
generation, and faithfulness verification. If retrieval is weak, it rewrites the query once and can
fall back to web search.

Chat intentionally uses one route per question. A task that must combine uploaded documents with web
sources or datasets belongs in the Research workspace.

### Multi-agent research

The Research workspace runs a supervisor-worker LangGraph:

- `supervisor`: creates bounded specialist assignments.
- `document_investigator`: retrieves workspace document evidence from Qdrant.
- `web_researcher`: gathers current external evidence with Tavily.
- `data_analyst`: computes dataset summaries, derived profit, grouped totals, top/bottom performers,
  and charts.
- `evidence_critic`: audits evidence coverage and can request one bounded revision round.
- `deliverable_builder`: creates and stores a Markdown research report.

Every specialist records evidence IDs and source metadata in a shared MongoDB ledger. Reports and PNG
charts are stored as artifacts and can be previewed or downloaded from the UI. Research runs are
asynchronous durable jobs: the API immediately returns a task ID, LangGraph checkpoints after graph
steps, and the UI streams persisted activity events while work continues in the background.

Durability includes:

- request idempotency through the `Idempotency-Key` header;
- retry-safe evidence and artifact writes with deterministic operation keys;
- MongoDB-backed LangGraph checkpoints and pending writes;
- worker leases and stale-job recovery after an unexpected process shutdown;
- reconnectable server-sent events (SSE), cancellation, explicit retry, and job history.

Checkpointing resumes at graph boundaries, not in the middle of an external model/search call. If a
process stops during a node, that node may run again after its lease expires, while its persistent
side effects remain deduplicated.

Document and web evidence collection is deterministic to avoid unreliable free-model tool calls.
Data-only reports are assembled directly from computed results so the model cannot alter numeric
facts. Groq structured-output failures have deterministic supervisor and critic fallbacks.

### Dataset analysis

- Accepts CSV, JSON/JSONL, and XLSX.
- Enforces upload, row, and column limits.
- Stores datasets in bounded MongoDB batches.
- Detects numeric and categorical columns.
- Derives `profit = revenue - cost` when those columns exist.
- Produces descriptive totals, grouped sums, rankings, and a PNG chart.
- Produces a non-empty deterministic data report.

Sample inputs are included:

- `sample-policy.txt` for document and document-plus-web tests.
- `sales.csv` for data-analysis tests.

## Architecture

```mermaid
flowchart LR
    UI[Streamlit UI] --> API[FastAPI]

    API --> CHAT[Adaptive chat graph]
    CHAT --> CLASSIFY[Groq route classifier]
    CLASSIFY --> GENERAL[General Groq answer]
    CLASSIFY --> RAG[Qdrant retrieval]
    CLASSIFY --> WEB[Tavily search]
    RAG --> RERANK[Local FlashRank reranker]
    RERANK --> GENERATE[Grounded Groq generation]
    WEB --> GENERATE
    GENERATE --> VERIFY[Faithfulness verification]

    API --> JOBS[(Durable research jobs)]
    JOBS --> WORKER[Leased async worker]
    WORKER --> RESEARCH[Research orchestration graph]
    RESEARCH --> CHECKPOINTS[(LangGraph checkpoints)]
    RESEARCH --> EVENTS[(Persisted SSE events)]
    EVENTS --> UI
    RESEARCH --> SUPERVISOR[Supervisor]
    SUPERVISOR --> DOC[Document investigator]
    SUPERVISOR --> WEBAGENT[Web researcher]
    SUPERVISOR --> DATA[Data analyst]
    DOC --> LEDGER[MongoDB evidence ledger]
    WEBAGENT --> LEDGER
    DATA --> LEDGER
    LEDGER --> CRITIC[Evidence critic]
    CRITIC --> REPORT[Report and chart artifacts]

    RAG --> QDRANT[(Qdrant)]
    API --> MONGO[(MongoDB)]
```

### Document ingestion

1. FastAPI validates the file, workspace ID, description, type, and size.
2. PDF pages or UTF-8 text are loaded.
3. Text is split into overlapping chunks.
4. FastEmbed creates 384-dimensional vectors locally.
5. Chunks and metadata are upserted into one Qdrant collection.
6. `session_id` payload filtering isolates retrieval between workspaces.

Uploads are additive. Qdrant payloads preserve source, document ID, page, and chunk metadata. The
active backend is Qdrant; the old in-process FAISS implementation has been removed.

### Chat query flow

1. Load bounded history from MongoDB.
2. Retrieve candidate workspace chunks for classifier context.
3. Classify the question as `index`, `general`, or `search`.
4. For `index`, retrieve up to `RETRIEVAL_TOP_K`, rerank locally, and retain `RERANK_TOP_N`.
5. If relevance is insufficient, rewrite once and then optionally use web search.
6. Generate a grounded answer and sources.
7. Verify faithfulness; regenerate once or return a safe fallback.
8. Persist the user and assistant messages in MongoDB.

### Research flow

1. The UI submits an objective with an idempotency key; FastAPI persists a queued job and returns
   HTTP `202` with its task ID.
2. A background worker atomically leases the job. A stale lease makes interrupted work reclaimable.
3. The supervisor selects distinct relevant specialists.
4. Specialists run in parallel where possible, write idempotent evidence, and emit progress events.
5. LangGraph stores checkpoints and pending writes in MongoDB after graph steps.
6. The critic evaluates coverage and may request one deduplicated revision round.
7. The deliverable builder writes retry-safe artifacts; data-only deliverables remain deterministic.
8. The result is persisted on the job. The UI receives live SSE updates and can later reconnect by ID.

## Technology choices

| Technology | Role | Why it is used |
|---|---|---|
| Python 3.12 | Runtime | Strong AI/data ecosystem and async support |
| FastAPI | Backend API | Typed validation, async endpoints, automatic OpenAPI docs |
| Streamlit | Demo UI | Fast iteration for chat, uploads, agent activity, and artifacts |
| LangGraph | Workflows | Explicit state, branching, parallel dispatch, and resumable checkpoints |
| LangChain Core/Groq | Model adapters | Structured output, prompts, messages, and tool interfaces |
| Groq `openai/gpt-oss-20b` | Chat inference | Accessible free-tier inference for an educational demo |
| FastEmbed `BAAI/bge-small-en-v1.5` | Embeddings | Free local ONNX inference with small 384-dimension vectors |
| FlashRank TinyBERT | Reranking | Free local cross-encoder relevance scoring |
| Qdrant | Vector database | Persistent cosine search plus workspace payload filtering |
| MongoDB | Application storage | Flexible storage for history, datasets, evidence, and binary artifacts |
| Tavily | Web search | Search results with source titles, URLs, snippets, and scores |
| pandas/Matplotlib | Data agent | Safe tabular calculations and server-side PNG charts |
| Docker Compose | Local stack | Reproducible API, UI, MongoDB, Qdrant, and persistent volumes |

Important trade-offs:

- Local embeddings and reranking avoid usage charges but increase image size, memory usage, and first-run
  model-download time.
- Streamlit is ideal for a demo but offers less UI control than a dedicated React frontend.
- Groq free-tier limits require request pacing and bounded prompts; complex research may be slower.
- MongoDB is convenient for mixed record shapes and artifacts, while a relational database would offer
  stronger joins and transactional constraints.
- One shared Qdrant collection is simpler than a collection per workspace, but correct payload filters
  are essential.
- Single-route Chat is easy to understand; the Research workspace handles hybrid document/web/data tasks.
- MongoDB polling and leases keep the demo deployment simple and durable without Redis/Celery; a
  dedicated queue would offer higher throughput and scheduling controls at larger scale.
- SSE is one-way and ideal for activity feeds; WebSockets would be preferable for highly interactive,
  bidirectional control.

## Project structure

```text
.
|-- README.md                    Project, setup, operation, and deployment guide
|-- CODE_STYLE_GUIDE.md          Development conventions (kept separately)
|-- .env.example                 Complete environment template
|-- docker-compose.yml           Local four-service stack
|-- Dockerfile                   FastAPI image
|-- Dockerfile.streamlit         Streamlit image
|-- requirements*.txt            Backend, frontend, and development dependencies
|-- sample-policy.txt            Document test fixture
|-- sales.csv                    Dataset test fixture
|-- src/
|   |-- agents/                  Supervisor, specialists, critic, deliverable builder
|   |-- api/                     RAG and multi-agent HTTP endpoints
|   |-- config/                  Prompt templates
|   |-- core/                    Settings, logging, prompt budgets, integration errors
|   |-- data/                    Dataset parsing and validation
|   |-- db/                      MongoDB jobs, checkpoints, datasets, evidence, artifacts
|   |-- llms/                    Groq, FastEmbed, and FlashRank factories
|   |-- memory/                  MongoDB chat history
|   |-- models/                  Pydantic and LangGraph state models
|   |-- orchestration/           Multi-agent research graph
|   |-- rag/                     Upload, Qdrant retrieval, adaptive chat graph
|   `-- tools/                   Deterministic graph routing decisions
|-- streamlit_app/               Home, Chat, Research, and API client
`-- tests/                       Unit, graph, retrieval, frontend, and agent tests
```

## Prerequisites

For the recommended local setup:

- Docker Desktop with Docker Compose v2.
- A Groq API key.
- A Tavily API key.

The local MongoDB and Qdrant containers require no external accounts.

## Environment setup

Copy the template:

```powershell
Copy-Item .env.example .env
```

At minimum, replace these placeholders:

```env
GROQ_API_KEY=your-groq-key
TAVILY_API_KEY=your-tavily-key
QDRANT_API_KEY=choose-a-long-random-local-key
MONGO_ROOT_PASSWORD=choose-a-long-random-local-password
MONGODB_URL=mongodb://adaptive_rag:the-same-password@mongo:27017/adaptive_rag?authSource=admin
```

The password inside `MONGODB_URL` must match `MONGO_ROOT_PASSWORD`. Never commit `.env`; it is ignored
by Git and excluded from Docker build contexts.

### Model settings

```env
LLM_PROVIDER=groq
GROQ_CHAT_MODEL=openai/gpt-oss-20b
GROQ_MAX_OUTPUT_TOKENS=1200
GROQ_REQUESTS_PER_SECOND=0.2
EMBEDDING_PROVIDER=fastembed
FASTEMBED_MODEL=BAAI/bge-small-en-v1.5
FASTEMBED_CACHE_DIR=/models/fastembed
EMBEDDING_DIMENSIONS=384
RERANKER_MODEL=ms-marco-TinyBERT-L-2-v2
RERANKER_CACHE_DIR=/models/flashrank
```

`GROQ_REQUESTS_PER_SECOND=0.2` deliberately spaces model calls to reduce free-tier bursts. The model
cache is mounted at `/models` in Docker.

### Storage settings

```env
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=agentic_workspace_bge_v1
MONGODB_DB_NAME=adaptive_rag
```

The configured embedding dimension must match the Qdrant collection. If the embedding model or
dimension changes, use a new collection name and re-index documents. Do not point 384-dimensional
FastEmbed vectors at an older 1536-dimensional collection.

### Retrieval and ingestion settings

| Variable | Default | Meaning |
|---|---:|---|
| `RETRIEVAL_TOP_K` | `12` | Initial Qdrant candidates |
| `RERANK_TOP_N` | `5` | Documents retained after reranking |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.2` | Qdrant candidate threshold |
| `RERANK_RELEVANCE_THRESHOLD` | `0.45` | Minimum reranked score used for generation |
| `MAX_RETRIEVAL_RETRIES` | `1` | Query rewrite attempts |
| `CHUNK_SIZE` | `1000` | Document chunk characters |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `MAX_UPLOAD_BYTES` | `20971520` | PDF/TXT limit (20 MiB) |
| `MAX_DATASET_UPLOAD_BYTES` | `10485760` | Dataset limit (10 MiB) |
| `MAX_DATASET_ROWS` | `10000` | Maximum stored rows |
| `MAX_DATASET_COLUMNS` | `100` | Maximum columns |
| `MAX_HISTORY_MESSAGES` | `30` | Bounded chat history |

### Agent budget settings

| Variable | Default | Meaning |
|---|---:|---|
| `AGENT_MAX_ITERATIONS` | `3` | Generic tool-agent loop bound |
| `SUPERVISOR_MAX_WORKERS` | `2` | Initial distinct specialists |
| `AGENT_MAX_REVISIONS` | `1` | Critic revision rounds |
| `GRAPH_RECURSION_LIMIT` | `15` | Maximum graph supersteps per attempt |
| `RESEARCH_WORKER_POLL_SECONDS` | `1.0` | Delay while no durable job is available |
| `RESEARCH_JOB_LEASE_SECONDS` | `60` | Worker ownership window before crash recovery |
| `RESEARCH_JOB_MAX_ATTEMPTS` | `3` | Maximum automatic lease claims before manual retry |
| `RESEARCH_EVENT_POLL_SECONDS` | `0.5` | MongoDB-to-SSE progress polling interval |
| `AGENT_TOOL_RESULT_CHARS` | `8000` | Tool-message context cap |
| `CRITIC_RESULTS_CHARS` | `3000` | Specialist-summary budget |
| `CRITIC_EVIDENCE_CHARS` | `6000` | Evidence context budget |

See `.env.example` for the complete authoritative list.

## Run locally with Docker

Start Docker Desktop, then run:

```powershell
cd C:\Projects\Adaptive-Rag
docker compose up --build -d
docker compose ps
Invoke-RestMethod http://localhost:8000/health/ready
```

Expected URLs:

- UI: `http://localhost:8501`
- API root: `http://localhost:8000`
- OpenAPI docs: `http://localhost:8000/docs`
- Liveness: `http://localhost:8000/health/live`
- Readiness: `http://localhost:8000/health/ready`

Use this on later runs when dependencies and source files have not changed:

```powershell
docker compose up -d
```

The first document upload downloads the FastEmbed model. The first indexed query downloads the small
FlashRank reranker. The `model_cache` Docker volume reuses them on later starts.

### Logs

```powershell
docker compose logs api --tail 150
docker compose logs frontend --tail 100
docker compose logs -f api
```

Press `Ctrl+C` to leave live log viewing; it does not stop the containers.

### Stop

```powershell
docker compose down
```

This removes containers and the Compose network but preserves named volumes. Do not add `-v` unless
you intentionally want to delete MongoDB data, Qdrant documents, and cached models.

## Full UI test checklist

Use one workspace ID throughout so uploads remain available when switching between pages. Allow a
pause between model-heavy tests because provider free-tier limits are account-dependent.

1. Confirm `/health/ready` reports MongoDB, Qdrant, Groq, and Tavily as ready.
2. Open Chat, upload `sample-policy.txt`, enter a description, and click **Index document**.
3. Ask `How many days per week may Acme employees work remotely?`; expect route `index` and the file
   as a source.
4. Ask `Explain recursion in simple terms.`; expect route `general`.
5. Ask a clearly current question; expect route `search` and web links.
6. Click **Research workspace**. Hybrid tasks must be run here, not in Chat.
7. Confirm `sample-policy.txt` appears under **Available data**; index it here if necessary.
8. Upload `sales.csv` as a Dataset and confirm its ID and row count.
9. Run document-plus-web research:

   > Compare the uploaded Acme remote-work policy with current authoritative remote-work
   > cybersecurity guidance. Identify strengths, gaps, and recommended changes. Produce a sourced
   > report.

   Expect live planning/specialist/critique/report activity, both `document_investigator` and
   `web_researcher`, an evidence audit, and a non-empty report. Research results do not display a
   single Chat route label. Refreshing the page does not cancel the job; use **Durable job history**
   to reconnect.

10. Run dataset research:

    > Analyze sales.csv. Calculate total revenue and profit overall, by region, and by quarter.
    > Identify top and bottom performers, create a chart, and produce an evidence-backed report.

    Expect one `data_analyst`, calculated totals, a previewable `analysis-chart.png`, and a non-empty
    deterministic `research-report.md`. For the included sample, revenue is `765000` and profit is
    `255000`.

Research commonly takes 20–40 seconds because Groq calls are deliberately paced.

## Run without Docker

Use Python 3.10 or newer. Run MongoDB and Qdrant separately, and change Docker service hostnames in
`.env` to local addresses:

```env
QDRANT_URL=http://localhost:6333
MONGODB_URL=mongodb://localhost:27017/adaptive_rag
FASTEMBED_CACHE_DIR=.cache/fastembed
RERANKER_CACHE_DIR=.cache/flashrank
RAG_API_URL=http://127.0.0.1:8000
```

Backend terminal:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8000
```

Frontend terminal:

```powershell
venv\Scripts\Activate.ps1
pip install -r requirements-frontend.txt
streamlit run streamlit_app/home.py --server.port 8501
```

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health/live` | Process liveness |
| `GET` | `/health/ready` | Database and credential readiness |
| `POST` | `/rag/documents/upload` | Index PDF/TXT into a workspace |
| `POST` | `/rag/query` | Run adaptive Chat |
| `DELETE` | `/rag/documents/{document_id}` | Delete one workspace document |
| `DELETE` | `/rag/history` | Clear workspace chat history |
| `POST` | `/agents/datasets/upload` | Store CSV/JSON/XLSX |
| `GET` | `/agents/datasets` | List workspace datasets |
| `POST` | `/agents/research` | Queue research; returns HTTP 202 and a task ID |
| `GET` | `/agents/tasks` | List durable jobs for a workspace |
| `GET` | `/agents/tasks/{task_id}` | Read job status and progress |
| `GET` | `/agents/tasks/{task_id}/events` | Read persisted activity after a sequence number |
| `GET` | `/agents/tasks/{task_id}/events/stream` | Stream reconnectable activity with SSE |
| `GET` | `/agents/tasks/{task_id}/result` | Read a completed research result |
| `DELETE` | `/agents/tasks/{task_id}` | Request cooperative cancellation |
| `POST` | `/agents/tasks/{task_id}/retry` | Retry failed or cancelled work from its checkpoint |
| `GET` | `/agents/artifacts/{artifact_id}` | Download a workspace artifact |

### Query example

```powershell
$body = @{
    query = "How many remote days are allowed?"
    session_id = "demo-session-001"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri http://localhost:8000/rag/query `
    -ContentType application/json `
    -Body $body
```

### Research example

```powershell
$session = "demo-session-001"
$idempotencyKey = [guid]::NewGuid().ToString()
$body = @{
    objective = "Compare the uploaded policy with current guidance."
    session_id = $session
    available_data = @("Uploaded document: sample-policy.txt")
} | ConvertTo-Json -Depth 5

$job = Invoke-RestMethod `
    -Method Post `
    -Uri http://localhost:8000/agents/research `
    -ContentType application/json `
    -Headers @{ "Idempotency-Key" = $idempotencyKey } `
    -Body $body

do {
    Start-Sleep -Seconds 2
    $status = Invoke-RestMethod `
        -Uri "http://localhost:8000/agents/tasks/$($job.task_id)" `
        -Headers @{ "X-Session-ID" = $session }
    "$($status.status) - $($status.stage) - $($status.progress)%"
} while ($status.status -in @("queued", "running", "cancel_requested"))

if ($status.status -eq "completed") {
    Invoke-RestMethod `
        -Uri "http://localhost:8000/agents/tasks/$($job.task_id)/result" `
        -Headers @{ "X-Session-ID" = $session }
}
```

Repeating the submission with the same idempotency key and identical body returns the same task ID.
Reusing it with a different body returns HTTP `409`.

Interactive request schemas and responses are available at `/docs`.

## Persistence and workspace isolation

Docker volumes:

- `mongo_data`: chat history, dataset rows and metadata, evidence, artifacts.
- `qdrant_data`: document vectors and payloads.
- `model_cache`: FastEmbed and FlashRank model files.

MongoDB collections include chat history, datasets, dataset batches, evidence, artifacts,
`research_jobs`, `research_events`, `langgraph_checkpoints`, and `langgraph_writes`. Job results,
events, checkpoints, and side effects survive ordinary refreshes and process restarts. Qdrant uses
one shared collection filtered by `session_id` and `document_id`.

The API process also runs the lightweight leased worker. Keep `WEB_CONCURRENCY=1` for this demo.
Leases still prevent duplicate ownership if more than one process is accidentally started, but this
is not designed as a high-throughput distributed queue.

Workspace IDs prevent accidental cross-workspace retrieval in normal application paths, but without
authentication anyone who knows an ID could use it. Add authentication and authorization before using
the project with private or multi-user data.

## Minimal Render deployment

The minimal managed deployment uses four components:

1. Render Docker web service for FastAPI using `Dockerfile` and health path `/health/live`.
2. Render Docker web service for Streamlit using `Dockerfile.streamlit`.
3. Managed MongoDB, such as MongoDB Atlas.
4. Managed Qdrant, such as Qdrant Cloud.

API service environment:

- All model, retrieval, and agent settings from `.env.example`.
- `GROQ_API_KEY` and `TAVILY_API_KEY` as secrets.
- Cloud `MONGODB_URL`, `QDRANT_URL`, and `QDRANT_API_KEY`.
- `CORS_ORIGINS` as a JSON list containing the deployed UI URL.
- `WEB_CONCURRENCY=1` for a small demo instance.

Frontend service environment:

```env
RAG_API_URL=https://your-api-service.onrender.com
RAG_REQUEST_TIMEOUT_SECONDS=300
```

Both Docker images honor Render's injected `PORT`. Free instances may sleep and may lose local model
caches after replacement, so the first upload/query after a cold start can be slower. Do not deploy
MongoDB or Qdrant inside the stateless web services.

For Qdrant Cloud, create a cluster and API key, set its HTTPS URL and key, and keep
`QDRANT_COLLECTION=agentic_workspace_bge_v1`. The API creates the collection and payload indexes at
startup if absent.

## Troubleshooting

### Readiness is `not_ready`

- Confirm `GROQ_API_KEY` and `TAVILY_API_KEY` are not placeholders.
- Confirm the MongoDB password matches the connection string.
- Inspect `docker compose logs api --tail 150`.

### Document upload fails

- Only PDF and UTF-8 TXT are supported.
- Confirm Qdrant is running and the collection dimension is 384.
- The first upload may wait while FastEmbed downloads.

### Chat returns `search` despite an uploaded document

Chat chooses one route. A question asking for current external guidance can correctly choose `search`.
Use Research to combine the uploaded document and web evidence.

### Groq returns 429 or token-limit errors

- Wait for the provider window to reset.
- Avoid multiple back-to-back research runs.
- Keep objectives concise and retain the configured prompt budgets and pacing.
- Free-tier limits are provider/account dependent.

### Research completes with limitations

The critic may judge evidence coverage incomplete even when the request succeeds. Inspect Specialist
activity and the Evidence audit. This is distinct from an API/service failure.

### Empty or inaccurate old report

Artifacts are immutable records from their original runs. Rerun the objective after updating the
project. Current data-only reports are deterministic and reject empty model output.

## Development and verification

Install development dependencies:

```powershell
pip install -r requirements-dev.txt
```

Run checks:

```powershell
python -m pytest -q
ruff check src tests streamlit_app
ruff format --check src tests streamlit_app
```

The test suite covers configuration, upload validation, Qdrant isolation, routing, model adapters,
prompt budgets, frontend error handling, specialist orchestration, durability contracts, data
analysis, and non-empty deliverables.

Development conventions remain in [CODE_STYLE_GUIDE.md](CODE_STYLE_GUIDE.md).

## Current scope

- Authentication and authorization are deliberately deferred.
- Chat uses a single selected route; Research performs hybrid work.
- Research is asynchronous, checkpointed, leased, and intended for demo-sized tasks.
- Cancellation is cooperative at graph-node boundaries; in-flight provider calls are not forcibly
  terminated.
- Agent and prompt budgets prioritize Groq free-tier reliability over maximum parallelism.
- Local Qdrant and MongoDB ports are not published outside the Compose network.
- No load balancer or multi-node infrastructure is required for the intended educational deployment.
