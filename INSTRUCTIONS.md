# AgentForge AI Instructions

This file contains the longer operational notes, examples, endpoint reference, deployment guidance,
and troubleshooting details that are useful while developing or demoing the project.

## Environment setup

Copy the template and edit it:

```powershell
Copy-Item .env.example .env
```

Important environment groups:

### Core provider keys

```env
GROQ_API_KEY=...
GROQ_CHAT_MODEL=openai/gpt-oss-20b
TAVILY_API_KEY=...
```

### Storage

```env
MONGO_ROOT_USERNAME=admin
MONGO_ROOT_PASSWORD=password
MONGODB_URI=mongodb://admin:password@mongo:27017/agentic_rag?authSource=admin
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=local-dev-key
QDRANT_COLLECTION=agentic_workspace_bge_v1
```

For local Docker Compose, keep service hostnames like `mongo` and `qdrant`. For non-Docker local
runs, change them to `localhost`.

### Retrieval and ingestion

```env
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSION=384
RETRIEVAL_TOP_K=8
RERANK_TOP_N=4
CHUNK_SIZE=900
CHUNK_OVERLAP=150
```

### RAGAS

```env
RAGAS_ENABLED=true
RAGAS_DO_NOT_TRACK=true
RAGAS_JUDGE_MODEL=
RAGAS_WORKER_POLL_SECONDS=1
RAGAS_EVAL_MAX_CONTEXTS=5
RAGAS_EVAL_CONTEXT_CHARS=1800
```

If `RAGAS_JUDGE_MODEL` is blank, the app defaults to `GROQ_CHAT_MODEL`.

### Sarvam / multilingual voice

```env
ENABLE_VOICE_FEATURES=true
ENABLE_MULTILINGUAL_DOCS=true
SARVAM_API_KEY=...
SARVAM_STT_MODEL=saaras:v3
SARVAM_TTS_MODEL=bulbul:v3
SARVAM_TTS_DEFAULT_SPEAKER=auto
SARVAM_TTS_DEFAULT_PACE=1.0
DOCUMENT_PARSER_PROVIDER=auto
```

English documents use local parsing by default. Non-English/Indic documents can use Sarvam document
digitization when configured.

### Repository and dataset limits

```env
MAX_DATASET_UPLOAD_BYTES=10485760
MAX_DATASET_ROWS=10000
MAX_REPOSITORY_UPLOAD_BYTES=104857600
MAX_REPOSITORY_FILES=1000
MAX_REPOSITORY_FILE_BYTES=524288
MAX_REPOSITORY_TOTAL_BYTES=20971520
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
```

`STREAMLIT_SERVER_MAX_UPLOAD_SIZE` is in MB.

## Run locally with Docker

```powershell
docker compose up --build -d
docker compose ps
Invoke-RestMethod http://localhost:8000/health/ready
```

Open:

- UI: http://localhost:8501
- API docs: http://localhost:8000/docs

Logs:

```powershell
docker compose logs api --tail 150
docker compose logs frontend --tail 100
docker compose logs -f api
```

Stop:

```powershell
docker compose down
```

Reset local data:

```powershell
docker compose down -v
```

## Run without Docker

Run MongoDB and Qdrant separately, update `.env` hostnames to `localhost`, then:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-frontend.txt

uvicorn src.main:app --host 0.0.0.0 --port 8000
streamlit run streamlit_app/home.py
```

Docker is recommended because it starts MongoDB, Qdrant, API, and Streamlit together.

## Suggested UI demo checklist

1. Open http://localhost:8501.
2. Rename the default workspace in the sidebar.
3. Open Chat.
4. Upload `sample-policy.txt`.
5. Ask a document question, for example:

   > How many days per week can Acme employees work remotely?

6. Ask a general question:

   > Explain recursion in simple terms.

7. Ask a current/external question to trigger search.
8. Expand RAGAS evaluation below an answer and run a reference-free evaluation.
9. Open Research.
10. Upload `sales.csv` as a dataset and ask for a chart/report.
11. Upload a repository ZIP and run repository analysis.
12. Try voice input/output if Sarvam is configured.

## API reference

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/health/ready` | MongoDB/Qdrant/provider readiness |
| `POST` | `/rag/query` | Run adaptive Chat |
| `GET` | `/rag/history` | Load bounded chat history |
| `POST` | `/rag/documents/upload` | Upload and index PDF/TXT |
| `GET` | `/rag/documents` | List indexed workspace documents |
| `DELETE` | `/rag/documents/{document_id}` | Delete one indexed document |
| `POST` | `/rag/evaluations` | Submit RAGAS evaluation job |
| `GET` | `/rag/evaluations/{evaluation_id}` | Read evaluation status/results |
| `POST` | `/rag/speech/transcribe` | Sarvam STT |
| `POST` | `/rag/speech/synthesize` | Sarvam TTS |
| `GET` | `/rag/speech/voices` | Supported voice metadata |
| `POST` | `/agents/datasets/upload` | Store dataset |
| `GET` | `/agents/datasets` | List workspace datasets |
| `POST` | `/agents/repositories/upload` | Store repository ZIP |
| `GET` | `/agents/repositories` | List repositories |
| `POST` | `/agents/research` | Create durable research job |
| `GET` | `/agents/tasks` | List research jobs |
| `GET` | `/agents/tasks/{task_id}` | Read job status |
| `GET` | `/agents/tasks/{task_id}/events` | Stream persisted SSE job events |
| `GET` | `/agents/tasks/{task_id}/result` | Read completed result |
| `DELETE` | `/agents/tasks/{task_id}` | Request cancellation |
| `POST` | `/agents/tasks/{task_id}/retry` | Retry failed/cancelled job |

## Query example

```powershell
$body = @{
    query = "How many days per week can Acme employees work remotely?"
    session_id = "demo-session-001"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://localhost:8000/rag/query" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

## RAGAS evaluation example

```powershell
$evaluationBody = @{
    response_id = "<response_id from /rag/query>"
    reference = $null
} | ConvertTo-Json

$evaluation = Invoke-RestMethod `
    -Uri "http://localhost:8000/rag/evaluations" `
    -Method Post `
    -ContentType "application/json" `
    -Headers @{ "X-Session-ID" = "demo-session-001"; "Idempotency-Key" = [guid]::NewGuid().ToString() } `
    -Body $evaluationBody

Invoke-RestMethod `
    -Uri "http://localhost:8000/rag/evaluations/$($evaluation.evaluation_id)" `
    -Headers @{ "X-Session-ID" = "demo-session-001" }
```

## Research job example

```powershell
$researchBody = @{
    objective = "Compare the uploaded remote-work policy with current cybersecurity guidance and produce a sourced report."
    session_id = "demo-session-001"
    available_data = @("Uploaded document: sample-policy.txt")
} | ConvertTo-Json -Depth 5

$job = Invoke-RestMethod `
    -Uri "http://localhost:8000/agents/research" `
    -Method Post `
    -ContentType "application/json" `
    -Headers @{ "Idempotency-Key" = [guid]::NewGuid().ToString() } `
    -Body $researchBody

do {
    Start-Sleep -Seconds 2
    $status = Invoke-RestMethod `
        -Uri "http://localhost:8000/agents/tasks/$($job.task_id)" `
        -Headers @{ "X-Session-ID" = "demo-session-001" }
    $status
} while ($status.status -notin @("completed", "failed", "cancelled"))

Invoke-RestMethod `
    -Uri "http://localhost:8000/agents/tasks/$($job.task_id)/result" `
    -Headers @{ "X-Session-ID" = "demo-session-001" }
```

## Repository-analysis example

Create a small ZIP:

```powershell
Compress-Archive `
    -Path .\src, .\README.md, .\requirements.txt `
    -DestinationPath demo-repository.zip `
    -Force
```

Upload through the Research UI, or through the interactive FastAPI `/docs` page. After upload, run
a Research objective such as:

> Explain this repository architecture, likely entry points, dependencies, runtime flow, and
> limitations with source-file evidence.

## Benchmark

Run the bundled RAGAS benchmark against a running local stack:

```powershell
python -m src.evaluation.benchmark --base-url http://localhost:8000
```

Reports are written under the benchmark output directory as timestamped JSON/CSV.

## Minimal Render deployment

The simple managed deployment uses:

1. Render Docker web service for FastAPI using `Dockerfile`.
2. Render Docker web service for Streamlit using `Dockerfile.streamlit`.
3. Render MongoDB alternative or external MongoDB Atlas database.
4. Qdrant Cloud free cluster or another hosted Qdrant instance.

### Backend service

- Environment: Docker
- Dockerfile: `Dockerfile`
- Health path: `/health/live`
- Required env vars:
  - `MONGODB_URI`
  - `QDRANT_URL`
  - `QDRANT_API_KEY`
  - `GROQ_API_KEY`
  - `TAVILY_API_KEY`
  - optional `SARVAM_API_KEY`

### Frontend service

- Environment: Docker
- Dockerfile: `Dockerfile.streamlit`
- Required env vars:
  - `RAG_API_URL=https://<your-backend-service>.onrender.com`
  - `RAG_REQUEST_TIMEOUT_SECONDS=300`
  - `STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200`

Both Docker images honor Render's injected `PORT`.

Free instances may sleep. The first model upload/query after a cold start can be slower because local
model caches may need to warm up.

## Troubleshooting

### Readiness is not OK

- Confirm MongoDB and Qdrant are running.
- Check credentials in `.env`.
- Inspect API logs:

```powershell
docker compose logs api --tail 150
```

### Document upload fails

- Check file type and size.
- For non-English or scanned PDFs, configure `SARVAM_API_KEY`.
- Re-index documents after parser changes.

### Chat returns search despite uploaded documents

- Confirm the document appears under "Indexed this session".
- Make sure you are in the same workspace ID.
- Re-index if the document was uploaded before parser/retrieval changes.

### Groq rate limit errors

- Wait and retry.
- Use shorter questions/objectives.
- Avoid multiple simultaneous RAGAS/research jobs on the free tier.

### Repository ZIP upload fails

- Keep ZIPs below `MAX_REPOSITORY_UPLOAD_BYTES`.
- Avoid huge `node_modules`, `.git`, build outputs, or media assets.
- Increase `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` if Streamlit rejects the upload before the backend.

## Development checks

```powershell
ruff check src streamlit_app tests
python -m pytest
python -m compileall src streamlit_app
docker compose build
```

