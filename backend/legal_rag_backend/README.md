# Legal RAG Backend

FastAPI backend for the validated Legal RAG baseline.

## Frozen baseline

- Hybrid + RRF
- candidate K = 10
- final K = 5
- MMR = false
- Azure OpenAI `text-embedding-3-large`, 1536 dimensions
- GPT deployment from `AZURE_OPENAI_GPT_DEPLOYMENT`

## API

### `GET /health`

Health check.

### `GET /api/v1/acts`

Returns act filter options.

### `POST /api/v1/chat`

Request:

```json
{
  "question": "What is the punishment for murder under the BNS?",
  "act": "bns",
  "conversation_id": null
}
```

Response contains:

- grounded answer
- structured citations
- source cards
- insufficient-evidence state
- retrieval metadata
- token usage
- request count
- disclaimer

`conversation_id` is intentionally a placeholder. Memory is not implemented yet.

## Run

From the backend directory:

```bash
pip install -r requirements.txt
uvicorn legal_rag_backend.main:app --app-dir src --reload
```

Open `/docs` for the generated API contract.

## Cost telemetry

Costs are server-side only and are never returned to the UI. Enter current pricing in `.env` when you want monetary estimates. The server logs:

- embedding input tokens
- GPT input tokens
- GPT output tokens
- estimated search cost per operation
- estimated total request cost
- requests/day

Search pricing is represented as a configurable per-operation estimate because Azure AI Search is provisioned capacity rather than a token-metered model API.
