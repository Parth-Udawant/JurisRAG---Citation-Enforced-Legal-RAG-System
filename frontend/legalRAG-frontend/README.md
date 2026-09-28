# LexiRAG Frontend

Next.js + React + TypeScript frontend for the LexiRAG FastAPI backend.

## Requirements

- Node.js 20+
- Legal RAG backend running at `http://127.0.0.1:8000`

## Setup

```bash
npm install
copy .env.example .env.local
npm run dev
```

On PowerShell, use:

```powershell
Copy-Item .env.example .env.local
```

Open `http://localhost:3000`.

## Backend contract

The frontend calls exactly:

```text
POST {NEXT_PUBLIC_API_BASE_URL}/api/v1/chat
```

Request:

```json
{
  "question": "What is the punishment for murder under the BNS?",
  "act": "bns",
  "conversation_id": null
}
```

The frontend consumes `answer`, `citations`, `sources`, `insufficient_evidence`, `retrieval`, `conversation_id`, and `disclaimer` from the backend response.

## Current UX

- LexiRAG dark glassmorphism UI based on the supplied reference
- New Chat
- All Acts / BNS / ICA filtering
- Clickable `[BNS §...]` and `[ICA §...]` citations
- Source drawer and expandable source cards
- Insufficient-evidence state
- Loading and error states
- Conversation ID placeholder for future memory
- Legal disclaimer
- Responsive desktop/mobile layout

No Azure credentials or cost telemetry are exposed to the browser.
