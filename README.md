# langgraph-rag-agent

Multi-agent RAG system built with LangGraph — supervisor, retrieval, planner, and verifier nodes with checkpointing and human-in-the-loop interrupts. Deployed on AWS Lambda with Langfuse tracing.

> Status: in development.

## Architecture

_TODO: add diagram and writeup once Day 2 is complete._

## Stack

- **LangGraph** — agent orchestration
- **ChromaDB** — local vector store
- **Anthropic Claude** (Sonnet 4.5 + Haiku 4.5) — LLM with model routing
- **OpenAI embeddings** — `text-embedding-3-small`
- **Langfuse** — prompt-level tracing and cost tracking
- **AWS Lambda + API Gateway** — deployment
- **GitHub Actions** — CI/CD with eval-gated deploys

## Setup

_TODO: write after the basic flow works._

## License

MIT
