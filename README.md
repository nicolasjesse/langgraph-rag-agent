# langgraph-rag-agent

Multi-agent RAG system built with LangGraph — supervisor, retrieval, planner, and verifier nodes with state-driven retry, checkpointing, and human-in-the-loop interrupts. Deployed on AWS Lambda behind an API-Gateway-keyed endpoint, with Langfuse tracing and a GitHub Actions pipeline that gates deploys on a real-API eval suite.

## Architecture

```
                     ┌──────────────┐
              user ─▶│  supervisor  │   classifies query (Haiku)
                     └──────┬───────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
       ┌──────────────┐            ┌──────────────┐
       │  retrieval   │            │   direct     │   simple / off-topic
       │  ChromaDB    │            │   answer     │   skips retrieval
       └──────┬───────┘            └──────┬───────┘
              │                           │
              ▼                           │
       ┌──────────────┐                   │
       │   planner    │ ◀─── retry ──┐    │
       │  Sonnet 4.5  │              │    │
       └──────┬───────┘              │    │
              │                      │    │
              ▼                      │    │
       ┌──────────────┐              │    │
       │   verifier   │ ─── fail ────┘    │   max 3 iterations
       │  (Haiku)     │ ─── pass ───┐     │   HITL interrupt at iter=2
       └──────────────┘             │     │
                                    ▼     ▼
                                  streamed answer
```

Five nodes, two model tiers (Haiku for routing/classification, Sonnet for the planner), one conditional branch off the supervisor, one cyclic retry loop between planner and verifier. The verifier escalates to a human via `interrupt()` after the second failed attempt; on resume, the human can `approve` / `reject` / `rewrite` the answer.

## Stack

| Component | Tool | Purpose |
|---|---|---|
| Language | Python 3.12 | LangGraph's primary SDK |
| Orchestration | **LangGraph 1.x** | Multi-node agent graph |
| Vector DB | **ChromaDB** (local file mode) | Embedding storage + similarity search |
| Planner LLM | **Claude Sonnet 4.5** (Anthropic API) | Answer generation |
| Classifier / verifier LLM | **Claude Haiku 4.5** | Routing + groundedness check |
| Embeddings | **OpenAI `text-embedding-3-small`** | Query/chunk vectors |
| Persistence | **SqliteSaver** (`langgraph.checkpoint.sqlite`) | Conversation state across runs, HITL pause/resume |
| Observability | **Langfuse** | Prompt-level tracing, cost, latency |
| Deployment | **AWS Lambda + API Gateway** | Serverless inference (container image, REST API, API-key auth) |
| CI/CD | **GitHub Actions** | Eval-gated deploys (lint → tests → ingest → evals → ECR push → Lambda update) |

## Corpus

168 markdown pages of public LangChain + LangGraph Python documentation, pulled via the official `llms.txt` index and Mintlify's markdown mode. Chunked at 800 tokens with 100-token overlap and embedded into a local ChromaDB collection (~1,575 chunks).

The `corpus/` and `chroma_db/` directories are gitignored — rebuild locally with the steps below.

## Setup

```bash
git clone https://github.com/nicolasjesse/langgraph-rag-agent
cd langgraph-rag-agent

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY and OPENAI_API_KEY (Langfuse keys are
# optional — tracing is opt-in)
```

Build the local corpus + vector store (one-time, ~1 minute total):

```bash
python scrape_corpus.py    # downloads ~168 LangChain docs into corpus/
python ingest.py           # chunks (800-token / 100-overlap), embeds, stores in chroma_db/
```

## Usage

Ask a question — answer is streamed and grounded in retrieved chunks with filename citations:

```bash
python cli.py "What is LangGraph and how does it differ from LangChain?"
```

Same conversation across runs (state checkpointed in `checkpoints/agent.db`):

```bash
python cli.py --thread my-conv "What is a checkpointer?"
python cli.py --thread my-conv "How do I attach one?"
```

Inspect retrieval directly without running the LLM (useful for tuning chunk size or `top_k`):

```bash
python query.py "How do I add memory to an agent?" -k 5
python query.py "streaming token by token" --full
```

Two reproducible demos exercise the verifier loop and HITL paths against real LLM calls:

```bash
python demos/verifier_loop_demo.py        # forces fail-then-pass recovery
python demos/hitl_demo.py approve         # forces fail-fail → human approve
python demos/hitl_demo.py reject          # forces fail-fail → human reject (budget exhausted)
python demos/hitl_demo.py rewrite         # forces fail-fail → human rewrites the answer
```

## Tests

Mocked unit tests for the graph + nodes — run in <1s, no API keys needed:

```bash
python -m pytest tests/
```

14 tests cover node logic, state shape, conditional routing, the verifier loop (happy path, retry-then-pass, budget exhaustion), all three HITL resume actions, and checkpointer state persistence.

## Evals

A real-API eval harness runs 10 question/expected-keyword pairs through the full graph and grades each answer. Exits non-zero on any failure, so it can gate deploys in CI:

```bash
python evals/run.py
```

Each run takes ~2 minutes and costs ~$0.15 in LLM calls. The grader is a simple case-insensitive keyword check — minimum viable for catching catastrophic regressions when prompts or retrieval change.

## Deployment

The agent ships as a container image to **AWS Lambda**, fronted by **API Gateway** with an API-key requirement and a per-day quota.

```bash
./deploy.sh        # build → push to ECR → create/update Lambda + API Gateway
./teardown.sh      # delete everything when you're done
```

`deploy.sh` is idempotent — re-running just rebuilds the image and updates the function code; the API Gateway, key, and usage plan are reused.

The endpoint URL and API key are written into `.env` (gitignored). Sample call:

```bash
curl -XPOST "$LAMBDA_API_URL" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $LAMBDA_API_KEY" \
  -d '{"query": "What is a checkpointer in LangGraph?"}'
```

Without the `x-api-key` header the gateway returns `403 Forbidden`. The default usage plan is **30 requests/day · 1 RPS · burst 2**, capping worst-case spend at roughly $1.50/day if the key ever leaks.

The Lambda runtime sets `HITL_ENABLED=false` (no checkpointer in stateless invocations) and `CHROMA_DB_PATH=/tmp/chroma_db` (Lambda's writable dir). The graph code itself is unchanged from the CLI build — both runtimes import the same `build_graph()`.

## CI/CD

Every push to `main` runs the full pipeline in `.github/workflows/ci.yml`:

```
checkout → install deps → ruff → unit tests
        → scrape corpus → ingest → eval suite (the gate)
        → if eval passed: docker buildx → push to ECR → update Lambda
```

The eval suite is the **deploy gate**: if any of the 10 keyword-graded test cases fails, the workflow halts before pushing the new image. This is *"treat prompts and evals like code"* in practice — a prompt change that regresses the test set never reaches production.

CI uses a **deploy-only IAM user** (`langgraph-rag-agent-ci`) with permissions scoped to this single Lambda function and ECR repo. Worst-case impact of leaked CI credentials is bounded to *that one Lambda*. Images are double-tagged (`latest` + `${GITHUB_SHA}`) so any past commit can be rolled to with one `update-function-code` call.

GitHub secrets used: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.

## Observability

Every LLM call (supervisor, planner, verifier, direct_answer) is forwarded to Langfuse via a single callback handler attached at the CLI driver. Traces are tagged with `session_id = thread_id`, so multi-turn conversations group together in the dashboard.

![Langfuse trace showing the supervisor → planner → verifier graph as nested spans, with the full prompt visible.](assets/langfuse-trace.png)

Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env` to enable. Without them, the CLI runs unchanged — tracing is opt-in per environment.

## Repo layout

```
agent.py                # graph: state, prompts, nodes, build_graph()
cli.py                  # CLI driver: argparse, SqliteSaver, streaming, HITL prompt
lambda_handler.py       # Lambda driver: copy chroma_db to /tmp, build graph, handle event
ingest.py               # chunk corpus/*.md, embed, write to chroma_db/
scrape_corpus.py        # build corpus/ from llms.txt
query.py                # ad-hoc retrieval inspection (no LLM call)
evals/                  # eval harness — test_set.json + run.py
demos/                  # reproducible scripts for verifier loop + HITL
tests/                  # mocked unit tests
Dockerfile              # Lambda container image
deploy.sh / teardown.sh # AWS lifecycle
.github/workflows/ci.yml
```

The graph layer (`agent.py`) has zero AWS / CLI / I/O code — only state, nodes, and graph wiring. Drivers (`cli.py`, `lambda_handler.py`) wrap it.

## License

MIT
