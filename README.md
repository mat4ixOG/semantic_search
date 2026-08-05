# Semantic Search

Local RAG over a PDF corpus: hybrid retrieval, query expansion, HyDE,
reranking, contextual compression, and a plan-and-execute agent with tools.
Everything runs offline against ollama.

## Run it

From the project root:

```bash
./run.sh
```

That starts the API on <http://127.0.0.1:8000> and the UI on
<http://localhost:3000>, and stops both on Ctrl+C.

Individually:

| Script         | What it starts                          |
| -------------- | --------------------------------------- |
| `./run.sh`     | API + UI together                       |
| `./run-api.sh` | API only, port 8000                     |
| `./run-ui.sh`  | UI only, port 3000 (needs the API up)   |
| `./run-cli.sh` | The terminal chat, no API and no UI     |

## First time setup

```bash
.venv/bin/pip install -r requirements.txt
ollama pull gemma3:4b
ollama pull gemma3:1b
.venv/bin/python semanticSearch/ingest.py
```

`gemma3:1b` is the utility model used for routing, query expansion, HyDE and
planning. Without it everything falls back to `gemma3:4b`, which works but is
several times slower.

The UI needs Node 18.18+. The default node on this machine is v10, so the
scripts switch to 20.20.1 through nvm automatically.

## How a question is answered

```
question
   |
[router]        document / memory / direct        (schema constrained)
   |
[planner]       decompose into tool calls         (agent mode only)
   |
[executor]      run each step
   |            search_documents -> the retrieval pipeline below
   |            current_time, date_math, calculator
   |
[critic]        one retry for steps that came back empty
   |
[synthesis]     JSON answer with resolved citations
```

The retrieval pipeline inside `search_documents`:

```
query
  |
[expand]     rewrite + N variations in one LLM call
  |
[search]     vector + BM25 per query, plus HyDE vector search
  |
[fuse]       Reciprocal Rank Fusion across every list
  |
[rerank]     cross encoder, top K
  |
[compress]   sentence level filtering with the same cross encoder
```

## Configuration

`semanticSearch/config.py` holds the defaults. Every one can be overridden
with an environment variable prefixed `RAG_`, so nothing needs editing to
change behaviour:

```bash
RAG_USE_HYDE=false ./run-api.sh
RAG_MODEL_NAME=qwen3:8b ./run-api.sh
RAG_COMPRESSION_MODE=llm ./run-cli.sh
```

Useful ones:

| Variable                | Default         | Effect                                        |
| ----------------------- | --------------- | --------------------------------------------- |
| `RAG_MODEL_NAME`        | `gemma3:4b`     | Model that writes answers                     |
| `RAG_UTILITY_MODEL`     | `gemma3:1b`     | Routing, expansion, HyDE, planning            |
| `RAG_USE_AGENT`         | `true`          | Plan and execute agent                        |
| `RAG_USE_HYDE`          | `true`          | Hypothetical document embeddings              |
| `RAG_USE_COMPRESSION`   | `true`          | Contextual compression                        |
| `RAG_COMPRESSION_MODE`  | `cross-encoder` | `cross-encoder` (fast) or `llm` (better)      |
| `RAG_RERANK_TOP_K`      | `5`             | Chunks kept after reranking                   |

## Evaluation

Build a golden set from the indexed documents, then score pipeline variants
against it:

```bash
cd semanticSearch
../.venv/bin/python generate_golden.py -n 30
../.venv/bin/python evaluate.py --variants baseline,compression --no-judge
../.venv/bin/python evaluate.py --variants all
```

`--no-judge` skips the LLM graded metrics (faithfulness, correctness,
relevance) and keeps only the free deterministic ones (hit rate, MRR, context
precision and recall). Start there: the judged sweep costs roughly three extra
model calls per question per variant.

Generated questions are a draft. They reuse the vocabulary of the chunk they
came from, so retrieval scores look better than reality. Comparing variants is
still valid, which is the point.

## API

| Method   | Path                            | Purpose                          |
| -------- | ------------------------------- | -------------------------------- |
| `GET`    | `/api/health`                   | Status and indexed chunk count   |
| `GET`    | `/api/config`                   | Active models and pipeline flags |
| `POST`   | `/api/ask`                      | One complete JSON answer         |
| `POST`   | `/api/ask/stream`               | SSE: retrieval, tokens, done     |
| `GET`    | `/api/conversations`            | List conversations               |
| `POST`   | `/api/conversations`            | Start one                        |
| `GET`    | `/api/conversations/{id}`       | Full message history             |
| `DELETE` | `/api/conversations/{id}`       | Delete one                       |

Interactive docs at <http://127.0.0.1:8000/docs>.

Conversations persist to `semanticSearch/conversations.db`.

## The CLI

`./run-cli.sh` gives the same pipeline in the terminal:

- `json on` / `json off` — full JSON payload, or streamed prose
- `agent on` / `agent off` — plan and execute, or straight retrieval
- `fast on` / `fast off` — answer with `FAST_MODEL` instead of `MODEL_NAME`
- `exit`

## Speed

Most of the wall clock is the answer model, and most of that is processing
the retrieved context rather than generating the reply. Rough per question
cost, measured on a CPU-only 12th gen i5 with no GPU:

| Setting                            | Effect                          |
| ---------------------------------- | ------------------------------- |
| default, simple question           | 4 LLM calls                     |
| default, multi part question       | 6 LLM calls                     |
| `fast on`                          | answers on the 1b, several times faster |
| `RAG_USE_HYDE=false`               | one call fewer per search       |
| `RAG_RERANK_TOP_K=3`               | shorter answer prompt           |

Short single clause questions skip the planner and the critic automatically,
since decomposing them produces the one search step plain retrieval already
runs. Look for `simple` in the trace.

If ollama is evicting models between calls, every call pays a cold load from
disk that costs more than the generation. Check with `ollama ps`: if it is
empty between questions, set `OLLAMA_KEEP_ALIVE=30m` and
`OLLAMA_MAX_LOADED_MODELS=2`, and make sure the machine has enough free RAM
to hold both models at once.

## Layout

```
semanticSearch/
  api.py            FastAPI app
  service.py        pipeline entry point shared by CLI and API
  main.py           terminal chat
  store.py          SQLite conversations

  router.py         document / memory / direct classification
  agent.py          plan, execute, critique, retry
  planner.py        decomposition and the critic
  tools.py          search_documents, current_time, date_math, calculator

  retrieval.py      the retrieval pipeline
  query.py          vector + BM25 + RRF fusion
  query_rewriter.py rewrite and multi query in one call
  hyde.py           hypothetical document embeddings
  reranker.py       cross encoder
  compressor.py     sentence level compression
  responder.py      answer assembly and citation resolution
  prompts.py        every prompt and JSON schema

  ingest.py         PDF -> chunks -> embeddings -> Chroma
  evaluate.py       scoring harness
  metrics.py        retrieval and LLM judged metrics
  generate_golden.py

frontend/           Next.js UI
```
