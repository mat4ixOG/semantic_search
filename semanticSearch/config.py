import os

# Every setting can be overridden with an environment variable so the same
# code runs from the CLI, from the API and from a container without edits.
PREFIX = "RAG_"


def text(name, default):
    return os.getenv(PREFIX + name, default)


def flag(name, default):
    raw = os.getenv(PREFIX + name)

    if raw is None:
        return default

    return raw.strip().lower() in {"1", "true", "yes", "on"}


def number(name, default):
    raw = os.getenv(PREFIX + name)

    if raw is None:
        return default

    try:
        return type(default)(raw)
    except ValueError:
        return default


# Model that writes the final answer
MODEL_NAME = text("MODEL_NAME", "gemma3:4b")

# Model for the cheap pipeline stages: routing, query expansion, HyDE,
# planning. These are classification and rewriting jobs, a small model does
# them fine and several times faster. Falls back to MODEL_NAME automatically
# if it is not pulled.
UTILITY_MODEL = text("UTILITY_MODEL", "gemma3:1b")

EMBEDDING_MODEL = text("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

CHROMA_DB_PATH = text("CHROMA_DB_PATH", "chroma_db")

COLLECTION_NAME = text("COLLECTION_NAME", "employee_handbook")

DATABASE_PATH = text("DATABASE_PATH", "conversations.db")

CHUNK_SIZE = number("CHUNK_SIZE", 800)

CHUNK_OVERLAP = number("CHUNK_OVERLAP", 100)

TOP_K = number("TOP_K", 3)

CROSS_ENCODER_MODEL = text(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# Answer as a JSON object instead of streamed prose. Toggle at the prompt
# with `json on` / `json off`.
JSON_OUTPUT = flag("JSON_OUTPUT", True)

# Include the pipeline trace (routing, queries, HyDE passage) in JSON output
JSON_INCLUDE_TRACE = flag("JSON_INCLUDE_TRACE", True)

# Plan and execute agent: the planner decomposes the question into tool
# calls, a critic gets one chance to retry the steps that came back empty.
# Toggle at the prompt with `agent on` / `agent off`.
USE_AGENT = flag("USE_AGENT", True)

AGENT_MAX_STEPS = number("AGENT_MAX_STEPS", 3)

AGENT_ALLOW_REPLAN = flag("AGENT_ALLOW_REPLAN", True)

# Chunks kept after merging every search step
AGENT_MAX_CHUNKS = number("AGENT_MAX_CHUNKS", 6)

# Retrieval pipeline stages
USE_QUERY_REWRITE = flag("USE_QUERY_REWRITE", True)

USE_MULTI_QUERY = flag("USE_MULTI_QUERY", True)

USE_HYDE = flag("USE_HYDE", True)

USE_COMPRESSION = flag("USE_COMPRESSION", True)

# Compression backend.
# "cross-encoder" scores each sentence with the reranker already in memory,
# costs milliseconds and no LLM call.
# "llm" asks the model to extract sentences: slightly better, one LLM call
# per surviving chunk.
COMPRESSION_MODE = text("COMPRESSION_MODE", "cross-encoder")

# Sentences scoring below this are dropped from a chunk
COMPRESSION_MIN_SCORE = number("COMPRESSION_MIN_SCORE", -6.0)

# Chunks whose best sentence scores below this are dropped entirely
COMPRESSION_DROP_SCORE = number("COMPRESSION_DROP_SCORE", -9.0)

# Sentences shorter than this are glued to the next one, PDF text is full of
# stray line breaks and headings
MIN_SENTENCE_LENGTH = number("MIN_SENTENCE_LENGTH", 40)

# Extra query variations generated for multi query retrieval
MULTI_QUERY_COUNT = number("MULTI_QUERY_COUNT", 3)

# Chunks pulled per query before fusion
CANDIDATE_TOP_K = number("CANDIDATE_TOP_K", 20)

# Chunks kept after cross encoder reranking
RERANK_TOP_K = number("RERANK_TOP_K", 5)

# Reciprocal Rank Fusion smoothing constant
RRF_K = number("RRF_K", 60)

# Longest LLM generated query accepted. Anything longer means the model
# answered the question instead of rewriting it.
MAX_QUERY_LENGTH = number("MAX_QUERY_LENGTH", 300)

# Messages kept in a conversation's working memory
MEMORY_MAX_MESSAGES = number("MEMORY_MAX_MESSAGES", 10)

# Browser origins allowed to call the API
CORS_ORIGINS = text("CORS_ORIGINS", "http://localhost:3000")
