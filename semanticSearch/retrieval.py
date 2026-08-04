import config
from compressor import compress
from hyde import generate_hypothetical_document
from prompts import DOCUMENT_ROUTE
from query import merge_results, search, vector_search
from query_rewriter import expand_query
from reranker import rerank
from router import EMPTY_TRANSCRIPT, route


def build_query_set(question: str, transcript: str) -> tuple[list[str], str]:
    """
    The original question, its rewrite and the multi query variations,
    deduplicated. Every one of them is searched independently.
    """

    rewritten_query, variations = expand_query(question, transcript)

    queries = [question]
    seen = {question.lower()}

    for candidate in [rewritten_query, *variations]:
        if candidate.lower() in seen:
            continue

        seen.add(candidate.lower())
        queries.append(candidate)

    return queries, rewritten_query


def empty_trace() -> dict:
    return {
        "route": None,
        "route_reason": None,
        "used": False,
        "queries": [],
        "rewritten_query": None,
        "hyde_used": False,
        "hyde_document": None,
        "fused": 0,
        "reranked": 0,
        "compressed": False,
        # LLM calls spent on retrieval, the answer call is not counted here
        "llm_calls": 0,
    }


def retrieve_query(query: str, transcript: str = "") -> tuple[list[dict], dict]:
    """
    Retrieval without routing.

    rewrite -> multi query -> HyDE -> hybrid search per query -> RRF fusion
            -> cross encoder rerank -> contextual compression

    Called directly by the agent's executor, which already knows it wants a
    document lookup because the planner said so.
    """

    trace = empty_trace()
    trace["used"] = True

    if config.USE_QUERY_REWRITE or config.USE_MULTI_QUERY:
        trace["llm_calls"] += 1

    queries, rewritten_query = build_query_set(query, transcript)
    trace["queries"] = queries
    trace["rewritten_query"] = rewritten_query

    hit_lists = [
        search(candidate, top_k=config.CANDIDATE_TOP_K)
        for candidate in queries
    ]

    hypothetical_document = generate_hypothetical_document(rewritten_query)

    if config.USE_HYDE:
        trace["llm_calls"] += 1

    if hypothetical_document:
        trace["hyde_used"] = True
        trace["hyde_document"] = hypothetical_document

        # HyDE is a vector only trick, the invented passage would only add
        # noise to BM25.
        hit_lists.append(
            vector_search(hypothetical_document, top_k=config.CANDIDATE_TOP_K)
        )

    fused_hits = merge_results(hit_lists, config.CANDIDATE_TOP_K)
    trace["fused"] = len(fused_hits)

    if not fused_hits:
        return [], trace

    # Rerank and compress against the standalone rewrite, since both stages
    # judge relevance and a bare follow up like "how many days?" tells them
    # nothing.
    reranked_hits = rerank(rewritten_query, fused_hits, top_k=config.RERANK_TOP_K)
    trace["reranked"] = len(reranked_hits)

    if config.USE_COMPRESSION and config.COMPRESSION_MODE == "llm":
        trace["llm_calls"] += len(reranked_hits)

    compressed_hits = compress(rewritten_query, reranked_hits)
    trace["compressed"] = any(
        hit.get("compressed")
        for hit in compressed_hits
    )

    return compressed_hits, trace


def retrieve(question: str, transcript: str = EMPTY_TRANSCRIPT) -> tuple[list[dict], dict]:
    """
    Route first, then retrieve if the question needs the documents.
    """

    decision = route(question, transcript)

    if decision["route"] != DOCUMENT_ROUTE:
        trace = empty_trace()
        trace["route"] = decision["route"]
        trace["route_reason"] = decision["reason"]
        trace["llm_calls"] = 1

        return [], trace

    hits, trace = retrieve_query(question, transcript)

    trace["route"] = decision["route"]
    trace["route_reason"] = decision["reason"]
    trace["llm_calls"] += 1

    return hits, trace
