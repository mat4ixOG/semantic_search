import time

import config
from agent import run_agent
from memory import ConversationMemory
from responder import answer_question, describe_hit, stream_answer
from retrieval import retrieve


class PipelineError(RuntimeError):
    """
    The LLM backend is unreachable or refused the request. Raised so the API
    can answer 503 instead of leaking a stack trace.
    """


def resolve(value, default):
    return default if value is None else value


def gather(question: str, transcript: str, agent_mode: bool):
    """
    Retrieval, through the agent or the plain pipeline.
    Returns (hits, tool_notes, trace).
    """

    if agent_mode:
        return run_agent(question, transcript)

    hits, trace = retrieve(question, transcript)

    return hits, [], trace


def build_payload(question, answer, trace, hits, tool_notes, elapsed=None):
    payload = {
        "question": question,
        "answer": answer["answer"],
        "answered": answer["answered"],
        "sources": answer["sources"],
    }

    if config.JSON_INCLUDE_TRACE:
        payload["retrieval"] = {
            **trace,
            "tool_notes": tool_notes,
            "chunks": [describe_hit(hit) for hit in hits],
        }

        if elapsed is not None:
            payload["retrieval"]["seconds"] = round(elapsed, 2)

    return payload


def ask(question: str, history: list[dict] = None, agent_mode: bool = None) -> dict:
    """
    One question, one complete JSON answer. The entry point for the API and
    for anything that does not need token streaming.
    """

    agent_mode = resolve(agent_mode, config.USE_AGENT)
    memory = ConversationMemory(history=history)

    started = time.perf_counter()

    try:
        hits, tool_notes, trace = gather(
            question,
            memory.as_transcript(),
            agent_mode,
        )

        answer = answer_question(
            question,
            hits,
            memory.get_history(),
            tool_notes,
        )
    except PipelineError:
        raise
    except Exception as error:
        raise PipelineError(str(error)) from error

    elapsed = time.perf_counter() - started

    return build_payload(question, answer, trace, hits, tool_notes, elapsed)


def ask_stream(question: str, history: list[dict] = None, agent_mode: bool = None):
    """
    The same pipeline, as a stream of events.

    Retrieval happens first and is emitted as a `retrieval` event, so a UI can
    show routing, the plan and the sources while the answer is still being
    written. Then `token` events, then a final `done` event carrying the whole
    payload.
    """

    agent_mode = resolve(agent_mode, config.USE_AGENT)
    memory = ConversationMemory(history=history)

    started = time.perf_counter()

    try:
        hits, tool_notes, trace = gather(
            question,
            memory.as_transcript(),
            agent_mode,
        )
    except Exception as error:
        raise PipelineError(str(error)) from error

    yield {
        "type": "retrieval",
        "trace": trace,
        "tool_notes": tool_notes,
        "chunks": [describe_hit(hit) for hit in hits],
    }

    answer = ""

    try:
        for chunk in stream_answer(question, hits, memory.get_history(), tool_notes):
            answer += chunk

            yield {
                "type": "token",
                "text": chunk,
            }
    except Exception as error:
        raise PipelineError(str(error)) from error

    elapsed = time.perf_counter() - started

    # Streaming produces prose, so there are no cited excerpt numbers to
    # resolve. Everything retrieved is reported as the sources instead.
    payload = build_payload(
        question,
        {
            "answer": answer.strip(),
            "answered": bool(answer.strip()),
            "sources": [describe_hit(hit) for hit in hits],
        },
        trace,
        hits,
        tool_notes,
        elapsed,
    )

    yield {
        "type": "done",
        "payload": payload,
    }


def settings() -> dict:
    """
    What the pipeline is currently configured to do, for the UI to display.
    """

    return {
        "model": config.MODEL_NAME,
        "utility_model": config.UTILITY_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "reranker": config.CROSS_ENCODER_MODEL,
        "agent": config.USE_AGENT,
        "query_rewrite": config.USE_QUERY_REWRITE,
        "multi_query": config.USE_MULTI_QUERY,
        "hyde": config.USE_HYDE,
        "compression": config.USE_COMPRESSION,
        "compression_mode": config.COMPRESSION_MODE,
    }
