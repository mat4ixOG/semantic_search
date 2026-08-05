import config
from llm import answer_model, ask_llm_json, stream_llm
from prompts import ANSWER_SCHEMA, build_chat_messages, build_messages

FALLBACK_ANSWER = "I don't know."


def build_answer_messages(question, hits, history, json_mode, tool_notes=None):
    if hits or tool_notes:
        return build_messages(
            question,
            hits,
            history,
            json_mode=json_mode,
            tool_notes=tool_notes,
        )

    return build_chat_messages(question, history, json_mode=json_mode)


def answer_question(question: str, hits: list[dict], history: list[dict],
                    tool_notes: list[str] = None, fast: bool = False) -> dict:
    """
    JSON mode. Ask for a schema constrained answer and resolve the excerpt
    numbers the model cited back into real document metadata.
    """

    messages = build_answer_messages(
        question,
        hits,
        history,
        json_mode=True,
        tool_notes=tool_notes,
    )

    response = ask_llm_json(
        messages=messages,
        schema=ANSWER_SCHEMA,
        pref_model=answer_model(fast),
        max_tokens=config.ANSWER_MAX_TOKENS,
    )

    if not response:
        return {
            "answer": FALLBACK_ANSWER,
            "answered": False,
            "sources": [],
        }

    answer = (response.get("answer") or "").strip() or FALLBACK_ANSWER
    answered = bool(response.get("answered")) and answer != FALLBACK_ANSWER

    return {
        "answer": answer,
        "answered": answered,
        "sources": resolve_sources(response.get("sources"), hits) if answered else [],
    }


def stream_answer(question: str, hits: list[dict], history: list[dict],
                  tool_notes: list[str] = None, fast: bool = False):
    """
    Text mode. Yields the answer as it is generated.
    """

    messages = build_answer_messages(
        question,
        hits,
        history,
        json_mode=False,
        tool_notes=tool_notes,
    )

    yield from stream_llm(
        messages,
        pref_model=answer_model(fast),
        max_tokens=config.ANSWER_MAX_TOKENS,
    )


def resolve_sources(cited, hits: list[dict]) -> list[dict]:
    """
    The model cites excerpts by number, so page and document never come from
    the model itself. Unknown numbers are dropped.
    """

    if not cited or not hits:
        return []

    sources = []
    seen = set()

    for number in cited:
        if not isinstance(number, int) or not 1 <= number <= len(hits):
            continue

        if number in seen:
            continue

        seen.add(number)
        sources.append(describe_hit(hits[number - 1]))

    return sources


def describe_hit(hit: dict) -> dict:
    return {
        "document": hit["document"],
        "page": hit["page"],
        "chunk_id": hit["id"],
        "rerank_score": hit.get("rerank_score"),
        "text": hit["text"],
    }
