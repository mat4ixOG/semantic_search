from llm import UTILITY_MODEL, ask_llm_json
from prompts import (
    DIRECT_ROUTE,
    DOCUMENT_ROUTE,
    MEMORY_ROUTE,
    ROUTE_SCHEMA,
    route_messages,
)

ROUTES = {DOCUMENT_ROUTE, MEMORY_ROUTE, DIRECT_ROUTE}

EMPTY_TRANSCRIPT = "(no conversation yet)"

DEFAULT_ROUTE = DOCUMENT_ROUTE


def route(question: str, transcript: str = EMPTY_TRANSCRIPT) -> dict:
    """
    Decide whether a question is answered from the documents, from the
    conversation, or directly.

    The decision is a schema constrained JSON reply, so there is no one word
    output to parse and no way to come back with a route outside the enum.
    """

    messages = route_messages(question, transcript)

    decision = ask_llm_json(
        messages=messages,
        schema=ROUTE_SCHEMA,
        pref_model=UTILITY_MODEL,
    )

    if not decision:
        return {
            "route": DEFAULT_ROUTE,
            "reason": "classifier returned invalid JSON",
        }

    chosen = (decision.get("route") or "").strip().lower()

    if chosen not in ROUTES:
        return {
            "route": DEFAULT_ROUTE,
            "reason": f"unknown route {chosen!r}",
        }

    return {
        "route": chosen,
        "reason": (decision.get("reason") or "").strip(),
    }


def should_retrieve(question: str, transcript: str = EMPTY_TRANSCRIPT) -> bool:
    return route(question, transcript)["route"] == DOCUMENT_ROUTE
