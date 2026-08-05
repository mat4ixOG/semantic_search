import re

import config
from llm import UTILITY_MODEL, ask_llm_json
from prompts import QUERY_EXPANSION_SCHEMA, query_expansion_messages

# "1. ", "2) ", "- ", "* ", "• "
_BULLET_PATTERN = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•])\s*")

# "Rewritten query: ", "Query - ", "Search query: "
_LABEL_PATTERN = re.compile(
    r"^\s*(?:rewritten\s+quer(?:y|ies)|quer(?:y|ies)|search\s+query)\s*[:\-]\s*",
    re.IGNORECASE,
)


def clean_query(line: str) -> str:
    """
    Strip the decoration small models like to add around a generated query.
    """

    query = line.strip()
    query = _BULLET_PATTERN.sub("", query)
    query = _LABEL_PATTERN.sub("", query)

    return query.strip().strip('"').strip("'").strip()


def is_usable_query(query: str) -> bool:
    return bool(query) and len(query) <= config.MAX_QUERY_LENGTH


def expand_query(question: str, transcript: str = "") -> tuple[str, list[str]]:
    """
    Rewrite the question and generate multi query variations in one LLM call.

    The transcript lets the rewrite resolve follow ups like "how many days of
    it do I get?" into a standalone query.

    Returns (rewritten_query, variations). The rewrite falls back to the
    original question whenever the model misbehaves, and variations to [].
    """

    if not config.USE_QUERY_REWRITE and not config.USE_MULTI_QUERY:
        return question, []

    count = config.MULTI_QUERY_COUNT if config.USE_MULTI_QUERY else 0

    messages = query_expansion_messages(question, count, transcript)

    response = ask_llm_json(
        messages=messages,
        schema=QUERY_EXPANSION_SCHEMA,
        pref_model=UTILITY_MODEL,
        max_tokens=config.EXPANSION_MAX_TOKENS,
    )

    if not response:
        return question, []

    rewritten_query = question

    if config.USE_QUERY_REWRITE:
        candidate = clean_query(response.get("rewritten") or "")

        if is_usable_query(candidate):
            rewritten_query = candidate

    if not config.USE_MULTI_QUERY:
        return rewritten_query, []

    variations = []
    seen = {question.lower(), rewritten_query.lower()}

    for raw in response.get("variations") or []:
        if not isinstance(raw, str):
            continue

        variation = clean_query(raw)

        if not is_usable_query(variation) or variation.lower() in seen:
            continue

        seen.add(variation.lower())
        variations.append(variation)

    return rewritten_query, variations[:count]


def rewrite(question: str, transcript: str = "") -> str:
    """
    Rewrite only, for callers that do not want the variations.
    """

    rewritten_query, _ = expand_query(question, transcript)

    return rewritten_query
