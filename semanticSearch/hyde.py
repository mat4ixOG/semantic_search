import config
from llm import UTILITY_MODEL, ask_llm
from prompts import hyde_messages


def generate_hypothetical_document(question: str) -> str | None:
    """
    HyDE: write a fake answer passage for the question.

    The passage is never shown to the user. It is embedded and used as the
    search query, because a hypothetical answer sits closer in embedding
    space to the real answer than the question does.

    Returns None when HyDE is disabled or the model returned nothing usable.
    """

    if not config.USE_HYDE:
        return None

    messages = hyde_messages(question)

    document = ask_llm(
        messages=messages,
        pref_model=UTILITY_MODEL,
    )

    document = (document or "").strip()

    return document or None
