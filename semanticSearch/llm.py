import json

from ollama import ResponseError, chat
import config


MODEL_NAME = config.MODEL_NAME

UTILITY_MODEL = config.UTILITY_MODEL

# Models we asked for and ollama does not have. Remembered so the warning is
# printed once instead of on every call.
_missing_models = set()


def _chat(messages, model, **kwargs):
    """
    Call ollama, falling back to the main model when the requested one has
    not been pulled.
    """

    if model in _missing_models:
        model = MODEL_NAME

    try:
        return chat(
            model=model,
            messages=messages,
            think=False,
            **kwargs,
        )
    except ResponseError as error:
        if "not found" not in str(error).lower() or model == MODEL_NAME:
            raise

        _missing_models.add(model)
        print(f"[llm] {model} not pulled, falling back to {MODEL_NAME}")
        print(f"[llm] run `ollama pull {model}` to speed the pipeline up")

        return chat(
            model=MODEL_NAME,
            messages=messages,
            think=False,
            **kwargs,
        )


def ask_llm(messages , pref_model = MODEL_NAME) -> str:
    response = _chat(messages, pref_model)

    return response.message.content


def ask_llm_json(messages, schema, pref_model = MODEL_NAME) -> dict:
    """
    Ask the LLM for a response constrained to a JSON schema.
    Returns the parsed object, or None if the model produced invalid JSON.
    """

    response = _chat(messages, pref_model, format=schema)

    raw = response.message.content or ""

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def stream_llm(messages, pref_model = MODEL_NAME):
    for chunk in _chat(messages, pref_model, stream=True):
        yield chunk.message.content or ""
