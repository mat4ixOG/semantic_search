from ollama import chat
import config


MODEL_NAME = config.MODEL_NAME


def ask_llm(messages) -> str:
    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": messages
            }
        ],
        think=False,
    )

    return response.message.content


def stream_llm(messages):
    for chunk in chat(
            model=MODEL_NAME,
            messages=messages,
            think=False,
            stream=True,
    ):
        yield chunk.message.content or ""