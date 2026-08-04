import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import config
import service
import store
from vector_store import count as chunk_count

app = FastAPI(
    title="Semantic Search RAG",
    description="Hybrid retrieval with query expansion, HyDE, reranking, "
                "compression and a plan-and-execute agent.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in config.CORS_ORIGINS.split(",")
        if origin.strip()
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    conversation_id: str | None = None
    # None means "whatever config says"
    agent: bool | None = None


@app.get("/api/health")
def health():
    """
    Cheap enough to poll. Reports whether anything has been ingested, since
    an empty collection is the most common reason for useless answers.
    """

    try:
        chunks = chunk_count()
    except Exception as error:
        raise HTTPException(503, f"vector store unavailable: {error}")

    return {
        "status": "ok",
        "chunks": chunks,
        "ingested": chunks > 0,
    }


@app.get("/api/config")
def get_config():
    return service.settings()


@app.get("/api/conversations")
def list_conversations(limit: int = 50):
    return {"conversations": store.list_conversations(limit)}


@app.post("/api/conversations")
def create_conversation():
    return store.create_conversation()


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    if not store.conversation_exists(conversation_id):
        raise HTTPException(404, "conversation not found")

    return {
        "id": conversation_id,
        "messages": store.get_messages(conversation_id),
    }


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    if not store.delete_conversation(conversation_id):
        raise HTTPException(404, "conversation not found")

    return {"deleted": conversation_id}


def resolve_conversation(conversation_id: str | None) -> str:
    """
    An unknown or missing id starts a new conversation rather than failing,
    so a client that lost its state can just keep talking.
    """

    if conversation_id and store.conversation_exists(conversation_id):
        return conversation_id

    return store.create_conversation()["id"]


@app.post("/api/ask")
def ask(request: AskRequest):
    conversation_id = resolve_conversation(request.conversation_id)
    history = store.history_for(conversation_id)

    try:
        payload = service.ask(request.question, history, request.agent)
    except service.PipelineError as error:
        raise HTTPException(503, f"model backend unavailable: {error}")

    store.add_message(conversation_id, "user", request.question)
    store.add_message(conversation_id, "assistant", payload["answer"], payload)

    payload["conversation_id"] = conversation_id

    return payload


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.post("/api/ask/stream")
def ask_stream(request: AskRequest):
    conversation_id = resolve_conversation(request.conversation_id)
    history = store.history_for(conversation_id)

    def events():
        yield sse({"type": "start", "conversation_id": conversation_id})

        payload = None

        try:
            for event in service.ask_stream(request.question, history, request.agent):
                if event["type"] == "done":
                    payload = event["payload"]
                    payload["conversation_id"] = conversation_id

                yield sse(event)
        except service.PipelineError as error:
            # The response has already started, so the failure has to travel
            # as an event rather than as a status code.
            yield sse({"type": "error", "message": str(error)})
            return

        if payload:
            store.add_message(conversation_id, "user", request.question)
            store.add_message(
                conversation_id,
                "assistant",
                payload["answer"],
                payload,
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
