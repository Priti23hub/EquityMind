"""EquityMind API — one streaming chat endpoint, plus resume for HITL.

The SSE event vocabulary grows one session at a time (see README,
"Streaming protocol"). S01: token/done/error · S02: + node/interrupt.
"""


import os

from dotenv import load_dotenv

load_dotenv()  # backend/.env — secrets stay out of code and out of git

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from langgraph.types import Command  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from .graph import graph  # noqa: E402  (imported after the env is loaded)
from .guardrails import StreamingOutputGuardrail, validate_input  # noqa: E402
from .sse import event  # noqa: E402

app = FastAPI(title="EquityMind")

# The Vite dev server is a different origin, so the browser preflights our
# POST; without these headers the stream never starts.
allowed_origins = ["http://localhost:8501"]
if frontend_origin := os.getenv("FRONTEND_ORIGIN"):
    allowed_origins.append(frontend_origin.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    # Vite increments the port when 5173 is already occupied. Keep local
    # development working on that fallback port (and when opened via
    # 127.0.0.1) without allowing non-local origins.
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):517\d+$",
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str  # client-generated; names this conversation's checkpoint


class ResumeRequest(BaseModel):
    thread_id: str
    answer: str


async def stream_graph(graph_input, thread_id: str):
    # thread_id selects which checkpointed conversation this run extends —
    # state lives server-side in the checkpointer, not in the request.
    config = {"configurable": {"thread_id": thread_id}}
    output_guardrail = StreamingOutputGuardrail()
    try:
        # LangGraph's current streaming API: astream with multiple modes.
        #   "messages" → per-token chunks from LLM calls inside nodes
        #   "custom"   → whatever nodes write via get_stream_writer()
        #   "updates"  → node results; also where an interrupt surfaces
        async for mode, chunk in graph.astream(
            graph_input, config, stream_mode=["messages", "custom", "updates"]
        ):
            if mode == "custom":
                yield event(**chunk)
            elif mode == "messages":
                token, meta = chunk
                # Only the respond node's tokens are the answer; the router's
                # structured-output call also streams here and must not leak.
                if token.content and meta.get("langgraph_node") == "respond":
                    safe_text, blocked = output_guardrail.feed(token.content)
                    if safe_text:
                        yield event("token", text=safe_text)
                    if blocked:
                        yield event(**blocked.as_event())
                        return
            elif mode == "updates" and "__interrupt__" in chunk:
                yield event("interrupt", question=chunk["__interrupt__"][0].value["question"])
        safe_text, blocked = output_guardrail.finish()
        if blocked:
            yield event(**blocked.as_event())
            return
        if safe_text:
            yield event("token", text=safe_text)
        yield event("done")
    except Exception as exc:
        # A failure mid-stream must be an event, not a dead socket.
        yield event("error", message=str(exc))


async def blocked_stream(result):
    yield event(**result.as_event())


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    input_check = validate_input(req.message)
    if not input_check.allowed:
        return StreamingResponse(blocked_stream(input_check), media_type="text/event-stream")
    return StreamingResponse(
        stream_graph({"messages": [("user", req.message)]}, req.thread_id),
        media_type="text/event-stream",
    )


@app.post("/api/chat/resume")
def resume(req: ResumeRequest) -> StreamingResponse:
    # Command(resume=...) hands the human's answer to the interrupt() call
    # that paused this thread; the graph picks up exactly where it stopped.
    input_check = validate_input(req.answer)
    if not input_check.allowed:
        return StreamingResponse(blocked_stream(input_check), media_type="text/event-stream")
    return StreamingResponse(
        stream_graph(Command(resume=req.answer), req.thread_id),
        media_type="text/event-stream",
    )
