"""Session 02: the raw model call becomes a LangGraph StateGraph.

Same product, real architecture — typed state, a router with conditional
edges, checkpointed conversation threads, and one honest human-in-the-loop
interrupt. (Old tutorials build agents with `langgraph.prebuilt
.create_react_agent`; that is deprecated in 1.x in favor of
`langchain.agents.create_agent`. We wire the graph by hand because seeing
the nodes and edges is the lesson.)
"""

import os
from typing import Annotated, Literal, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Old tutorials import MemorySaver — renamed InMemorySaver in 1.x, same class.
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .llm import SYSTEM_PROMPT  # the Session 01 prompt survives the rearchitecture


class DeskState(TypedDict):
    # `add_messages` is a reducer: what a node returns is *appended* to the
    # history instead of replacing it. Checkpointed per thread, this list is
    # exactly the memory Session 01 lacked.
    messages: Annotated[list, add_messages]
    route: str
    clarifying_question: Optional[str]


class RouteDecision(BaseModel):
    """Structured output for the router: the model must pick a label, not prose."""

    route: Literal["general", "market", "filings", "unclear"] = Field(
        description="Where the latest user message should go."
    )
    clarifying_question: Optional[str] = Field(
        default=None,
        description="Only when route is 'unclear': the ONE question to ask the user.",
    )


llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"])

ROUTER_PROMPT = (
    "You route requests arriving at an equity research desk. Classify the "
    "user's LATEST message given the conversation so far: 'market' = live "
    "prices, quotes, recent moves; 'filings' = annual reports, 10-Ks, "
    "disclosures; 'general' = anything answerable directly; 'unclear' = it "
    "cannot be acted on without asking the user one clarifying question "
    "(ambiguous reference, missing company, etc.)."
)


def announce(name: str) -> None:
    # Nodes announce themselves on LangGraph's custom stream channel; the
    # frontend renders the NodeTrail from these events. This is the visible
    # pipeline — the UI grows a layer as the architecture does.
    get_stream_writer()({"type": "node", "name": name})


def router(state: DeskState) -> dict:
    announce("router")
    decision = llm.with_structured_output(RouteDecision).invoke(
        [("system", ROUTER_PROMPT), *state["messages"]]
    )
    return {"route": decision.route, "clarifying_question": decision.clarifying_question}


def clarify(state: DeskState) -> dict:
    announce("clarify")
    # interrupt() pauses the graph mid-run and persists the thread in the
    # checkpointer; nothing resumes until /api/chat/resume feeds the human's
    # answer back via Command(resume=...). HITL as a graph primitive, not an
    # if-statement. NB: on resume this whole node re-executes from the top —
    # keep everything before interrupt() cheap and deterministic.
    answer = interrupt({"question": state["clarifying_question"] or "Could you clarify?"})
    return {"messages": [HumanMessage(content=answer)]}


def respond(state: DeskState) -> dict:
    announce("respond")
    # 'filings' and 'market' land here too for now: the routing skeleton
    # comes before the capabilities (retrieval in S03, tools in S04).
    reply = llm.invoke([("system", SYSTEM_PROMPT), *state["messages"]])
    return {"messages": [reply]}


builder = StateGraph(DeskState)
builder.add_node("router", router)
builder.add_node("clarify", clarify)
builder.add_node("respond", respond)
builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "general": "respond",
        "filings": "respond",  # Session 03: the retrieval path
        "market": "respond",  # Session 04: the tool-enabled path
        "unclear": "clarify",
    },
)
builder.add_edge("clarify", "respond")
builder.add_edge("respond", END)

# In-process checkpoints: perfect for a classroom, gone on restart.
# Session 07 swaps in a durable checkpointer without touching the graph.
graph = builder.compile(checkpointer=InMemorySaver())
