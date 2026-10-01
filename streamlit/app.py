import json
import uuid

import requests
import streamlit as st


import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://host.docker.internal:8000")

def get_sse_events(response):
    """Yield JSON objects from the backend's one-line SSE data messages."""
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        try:
            yield json.loads(line[6:])
        except json.JSONDecodeError:
            yield {"type": "error", "message": "The backend sent invalid event data."}


def stream_request(endpoint: str, payload: dict, assistant_placeholder, node_placeholder):
    """Send one request and update the assistant message while it streams."""
    try:
        response = requests.post(
            f"{BACKEND_URL}{endpoint}",
            json=payload,
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Could not reach the EquityMind backend: {exc}")
        return "", None, []

    assistant_text = ""
    interrupt_question = None
    nodes = []

    for event in get_sse_events(response):
        event_type = event.get("type")
        if event_type == "token":
            assistant_text += event.get("text", "")
            assistant_placeholder.markdown(assistant_text)
        elif event_type == "node":
            node_name = event.get("name", "unknown")
            nodes.append(node_name)
            node_placeholder.caption(f"Pipeline: {' -> '.join(nodes)}")
        elif event_type == "interrupt":
            interrupt_question = event.get("question", "Please provide an answer.")
        elif event_type == "guardrail":
            st.warning(event.get("reason", "This request was blocked by an EquityMind guardrail."))
        elif event_type == "error":
            st.error(f"Backend error: {event.get('message', 'Unknown error')}")

    return assistant_text, interrupt_question, nodes


def initialise_state():
    """Create the browser-session state used by this conversation."""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "nodes" not in st.session_state:
        st.session_state.nodes = []


def send_message(message: str, endpoint: str = "/api/chat"):
    """Add a user message and stream the matching assistant response."""
    st.session_state.messages.append({"role": "user", "content": message})
    with st.chat_message("assistant"):
        assistant_placeholder = st.empty()
        node_placeholder = st.empty()
        payload = (
            {"message": message, "thread_id": st.session_state.thread_id}
            if endpoint == "/api/chat"
            else {"answer": message, "thread_id": st.session_state.thread_id}
        )
        assistant_text, question, nodes = stream_request(
            endpoint, payload, assistant_placeholder, node_placeholder
        )
    if assistant_text:
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
    st.session_state.pending_question = question
    st.session_state.nodes = nodes


st.set_page_config(page_title="EquityMind", page_icon="A", layout="centered")
initialise_state()

st.title("EquityMind")
st.caption("AI research desk")

with st.sidebar:
    st.subheader("Conversation")
    st.caption(f"Thread: {st.session_state.thread_id}")
    if st.button("Start new conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.session_state.nodes = []
        st.rerun()
    if st.session_state.nodes:
        st.subheader("Last pipeline")
        st.write(" -> ".join(st.session_state.nodes))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.pending_question:
    st.info(st.session_state.pending_question)
    with st.form("approval_form"):
        answer = st.text_input("Your answer")
        submitted = st.form_submit_button("Continue")
    if submitted:
        if answer.strip():
            st.session_state.pending_question = None
            send_message(answer.strip(), endpoint="/api/chat/resume")
            st.rerun()
        else:
            st.warning("Please enter an answer before continuing.")

prompt = st.chat_input("Ask EquityMind a question")
if prompt and not st.session_state.pending_question:
    send_message(prompt)
    st.rerun()