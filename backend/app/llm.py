"""Session 01: one raw, streaming model call. No framework, no memory, no tools.

The point of this file is what it *lacks*. We send only the incoming message,
so the model cannot remember earlier turns (fixed with graph state in
Session 02) and cannot look anything up (fixed with retrieval and tools in
Sessions 03–04). Its two visible failures are the curriculum.

Superseded by graph.py in Session 02; kept as the Session 01 artifact —
`git switch -d session-01` shows it in service. Its SYSTEM_PROMPT lives on.
"""

import os
from collections.abc import Iterator

from openai import OpenAI

# Reads OPENAI_API_KEY from the environment (main.py loads .env first).
client = OpenAI()

SYSTEM_PROMPT = (
    "You are EquityMind, an equity research assistant. Be concise and "
    "plain-spoken. When you are not certain of a fact — especially exact "
    "figures, dates, or anything that may have changed since your training "
    "data — say so plainly rather than guessing. Never give investment advice."
)


def stream_reply(message: str) -> Iterator[str]:
    """Yield the model's reply token by token."""
    stream = client.chat.completions.create(
        model=os.environ["OPENAI_MODEL"],
        # Statelessness is deliberate: the API is stateless, and we send no
        # history — every request is the model's first. Session 02 fixes this
        # with checkpointed conversation state.
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        # stream=True flips the response from one JSON blob to a stream of
        # deltas as tokens are generated; we forward each one over SSE.
        # (OpenAI's newer Responses API streams the same way; we use Chat
        # Completions because it is what LangChain wraps in Session 02.)
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
