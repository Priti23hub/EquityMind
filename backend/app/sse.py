"""SSE envelope helpers.

Server-Sent Events is just a text protocol over one long-lived HTTP response:
each event is a `data: <payload>` line followed by a blank line. We put one
JSON object in every event, with a `type` field the frontend switches on.
The event vocabulary grows by one session's worth of types at a time — see
"Streaming protocol" in the README. That growth path is the architecture:
the frontend parses this envelope once, then only ever adds renderers.

Introduced in Session 01.
"""

import json


def event(type: str, **payload) -> str:  # noqa: A002 — shadowing `type` lets dicts splat straight in
    """Format one SSE event: data: {"type": ..., ...}\\n\\n"""
    return f"data: {json.dumps({'type': type, **payload})}\n\n"
