"""
Shared helper for instrumenting the pipeline with progress events.

Every pipeline module (ingest.py, retriever.py, rag.py) calls emit() at
key points so it can emit events to whoever's listening (e.g. the web
server, which forwards them over SSE). If nobody's listening
(on_event=None, the case for chat.py and eval/evaluate.py), emit() does
nothing — the pipeline behaves exactly as it did before.
"""
from __future__ import annotations

import time
from typing import Callable

EventCallback = Callable[[str, dict], None]


def emit(on_event: EventCallback | None, name: str, **payload) -> None:
    if on_event is None:
        return
    on_event(name, {**payload, "ts": time.time()})
