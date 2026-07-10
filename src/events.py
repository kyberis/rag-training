"""
Helper compartido para instrumentar el pipeline con eventos de progreso.

Cada módulo del pipeline (ingest.py, retriever.py, rag.py) llama a emit()
en los puntos clave para poder emitir eventos hacia quien esté escuchando
(por ejemplo, el server web que los transmite por SSE). Si no hay nadie
escuchando (on_event=None, el caso de chat.py y eval/evaluate.py), emit()
no hace nada — el pipeline se comporta exactamente igual que antes.
"""
from __future__ import annotations

import time
from typing import Callable

EventCallback = Callable[[str, dict], None]


def emit(on_event: EventCallback | None, name: str, **payload) -> None:
    if on_event is None:
        return
    on_event(name, {**payload, "ts": time.time()})
