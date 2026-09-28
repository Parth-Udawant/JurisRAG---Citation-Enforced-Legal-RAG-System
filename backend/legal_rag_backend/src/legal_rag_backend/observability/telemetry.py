from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from datetime import date
from time import perf_counter
from uuid import uuid4

logger = logging.getLogger("legal_rag")


class RequestTelemetry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day = date.today()
        self._requests = 0
        self._counters = defaultdict(int)

    def new_request_id(self) -> str:
        return str(uuid4())

    def start(self) -> float:
        return perf_counter()

    def elapsed_ms(self, started: float) -> float:
        return round((perf_counter() - started) * 1000, 2)

    def record_request(self) -> int:
        with self._lock:
            today = date.today()
            if today != self._day:
                self._day = today
                self._requests = 0
                self._counters.clear()
            self._requests += 1
            return self._requests

    def increment(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[key] += amount

    def log(self, **payload) -> None:
        logger.info(json.dumps(payload, ensure_ascii=False, default=str))
