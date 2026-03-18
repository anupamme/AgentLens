"""TraceWriter — file I/O for SessionTrace persistence."""

from __future__ import annotations

import threading
from pathlib import Path

from agentlens.schema.trace import SessionTrace


class TraceWriter:
    """Writes and reads SessionTrace objects to/from disk."""

    def __init__(self, output_dir: str = "./traces") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    def write_jsonl(self, trace: SessionTrace, filename: str = "traces.jsonl") -> None:
        path = self._output_dir / filename
        line = trace.model_dump_json() + "\n"
        with self._write_lock:
            with open(path, "a") as f:
                f.write(line)

    def write_json(self, trace: SessionTrace, filename: str | None = None) -> str:
        if filename is None:
            filename = f"{trace.session_id}.json"
        path = self._output_dir / filename
        with open(path, "w") as f:
            f.write(trace.to_json())
        return str(path)

    def read_traces(self, filename: str = "traces.jsonl") -> list[SessionTrace]:
        path = self._output_dir / filename
        traces: list[SessionTrace] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(SessionTrace.from_json(line))
        return traces

    def read_trace(self, filename: str) -> SessionTrace:
        path = self._output_dir / filename
        return SessionTrace.from_json(path.read_text())
