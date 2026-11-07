"""Token accounting."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, o: "Usage") -> "Usage":
        return Usage(self.input_tokens + o.input_tokens, self.output_tokens + o.output_tokens,
                     self.cache_read_tokens + o.cache_read_tokens,
                     self.cache_write_tokens + o.cache_write_tokens)

    @property
    def total(self) -> int:
        return (self.input_tokens + self.output_tokens
                + self.cache_read_tokens + self.cache_write_tokens)


class TokenLedger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: list[dict] = []
        if self.path.exists():
            self.entries = json.loads(self.path.read_text())["entries"]

    def add(self, exp_id: int, u: Usage, session_id: str | None) -> None:
        self.entries.append({"exp_id": exp_id, "session_id": session_id, **u.__dict__})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"entries": self.entries, "totals": self.totals().__dict__}, indent=2))
        os.replace(tmp, self.path)

    def totals(self) -> Usage:
        t = Usage()
        for e in self.entries:
            t = t + Usage(e["input_tokens"], e["output_tokens"],
                          e["cache_read_tokens"], e["cache_write_tokens"])
        return t

    def session_ids(self) -> set[str]:
        return {e["session_id"] for e in self.entries if e.get("session_id")}
