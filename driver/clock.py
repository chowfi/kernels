"""Active-vs-wall run clock, persisted to JSON so it survives driver restarts."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

OFFLINE_THRESHOLD_S = 120.0


class RunClock:
    """State JSON: start, pauses [{start,end,reason}], paused_seconds (closed
    pauses, incl. offline gaps), paused_since/paused_reason (open pause),
    last_heartbeat. Reads (active_seconds etc.) never write."""

    def __init__(self, state_path: Path, now: Callable[[], float] = time.time):
        self.path = Path(state_path)
        self._now = now
        self._s = {"start": None, "pauses": [], "paused_seconds": 0.0,
                   "paused_since": None, "paused_reason": None, "last_heartbeat": None}
        if self.path.exists():
            self._s.update(json.loads(self.path.read_text()))
            t = self._now()
            hb = self._s["last_heartbeat"]
            if (self._s["start"] is not None and self._s["paused_since"] is None
                    and hb is not None and t - hb > OFFLINE_THRESHOLD_S):
                self._add_pause(hb, t, "driver_offline")
                self._s["last_heartbeat"] = t
                self._save()

    # -- properties
    @property
    def started(self) -> bool:
        return self._s["start"] is not None

    @property
    def is_paused(self) -> bool:
        return self._s["paused_since"] is not None

    # -- internals
    def _add_pause(self, a: float, b: float, reason: str) -> None:
        self._s["pauses"].append({"start": a, "end": b, "reason": reason})
        self._s["paused_seconds"] += b - a

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self._s, indent=2))
        os.replace(tmp, self.path)

    # -- API
    def start(self) -> None:
        """Begin the clock. No-op (resume semantics) if state already has a start."""
        if self.started:
            return
        t = self._now()
        self._s["start"] = t
        self._s["last_heartbeat"] = t
        self._save()

    def pause(self, reason: str) -> None:
        if not self.started or self.is_paused:
            return
        t = self._now()
        self._s["paused_since"] = t
        self._s["paused_reason"] = reason
        self._s["last_heartbeat"] = t
        self._save()

    def resume(self) -> None:
        if not self.is_paused:
            return
        t = self._now()
        self._add_pause(self._s["paused_since"], t, self._s["paused_reason"])
        self._s["paused_since"] = None
        self._s["paused_reason"] = None
        self._s["last_heartbeat"] = t
        self._save()

    def heartbeat(self) -> None:
        if not self.started:
            return
        self._s["last_heartbeat"] = self._now()
        self._save()

    def wall_seconds(self) -> float:
        if not self.started:
            return 0.0
        return self._now() - self._s["start"]

    def active_seconds(self) -> float:
        if not self.started:
            return 0.0
        paused = self._s["paused_seconds"]
        if self.is_paused:
            paused += self._now() - self._s["paused_since"]
        return self.wall_seconds() - paused
