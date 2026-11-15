"""Calm rule bookkeeping: nothing already shown moves; new points wait behind a marker.

Pure and UI-free. Points are matched by card key, so a later rewrite of a card's
text never hides it again. A key the engine drops altogether is pruned, because
there is nothing left to draw.
"""
from __future__ import annotations

from typing import Iterable


class RevealState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """A new recording session starts from nothing."""
        self.shown: list[str] = []
        self.pending: list[str] = []
        self.divider_key: str | None = None  # first point of the last batch
        self.divider_label: str = ""
        self.batch: list[str] = []  # points that entered with the last click

    def observe(self, keys: Iterable[str]) -> None:
        """Feed the keys of the current update, in engine order."""
        current = list(dict.fromkeys(keys))
        present = set(current)
        self.shown = [k for k in self.shown if k in present]
        self.pending = [k for k in self.pending if k in present]
        self.batch = [k for k in self.batch if k in present]
        if self.divider_key is not None and self.divider_key not in self.shown:
            self.divider_key = self.batch[0] if self.batch else None
            if self.divider_key is None:
                self.divider_label = ""
        known = set(self.shown) | set(self.pending)
        fresh = [k for k in current if k not in known]
        if not fresh:
            return
        if not self.shown and not self.pending:
            self.shown.extend(fresh)  # an empty list has nothing that could move
        else:
            self.pending.extend(fresh)

    @property
    def pending_count(self) -> int:
        return len(self.pending)

    @property
    def marker_text(self) -> str | None:
        n = len(self.pending)
        if not n:
            return None
        return "1 neuer Punkt ↓" if n == 1 else f"{n} neue Punkte ↓"

    def reveal(self, time_label: str) -> int:
        """The reader clicked the marker. Returns how many points entered."""
        if not self.pending:
            return 0
        entered = len(self.pending)
        self.divider_key = self.pending[0]
        self.divider_label = f"Neu seit {time_label}"
        self.batch = list(self.pending)
        self.shown.extend(self.pending)
        self.pending = []
        return entered

    def divider_before(self, column_keys: Iterable[str]) -> str | None:
        """Within one column (ordered keys), the point the 'Neu seit' line sits above.
        None when the batch starts the column: a divider at the very top says nothing."""
        for index, key in enumerate(column_keys):
            if key in self.batch:
                return key if index > 0 else None
        return None

    def is_shown(self, key: str) -> bool:
        return key in self.shown
