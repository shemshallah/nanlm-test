"""Local CFT credit plus two-layer head/triad/field voting."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

HEADS = 256
TRIADS = 85
TRIAD_SIZE = 3
FIELD_THRESHOLD = 43


def triad_of(head: int) -> int:
    if not 0 <= head < HEADS:
        raise ValueError("head outside 0..255")
    return head // TRIAD_SIZE


def triad_members(triad: int) -> tuple[int, int, int]:
    if not 0 <= triad < TRIADS:
        raise ValueError("triad outside 0..84")
    base = triad * TRIAD_SIZE
    return base, base + 1, base + 2


class CFTField:
    """Small cycle-flux memory used as the base local credit."""

    def __init__(self):
        self.cycle = np.zeros((HEADS, HEADS), dtype=np.float32)

    def residual(self, triad: int) -> float:
        a, b, c = triad_members(triad)
        return float(self.cycle[a, b] + self.cycle[b, c] + self.cycle[c, a])

    def propose(self, triad: int, error: float, rate: float = 0.1) -> float:
        """CFT update: three edges change the loop by exactly the proposed delta."""
        a, b, c = triad_members(triad)
        delta = float(error) * rate / 3.0
        self.cycle[a, b] += delta
        self.cycle[b, c] += delta
        self.cycle[c, a] -= delta
        return self.residual(triad)


@dataclass
class HeadVote:
    head: int
    before_loss: float
    after_loss: float
    cft_before: float
    cft_after: float

    @property
    def improved(self) -> bool:
        return self.after_loss < self.before_loss


class TriadField:
    """Heads vote on triad changes; triads then vote on the global bundle."""

    def __init__(self, cft: CFTField | None = None):
        self.cft = cft or CFTField()
        self.head_votes: list[HeadVote] = []
        self.triad_votes: dict[int, bool] = {}

    def vote_head(self, head: int, before_loss: float, after_loss: float,
                  cft_before: float, cft_after: float) -> bool:
        vote = HeadVote(head, float(before_loss), float(after_loss),
                        float(cft_before), float(cft_after))
        # A head must improve its own loss and not make the triad CFT worse.
        cft_ok = abs(cft_before) < 1e-12 or abs(cft_after) <= abs(cft_before) + 1e-9
        approved = vote.improved and cft_ok
        self.head_votes.append(vote)
        return approved

    def vote_triads(self) -> dict[int, bool]:
        votes: dict[int, list[bool]] = {}
        for vote in self.head_votes:
            cft_ok = (abs(vote.cft_before) < 1e-12 or
                      abs(vote.cft_after) <= abs(vote.cft_before) + 1e-9)
            if vote.improved and cft_ok:
                votes.setdefault(triad_of(vote.head), []).append(True)
            else:
                votes.setdefault(triad_of(vote.head), []).append(False)
        self.triad_votes = {
            triad: len(values) >= TRIAD_SIZE and sum(values) >= 2
            for triad, values in votes.items()
        }
        return self.triad_votes

    def field_vote(self, *, require_full: bool = True) -> tuple[bool, int, int]:
        triads = self.vote_triads()
        accepted = sum(1 for value in triads.values() if value)
        total = len(triads)
        if require_full:
            return accepted >= FIELD_THRESHOLD and total == TRIADS, accepted, total
        # Candidate-only mode for reduced-head smoke tests.
        return accepted >= max(1, (2 * total + 2) // 3), accepted, total
