"""
Recovery-time measurement, as a reusable value type.

Phase 4 measured how long a recovery takes and printed a breakdown. That
number turns out to be an input to two other things, so it lives here rather
than inside the controller that happens to produce it:

  * **Phase 5's scheduler** needs the mean recovery cost (`R`) to estimate
    expected wasted compute.
  * **Phase 6's benchmark** needs the per-phase breakdown so it can report
    where recovery time goes under each checkpoint strategy.

The breakdown matters more than the total. On this testbed the total is ~98%
`reach_first_step` — a fresh interpreter importing torch, rebuilding the
model, loading the checkpoint and fast-forwarding the dataloader — and only
~2% controller bookkeeping. A benchmark that reported only the total would
make it look like the recovery *logic* was the cost, and send you optimizing
the wrong thing.

Timings are reconstructed from the controller's `recovery_log.jsonl` rather
than held in memory. That's not incidental: the process that would have held
them is the one that died.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# Ordered because the phases are sequential and a breakdown reads as a
# timeline; a dict comprehension over an unordered set would scramble it.
PHASES = ("terminate_workers", "select_checkpoint", "relaunch_workers",
          "reach_first_step")


@dataclass(frozen=True)
class RecoveryTiming:
    """One failure-to-resumed cycle, and where its time went."""

    attempt: int
    failed_rank: int
    failure_kind: str
    recovery_seconds: float
    terminate_workers: float = 0.0
    select_checkpoint: float = 0.0
    relaunch_workers: float = 0.0
    reach_first_step: float = 0.0
    exit_code: Optional[int] = None
    step_at_failure: Optional[int] = None
    restored_to_step: Optional[int] = None
    steps_lost: int = 0

    @property
    def breakdown(self) -> Dict[str, float]:
        return {phase: getattr(self, phase) for phase in PHASES}

    @property
    def accounted_seconds(self) -> float:
        """
        Sum of the phases.

        Should track `recovery_seconds` closely; a large gap means a phase
        boundary isn't being timed and the breakdown is lying by omission.
        """
        return sum(self.breakdown.values())

    def as_dict(self) -> dict:
        record = asdict(self)
        record["breakdown"] = self.breakdown
        return record

    @classmethod
    def from_event(cls, record: dict) -> "RecoveryTiming":
        """Rebuild from a `training_resumed` event in recovery_log.jsonl."""
        breakdown = record.get("breakdown") or {}
        return cls(
            attempt=record["attempt"],
            failed_rank=record["failed_rank"],
            failure_kind=record.get("failure_kind", "unknown"),
            recovery_seconds=record["recovery_seconds"],
            exit_code=record.get("exit_code"),
            step_at_failure=record.get("step_at_failure"),
            restored_to_step=record.get("restored_to_step"),
            steps_lost=record.get("steps_lost", 0),
            **{phase: float(breakdown.get(phase, 0.0)) for phase in PHASES},
        )


def load_recovery_timings(recovery_log: str) -> List[RecoveryTiming]:
    """
    Every completed recovery recorded in a job's controller log.

    Tolerates a truncated final line: the controller can itself be killed
    mid-write, and refusing to read the other N-1 recoveries because of it
    would throw away the measurement entirely.
    """
    if not os.path.exists(recovery_log):
        return []

    timings: List[RecoveryTiming] = []
    with open(recovery_log) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") != "training_resumed":
                continue
            try:
                timings.append(RecoveryTiming.from_event(record))
            except KeyError:
                continue
    return timings


def summarize(timings: List[RecoveryTiming]) -> dict:
    """
    Aggregate stats over a job's recoveries.

    Returns zeros-and-empties rather than raising on an empty list — a job
    that never failed is a legitimate benchmark data point, not an error.
    """
    if not timings:
        return {"recoveries": 0, "mean_seconds": 0.0, "min_seconds": 0.0,
                "max_seconds": 0.0, "total_seconds": 0.0, "steps_lost": 0,
                "mean_breakdown": {phase: 0.0 for phase in PHASES}}

    totals = [t.recovery_seconds for t in timings]
    return {
        "recoveries": len(timings),
        "mean_seconds": sum(totals) / len(totals),
        "min_seconds": min(totals),
        "max_seconds": max(totals),
        "total_seconds": sum(totals),
        "steps_lost": sum(t.steps_lost for t in timings),
        "mean_breakdown": {
            phase: sum(t.breakdown[phase] for t in timings) / len(timings)
            for phase in PHASES
        },
    }


def format_breakdown(breakdown: Dict[str, float], indent: str = "    ") -> str:
    """Render a breakdown as a timeline, widest phase last."""
    lines = []
    total = sum(breakdown.values()) or 1.0
    for phase in PHASES:
        seconds = breakdown.get(phase, 0.0)
        lines.append(f"{indent}{phase:<20} {seconds:>7.2f}s "
                     f"({100 * seconds / total:>4.1f}%)")
    return "\n".join(lines)
