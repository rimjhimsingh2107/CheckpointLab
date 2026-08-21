"""
Per-job cost accounting for the Phase 6 benchmark.

Everything here is *measured* from a job's own logs rather than inferred from
averages. That distinction matters: a benchmark that multiplies "steps lost"
by a mean step time can be off by a lot when step times drift (and they do,
under disk pressure from 20 generations of checkpoint writes). Every quantity
below traces back to a timestamped record some process actually wrote.

WHERE A JOB'S WALL CLOCK GOES
-----------------------------

    wall_clock
      = useful_compute          steps that survived into the final result
      + wasted_recompute        steps computed, lost to a crash, recomputed
      + checkpoint_overhead     time spent writing checkpoints
      + fast_forward            replaying a resumed epoch's dataloader
      + recovery_latency        failure detected -> training resumed
      + unaccounted             interpreter startup, imports, model build

The first five are attributed; `unaccounted` is the residual and is reported
rather than hidden, because a residual that grows with the number of
restarts is evidence the attribution is incomplete.

Two of these are easy to leave out and both would flatter a strategy:

  * **wasted_recompute** is derived from *duplicate* global steps in the loss
    log. Every computed step writes one record, so a step appearing twice was
    computed twice, and the earlier record's own `step_seconds` is exactly
    what that redundant work cost. No estimation.

  * **fast_forward** is the cost of replaying a resumed epoch's dataloader to
    reach the checkpointed batch. Those batches get decoded and augmented
    only to be discarded, and it is caused directly by the checkpoint
    interval — a longer interval leaves more to replay. Omitting it would
    quietly subsidise the long-interval strategies.
"""
import csv
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from metrics.recovery_timing import PHASES, load_recovery_timings, summarize


def _read_jsonl(path: str) -> List[dict]:
    """Tolerant JSONL reader — a killed writer can leave a partial line."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


@dataclass
class JobMetrics:
    """One (strategy, scenario) run, fully costed."""

    strategy: str
    scenario: str
    failure_rate: float
    wall_clock_seconds: float

    steps_useful: int = 0
    steps_computed: int = 0
    steps_redundant: int = 0

    useful_compute_seconds: float = 0.0
    wasted_recompute_seconds: float = 0.0
    checkpoint_overhead_seconds: float = 0.0
    fast_forward_seconds: float = 0.0
    recovery_latency_seconds: float = 0.0

    checkpoints_written: int = 0
    mean_checkpoint_seconds: float = 0.0
    mean_interval_steps: float = 0.0
    fast_forward_steps: int = 0

    failures: int = 0
    recoveries: int = 0
    mean_recovery_seconds: float = 0.0
    recovery_breakdown: Dict[str, float] = field(default_factory=dict)

    outcome: str = "unknown"
    generations: int = 0
    job_dir: str = ""

    # ---------------- derived ----------------

    @property
    def wasted_total_seconds(self) -> float:
        """Everything that wasn't useful forward progress."""
        return (self.wasted_recompute_seconds
                + self.checkpoint_overhead_seconds
                + self.fast_forward_seconds
                + self.recovery_latency_seconds)

    @property
    def accounted_seconds(self) -> float:
        return self.useful_compute_seconds + self.wasted_total_seconds

    @property
    def unaccounted_seconds(self) -> float:
        return max(0.0, self.wall_clock_seconds - self.accounted_seconds)

    @property
    def checkpoint_overhead_pct(self) -> float:
        return self._pct(self.checkpoint_overhead_seconds)

    @property
    def wasted_pct(self) -> float:
        return self._pct(self.wasted_total_seconds)

    @property
    def goodput_pct(self) -> float:
        """Share of wall clock that produced surviving work."""
        return self._pct(self.useful_compute_seconds)

    @property
    def throughput_steps_per_s(self) -> float:
        if self.wall_clock_seconds <= 0:
            return 0.0
        return self.steps_useful / self.wall_clock_seconds

    def _pct(self, seconds: float) -> float:
        if self.wall_clock_seconds <= 0:
            return 0.0
        return 100.0 * seconds / self.wall_clock_seconds

    def to_row(self) -> dict:
        row = {k: v for k, v in asdict(self).items()
               if k != "recovery_breakdown"}
        row.update({
            "wasted_total_seconds": round(self.wasted_total_seconds, 3),
            "wasted_pct": round(self.wasted_pct, 2),
            "checkpoint_overhead_pct": round(self.checkpoint_overhead_pct, 2),
            "goodput_pct": round(self.goodput_pct, 2),
            "unaccounted_seconds": round(self.unaccounted_seconds, 3),
            "throughput_steps_per_s": round(self.throughput_steps_per_s, 4),
        })
        for phase in PHASES:
            row[f"recovery_{phase}_seconds"] = round(
                self.recovery_breakdown.get(phase, 0.0), 3)
        for key, value in list(row.items()):
            if isinstance(value, float):
                row[key] = round(value, 4)
        return row


def collect_job_metrics(job_dir: str, strategy: str, scenario: str,
                        failure_rate: float, wall_clock_seconds: float,
                        controller_summary: Optional[dict] = None) -> JobMetrics:
    """Read a finished job's logs and cost it out."""
    metrics = JobMetrics(strategy=strategy, scenario=scenario,
                         failure_rate=failure_rate,
                         wall_clock_seconds=wall_clock_seconds,
                         job_dir=job_dir)

    # ---- steps: useful vs recomputed -------------------------------
    loss_records = _read_jsonl(os.path.join(job_dir, "loss_log.jsonl"))
    metrics.steps_computed = len(loss_records)

    # Records are appended chronologically, so for a step computed more than
    # once the LAST occurrence is the one that survived; the earlier ones are
    # the work a crash threw away.
    last_index: Dict[int, int] = {}
    for index, record in enumerate(loss_records):
        last_index[record["global_step"]] = index

    metrics.steps_useful = len(last_index)
    metrics.steps_redundant = metrics.steps_computed - metrics.steps_useful

    surviving = set(last_index.values())
    for index, record in enumerate(loss_records):
        seconds = float(record.get("step_seconds") or 0.0)
        if index in surviving:
            metrics.useful_compute_seconds += seconds
        else:
            metrics.wasted_recompute_seconds += seconds

    # ---- checkpoint writes and dataloader replay -------------------
    ckpt_records = _read_jsonl(os.path.join(job_dir, "checkpoint_log.jsonl"))
    writes = [r for r in ckpt_records if r.get("event") == "checkpoint"]
    replays = [r for r in ckpt_records if r.get("event") == "fast_forward"]

    metrics.checkpoints_written = len(writes)
    metrics.checkpoint_overhead_seconds = sum(
        float(r.get("duration_seconds") or 0.0) for r in writes)
    if writes:
        metrics.mean_checkpoint_seconds = (
            metrics.checkpoint_overhead_seconds / len(writes))
        intervals = [r.get("interval_used") for r in writes
                     if r.get("interval_used")]
        if intervals:
            metrics.mean_interval_steps = sum(intervals) / len(intervals)

    metrics.fast_forward_seconds = sum(
        float(r.get("duration_seconds") or 0.0) for r in replays)
    metrics.fast_forward_steps = sum(
        int(r.get("steps_skipped") or 0) for r in replays)

    # ---- failures and recoveries -----------------------------------
    failures = _read_jsonl(os.path.join(job_dir, "failure_log.jsonl"))
    metrics.failures = sum(1 for r in failures
                           if r.get("event") == "injected_failure")

    timings = load_recovery_timings(os.path.join(job_dir, "recovery_log.jsonl"))
    stats = summarize(timings)
    metrics.recoveries = stats["recoveries"]
    metrics.recovery_latency_seconds = stats["total_seconds"]
    metrics.mean_recovery_seconds = stats["mean_seconds"]
    metrics.recovery_breakdown = stats["mean_breakdown"]

    if controller_summary:
        metrics.outcome = controller_summary.get("outcome", "unknown")
        metrics.generations = controller_summary.get("generations", 0)

    return metrics


def write_csv(rows: List[JobMetrics], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    dicts = [r.to_row() for r in rows]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(dicts[0].keys()))
        writer.writeheader()
        writer.writerows(dicts)
