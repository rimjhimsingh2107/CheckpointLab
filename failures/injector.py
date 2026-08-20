"""
Configurable, reproducible worker-failure injection (Phase 3).

Chaos that can't be replayed is useless for benchmarking. Phase 6 compares
checkpoint strategies against each other, and that comparison is only fair
if every strategy faces the *identical* sequence of failures — otherwise a
strategy can win by having gotten lucky. So the requirement here isn't
"randomly kill workers", it's "kill workers on a schedule that is a pure
function of the seed".

Two design choices make that hold:

1.  **Failures are derived from a hash, not a sequential RNG.** The decision
    for step N is `sha256(seed:N:salt)`, so it depends on nothing but the
    seed and the step index. A sequential `random.Random(seed)` would give
    the same guarantee only if every step is consulted exactly once, in
    order — which quietly breaks the moment a run resumes from a checkpoint
    and replays steps, or a caller peeks ahead. Hashing is order-independent
    and O(1) per query, so extending a run's step budget can't retroactively
    change the failures it already saw.

2.  **The injector never touches the global RNG.** It owns no `random` or
    `torch` seed state. If it drew from the training RNG, merely *enabling*
    failure injection would shift the augmentation stream and change the
    loss trajectory — so a run with failures could never be compared
    against one without, and Phase 2's correctness guarantee would be
    unobservable through it.

Every injected failure is appended to a JSONL log with a timestamp and the
worker that died, so a run's actual failure history is auditable after the
fact rather than inferred from the schedule.

CLI (prints the schedule without running anything):
    python failures/injector.py --failure-rate 0.05 --workers 4 --seed 42
    python failures/injector.py --failure-rate 0.05 --workers 4 --seed 42 --max-steps 500 --json
"""
import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

DEFAULT_FAILURE_LOG = os.path.join("results", "failure_log.jsonl")

# Exit code used by a worker the injector kills. 137 is what a SIGKILL'd
# process reports (128 + 9), which is what preemption and OOM-kills look
# like in real cluster logs.
INJECTED_EXIT_CODE = 137


def _uniform(seed: int, step: int, salt: str) -> float:
    """
    A deterministic pseudo-random float in [0, 1) keyed on (seed, step, salt).

    SHA-256 rather than `random.Random` deliberately: this is stable across
    Python versions, platforms and call order, which are exactly the axes a
    reproducibility claim has to survive.
    """
    digest = hashlib.sha256(f"{seed}:{step}:{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / (1 << 64)


@dataclass(frozen=True, order=True)
class ScheduledFailure:
    """A single decision: at `step`, worker `worker` dies."""
    step: int
    worker: int

    def as_dict(self) -> Dict[str, int]:
        return {"step": self.step, "worker": self.worker}


class FailureSchedule:
    """
    The set of failures a run will experience, decided up front from the seed.

    `failure_rate` is the per-step probability that *some* worker dies, and
    the victim is drawn uniformly across workers. Both come from the same
    keyed hash, so the whole schedule is reproducible.
    """

    def __init__(self, failure_rate: float, workers: int, seed: int,
                 max_steps: int, min_step: int = 1):
        if not 0.0 <= failure_rate <= 1.0:
            raise ValueError(f"failure_rate must be in [0, 1], got {failure_rate}")
        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        if max_steps < 0:
            raise ValueError(f"max_steps must be >= 0, got {max_steps}")

        self.failure_rate = failure_rate
        self.workers = workers
        self.seed = seed
        self.max_steps = max_steps
        # Failing before there's ever been a checkpoint just means the run
        # restarts from zero, which tests nothing. min_step buys a floor of
        # real progress before the chaos starts.
        self.min_step = max(1, min_step)

        self._by_step: Dict[int, int] = {}
        if failure_rate > 0.0:
            for step in range(self.min_step, max_steps + 1):
                if _uniform(seed, step, "fire") < failure_rate:
                    victim = int(_uniform(seed, step, "victim") * workers)
                    # Clamp: _uniform is [0,1) so this can only bite on a
                    # pathological float, but an out-of-range rank would be
                    # a confusing crash rather than a skipped failure.
                    self._by_step[step] = min(victim, workers - 1)

    @property
    def failures(self) -> List[ScheduledFailure]:
        return [ScheduledFailure(step, worker)
                for step, worker in sorted(self._by_step.items())]

    def victim_at(self, step: int) -> Optional[int]:
        """Which worker dies at `step`, or None if this step is survivable."""
        return self._by_step.get(step)

    def fingerprint(self) -> str:
        """
        Short hash of the whole schedule.

        Lets two runs prove they faced the same failures by comparing one
        string, instead of eyeballing two lists.
        """
        payload = json.dumps([f.as_dict() for f in self.failures],
                             sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def describe(self) -> str:
        return (f"failure_rate={self.failure_rate} workers={self.workers} "
                f"seed={self.seed} steps={self.min_step}..{self.max_steps} "
                f"-> {len(self._by_step)} failures, "
                f"fingerprint={self.fingerprint()}")


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def load_fired_steps(log_path: str, run_id: Optional[str] = None) -> Set[int]:
    """
    Steps that have already killed a worker, according to `log_path`.

    Needed because recovery replays work: a run that dies at step 13 resumes
    from its last checkpoint and re-executes step 13. Without this, the
    schedule would fire there again and the job would never get past it.
    Reading it back from the log rather than tracking it in memory is what
    makes it survive the process death.
    """
    fired: Set[int] = set()
    if not os.path.exists(log_path):
        return fired

    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A worker killed mid-write can leave a partial final line.
                # Skipping it is right: an unparseable record means that
                # failure may not have been fully committed, and re-firing
                # is safer than silently treating it as done.
                continue
            if record.get("event") != "injected_failure":
                continue
            if run_id is None or record.get("run_id") == run_id:
                fired.add(record["step"])
    return fired


def log_failure(log_path: str, *, step: int, worker: int, schedule: FailureSchedule,
                run_id: str, extra: Optional[dict] = None) -> dict:
    """
    Append one failure record, durably, before the caller kills itself.

    Written with a single O_APPEND write and fsync'd: the process calling
    this is about to os._exit, so nothing else will flush it, and a lost
    record would make an injected failure indistinguishable from a real bug.
    """
    record = {
        "event": "injected_failure",
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S",
                                       time.localtime()),
        "step": step,
        "worker": worker,
        "pid": os.getpid(),
        "run_id": run_id,
        "seed": schedule.seed,
        "failure_rate": schedule.failure_rate,
        "workers": schedule.workers,
        "schedule_fingerprint": schedule.fingerprint(),
    }
    if extra:
        record.update(extra)

    parent = os.path.dirname(os.path.abspath(log_path))
    os.makedirs(parent, exist_ok=True)

    line = (json.dumps(record) + "\n").encode()
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)

    return record


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Decide (and print) a reproducible worker-failure schedule.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Prints the schedule only; it does not run or kill anything.\n"
               "Wire it into training with:\n"
               "  python training/train.py --failure-rate 0.05 --failure-seed 42",
    )
    p.add_argument("--failure-rate", type=float, required=True,
                   help="Per-step probability that some worker dies")
    p.add_argument("--workers", type=int, required=True,
                   help="Number of workers the failures are spread across")
    p.add_argument("--seed", type=int, required=True,
                   help="Same seed => same schedule, always")
    p.add_argument("--max-steps", type=int, default=200,
                   help="How many steps of schedule to compute (default 200)")
    p.add_argument("--min-step", type=int, default=1,
                   help="Don't inject failures before this step")
    p.add_argument("--json", action="store_true",
                   help="Emit the schedule as JSON instead of a table")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    schedule = FailureSchedule(args.failure_rate, args.workers, args.seed,
                               args.max_steps, min_step=args.min_step)

    if args.json:
        print(json.dumps({
            "failure_rate": schedule.failure_rate,
            "workers": schedule.workers,
            "seed": schedule.seed,
            "min_step": schedule.min_step,
            "max_steps": schedule.max_steps,
            "fingerprint": schedule.fingerprint(),
            "failures": [f.as_dict() for f in schedule.failures],
        }, indent=2))
        return 0

    print(schedule.describe())
    observed = len(schedule.failures) / max(1, args.max_steps - schedule.min_step + 1)
    print(f"observed rate over the window: {observed:.4f} "
          f"(requested {args.failure_rate})")

    if not schedule.failures:
        print("\nNo failures in this window — raise --failure-rate or --max-steps.")
        return 0

    print(f"\n  {'step':>6}  {'worker killed':>13}")
    for failure in schedule.failures:
        print(f"  {failure.step:>6}  {failure.worker:>13}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
