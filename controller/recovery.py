"""
Automatic failure detection and recovery.

Phases 2 and 3 built the two halves separately: training can resume from a
checkpoint bit-exactly, and workers can be killed on a reproducible
schedule. Both still needed a human to notice the job had died and relaunch
it. This closes the loop — the controller runs the job unattended, and a
worker dying is an event it handles rather than an outcome it reports.

The cycle:

    launch generation -> poll -> failure detected
        -> terminate the whole generation
        -> find the latest checkpoint that actually loads
        -> relaunch and resume
        -> back to polling

Three things here are less obvious than they look:

**The survivors have to be killed, not waited for.** When one rank dies the
others don't crash — they block inside a collective waiting for a peer
that's never coming back, and gloo's default timeout is 30 minutes. So
recovery starts by tearing down ranks that are, strictly speaking, still
healthy. The checkpoint is the only state all ranks can agree on, so
partial worlds aren't recoverable; the whole generation goes.

**"Latest checkpoint" is not good enough — it has to be the latest *valid*
one.** A worker killed mid-write is exactly what Phase 3 injects. Atomic
writes in checkpoint/manager.py mean that shouldn't produce a torn file, but
recovery is the wrong place to assume the layer below is perfect, so each
candidate is actually loaded before being trusted and the search walks
backwards until one passes.

**Recovery time is measured from detection, not from the failure.** Those
differ by up to one poll interval, and conflating them would let the
controller look faster by simply polling less often.

Usage:
    python controller/recovery.py --failure-rate 0.05 --failure-seed 42
    python controller/recovery.py --failure-rate 0.2 --max-restarts 10
"""
import argparse
import json
import os
import sys
import time
import uuid
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from checkpoint.manager import FORMAT_VERSION, list_checkpoints
from controller.worker_manager import WorkerFailure, WorkerManager

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_JOBS_ROOT = os.path.join("results", "jobs")

REQUIRED_CHECKPOINT_KEYS = {"model", "optimizer", "epoch", "step_in_epoch",
                            "global_step"}


def new_job_dir(jobs_root: str = DEFAULT_JOBS_ROOT) -> str:
    """
    A fresh directory for one job's entire state.

    Every job gets its own, always. Phase 3's failure log defaulted to one
    shared path with a run id derived from the failure rate and seed, which
    meant two jobs sharing those values would read each other's already-
    fired failures and suppress them. Job identity being a unique directory
    makes that collision unrepresentable instead of merely discouraged.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(jobs_root, f"{stamp}-{uuid.uuid4().hex[:6]}")


def validate_checkpoint(path: str) -> Optional[dict]:
    """
    Load `path` and return its metadata, or None if it can't be trusted.

    Deliberately loads the file rather than just stat-ing it. A truncated
    checkpoint has a plausible size and a plausible name; the only way to
    know it deserializes is to deserialize it. On a 90 MB ResNet-18
    checkpoint that costs a few hundred ms, which is cheap next to
    discovering the problem after relaunching. A production system would
    keep a checksum alongside and verify that instead.
    """
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("format_version") != FORMAT_VERSION:
        return None
    if not REQUIRED_CHECKPOINT_KEYS <= set(payload):
        return None

    return {
        "path": path,
        "global_step": payload["global_step"],
        "epoch": payload["epoch"],
        "step_in_epoch": payload["step_in_epoch"],
    }


def find_latest_valid_checkpoint(checkpoint_dir: str
                                ) -> Tuple[Optional[dict], List[str]]:
    """
    Newest checkpoint that actually loads, plus any rejects found on the way.

    Returns (metadata_or_None, rejected_paths).
    """
    rejected: List[str] = []
    for path in reversed(list_checkpoints(checkpoint_dir)):
        meta = validate_checkpoint(path)
        if meta is not None:
            return meta, rejected
        rejected.append(path)
    return None, rejected


class EventLog:
    """Append-only JSONL record of everything the controller did and when."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._path = path

    def emit(self, event: str, **fields) -> dict:
        record = {"event": event, "timestamp": time.time(),
                  "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
        with open(self._path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return record


class RecoveryController:
    """Runs a training job to completion, restarting it through failures."""

    def __init__(self, job_dir: str, world_size: int, train_flags: List[str],
                 max_restarts: int = 5, poll_interval: float = 0.2,
                 heartbeat_timeout: float = 90.0, quiet: bool = False):
        self.job_dir = job_dir
        self.world_size = world_size
        self.max_restarts = max_restarts
        self.poll_interval = poll_interval
        self.quiet = quiet

        os.makedirs(job_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(job_dir, "checkpoints")
        self.events = EventLog(os.path.join(job_dir, "recovery_log.jsonl"))

        self.manager = WorkerManager(
            job_dir=job_dir, world_size=world_size,
            train_flags=["--job-dir", job_dir] + list(train_flags),
            heartbeat_timeout=heartbeat_timeout)

        self.recoveries: List[dict] = []

    def _say(self, message: str) -> None:
        if not self.quiet:
            print(message, flush=True)

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def _wait_for_training_to_resume(self, restore_step: int,
                                     timeout: float = 300.0) -> Optional[float]:
        """
        Block until the new generation has actually computed a step.

        "Resumed" deliberately means a completed training step, not a
        started process. A worker that comes up, loads a checkpoint and then
        dies has not resumed anything, and counting the relaunch as success
        would report a recovery time for a job that isn't running.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            for beat in self.manager.heartbeats().values():
                step = beat.get("global_step")
                if step is None:
                    continue
                if beat.get("state") == "training":
                    return time.time()
                # The relaunched generation can finish the remaining work
                # between two polls, overwriting its last "training"
                # heartbeat with "finished". That's a successful resume, not
                # a hang — without this branch a job with only a step or two
                # left would sit here until the timeout and be reported as
                # resume_failed.
                if beat.get("state") == "finished" and step > restore_step:
                    return time.time()

            if not self.manager.any_alive():
                # Same race, seen from the process table instead of the
                # heartbeat: everyone exited cleanly, so the job resumed and
                # ran to completion.
                return time.time() if self.manager.all_exited_cleanly() else None

            failure = self.manager.poll()
            if failure is not None:
                return None  # died again before computing a step
            time.sleep(self.poll_interval)
        return None

    def _recover(self, failure: WorkerFailure, attempt: int) -> Optional[dict]:
        """Handle one failure. Returns timing detail, or None if unrecoverable."""
        detected_at = failure.detected_at

        self._say(f"\n  !! FAILURE DETECTED  {failure.describe()}")
        self.events.emit("failure_detected", rank=failure.rank,
                         kind=failure.kind, exit_code=failure.exit_code,
                         last_step=failure.last_step, attempt=attempt,
                         generation=self.manager.generation)

        for line in self.manager.tail_log(failure.rank, lines=3):
            self._say(f"     worker {failure.rank} log | {line}")

        # 1. Tear down the whole generation, survivors included.
        alive = [r for r, code in self.manager.exit_codes().items()
                 if code is None]
        self._say(f"     terminating generation "
                  f"{self.manager.generation} (survivors still up: {alive})")
        codes = self.manager.terminate_all()
        killed_at = time.time()
        self.events.emit("generation_terminated", exit_codes=codes,
                         survivors_killed=alive)

        # 2. Find something trustworthy to resume from.
        meta, rejected = find_latest_valid_checkpoint(self.checkpoint_dir)
        checkpoint_at = time.time()
        for bad in rejected:
            self._say(f"     REJECTED unreadable checkpoint {os.path.basename(bad)}")
            self.events.emit("checkpoint_rejected", path=bad)

        if meta is None:
            self._say("     no valid checkpoint yet — restarting from scratch")
            self.events.emit("no_checkpoint_available")
            restore_step = 0
        else:
            restore_step = meta["global_step"]
            self._say(f"     resuming from {os.path.basename(meta['path'])} "
                      f"(global_step={restore_step}, epoch={meta['epoch']}, "
                      f"step_in_epoch={meta['step_in_epoch']})")
            self.events.emit("checkpoint_selected", **meta,
                             rejected=len(rejected))

        # 3. Relaunch and wait for evidence of real progress.
        self.manager.launch(resume=True)
        relaunched_at = time.time()
        self.events.emit("generation_launched",
                         generation=self.manager.generation, resume=True,
                         restore_step=restore_step)
        self._say(f"     relaunched as generation {self.manager.generation}")

        resumed_at = self._wait_for_training_to_resume(restore_step)
        if resumed_at is None:
            self.events.emit("resume_failed", attempt=attempt)
            return None

        detail = {
            "attempt": attempt,
            "failed_rank": failure.rank,
            "failure_kind": failure.kind,
            "exit_code": failure.exit_code,
            "step_at_failure": failure.last_step,
            "restored_to_step": restore_step,
            "steps_lost": (max(0, (failure.last_step or 0) - restore_step)),
            "recovery_seconds": resumed_at - detected_at,
            "breakdown": {
                "terminate_workers": killed_at - detected_at,
                "select_checkpoint": checkpoint_at - killed_at,
                "relaunch_workers": relaunched_at - checkpoint_at,
                "reach_first_step": resumed_at - relaunched_at,
            },
        }
        self.events.emit("training_resumed", **detail)
        self._say(f"     RECOVERED in {detail['recovery_seconds']:.2f}s "
                  f"(lost {detail['steps_lost']} step(s) of work)")
        return detail

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        started = time.time()
        self.events.emit("job_started", job_dir=self.job_dir,
                         world_size=self.world_size,
                         max_restarts=self.max_restarts)

        self._say(f"controller: job_dir={self.job_dir}")
        self._say(f"controller: launching {self.world_size} workers "
                  f"(generation 1)")
        self.manager.launch(resume=True)
        self.events.emit("generation_launched", generation=1, resume=True)

        attempt = 0
        outcome = "completed"

        while True:
            failure = self.manager.poll()

            if failure is None:
                if not self.manager.any_alive():
                    if self.manager.all_exited_cleanly():
                        break
                    # Every worker is gone but somebody exited non-zero and
                    # poll() didn't catch it — treat as a failure rather than
                    # declaring victory on a job that didn't finish.
                    codes = self.manager.exit_codes()
                    bad = next((r for r, c in codes.items() if c not in (0, None)), 0)
                    failure = WorkerFailure(rank=bad, kind="exit",
                                            detected_at=time.time(),
                                            exit_code=codes.get(bad),
                                            detail="detected at drain")
                else:
                    time.sleep(self.poll_interval)
                    continue

            attempt += 1
            if attempt > self.max_restarts:
                self._say(f"\n  !! giving up after {self.max_restarts} restarts")
                self.events.emit("job_failed", reason="max_restarts_exceeded",
                                 attempts=attempt - 1)
                self.manager.terminate_all()
                outcome = "max_restarts_exceeded"
                break

            detail = self._recover(failure, attempt)
            if detail is None:
                self._say("\n  !! relaunched workers failed to resume")
                self.events.emit("job_failed", reason="resume_failed",
                                 attempts=attempt)
                self.manager.terminate_all()
                outcome = "resume_failed"
                break
            self.recoveries.append(detail)

        elapsed = time.time() - started
        summary = {
            "outcome": outcome,
            "job_dir": self.job_dir,
            "world_size": self.world_size,
            "generations": self.manager.generation,
            "recoveries": len(self.recoveries),
            "wall_clock_seconds": elapsed,
            "exit_codes": self.manager.exit_codes(),
            "recovery_details": self.recoveries,
        }
        if self.recoveries:
            times = [r["recovery_seconds"] for r in self.recoveries]
            summary["recovery_seconds_mean"] = sum(times) / len(times)
            summary["recovery_seconds_min"] = min(times)
            summary["recovery_seconds_max"] = max(times)
            summary["steps_lost_total"] = sum(r["steps_lost"]
                                              for r in self.recoveries)

        self.events.emit("job_finished", **{k: v for k, v in summary.items()
                                            if k != "recovery_details"})
        with open(os.path.join(self.job_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--job-dir", default=None,
                   help="Defaults to a fresh results/jobs/<stamp>-<id> dir")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--max-restarts", type=int, default=5)
    p.add_argument("--poll-interval", type=float, default=0.2)
    p.add_argument("--heartbeat-timeout", type=float, default=90.0,
                   help="Treat a worker with no heartbeat for this long as "
                        "dead even if its process is alive — this is what "
                        "catches ranks hung in a collective")

    # Passed through to the workers.
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--subset-size", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--failure-rate", type=float, default=0.05)
    p.add_argument("--failure-seed", type=int, default=42)
    p.add_argument("--failure-min-step", type=int, default=1)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--quiet", action="store_true")

    return p.parse_args(argv)


def train_flags_from(args: argparse.Namespace) -> List[str]:
    flags = [
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--subset-size", str(args.subset_size),
        "--max-steps", str(args.max_steps),
        "--checkpoint-every", str(args.checkpoint_every),
        "--seed", str(args.seed),
        "--failure-rate", str(args.failure_rate),
        "--failure-seed", str(args.failure_seed),
        "--failure-min-step", str(args.failure_min_step),
        "--log-every", str(args.log_every),
    ]
    if args.deterministic:
        flags.append("--deterministic")
    return flags


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 78)
    ok = summary["outcome"] == "completed"
    print(f"{'PASS' if ok else 'FAIL'} — job {summary['outcome']} "
          f"after {summary['recoveries']} automatic recovery(ies)")
    print(f"  job dir            {summary['job_dir']}")
    print(f"  generations run    {summary['generations']}")
    print(f"  wall clock         {summary['wall_clock_seconds']:.1f}s")
    print(f"  final exit codes   {summary['exit_codes']}")

    if summary["recoveries"]:
        print(f"\n  {'#':>2}  {'rank':>4}  {'kind':>16}  {'died@':>6}  "
              f"{'restored@':>9}  {'lost':>4}  {'recovery s':>10}")
        for detail in summary["recovery_details"]:
            print(f"  {detail['attempt']:>2}  {detail['failed_rank']:>4}  "
                  f"{detail['failure_kind']:>16}  "
                  f"{str(detail['step_at_failure']):>6}  "
                  f"{detail['restored_to_step']:>9}  "
                  f"{detail['steps_lost']:>4}  "
                  f"{detail['recovery_seconds']:>10.2f}")

        print(f"\n  recovery time: mean {summary['recovery_seconds_mean']:.2f}s  "
              f"min {summary['recovery_seconds_min']:.2f}s  "
              f"max {summary['recovery_seconds_max']:.2f}s")
        print(f"  total work lost to failures: "
              f"{summary['steps_lost_total']} step(s)")

        print("\n  where recovery time goes (first recovery):")
        for phase, seconds in summary["recovery_details"][0]["breakdown"].items():
            print(f"    {phase:<20} {seconds:>7.2f}s")
    print("=" * 78)


def main(argv=None) -> int:
    args = parse_args(argv)
    job_dir = args.job_dir or new_job_dir()

    controller = RecoveryController(
        job_dir=job_dir, world_size=args.world_size,
        train_flags=train_flags_from(args),
        max_restarts=args.max_restarts, poll_interval=args.poll_interval,
        heartbeat_timeout=args.heartbeat_timeout, quiet=args.quiet)

    summary = controller.run()
    print_summary(summary)
    return 0 if summary["outcome"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
