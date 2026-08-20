"""
Proof that injected failures are reproducible (Phase 3 milestone).

Phase 6 compares checkpoint strategies head to head. That comparison is only
honest if every strategy faces the identical sequence of failures — a
strategy must not be able to win by getting lucky. This script is what backs
that claim up.

The experiment: run training twice, same seed, everything else equal, in
separate checkpoint dirs so neither can resume from the other. Then assert
the two runs died at the same step, killed the same worker, and agree on the
schedule fingerprint.

Deliberately compares the *observed* failure logs written by the dying
workers, not the schedules computed up front. Two identical schedules prove
the arithmetic is deterministic; two identical logs prove the arithmetic
actually reached the training loop and killed the process it said it would.

Usage:
    python failures/verify_reproducibility.py
    python failures/verify_reproducibility.py --failure-rate 0.05 --failure-seed 42
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from failures.injector import FailureSchedule

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(REPO_ROOT, "training", "train.py")

# Sized so the seed-42 / rate-0.05 schedule's first failure (step 31) lands
# inside the run, without waiting on a full-size CIFAR-10 epoch on CPU.
DEFAULTS = dict(
    failure_rate=0.05,
    failure_seed=42,
    steps=40,
    checkpoint_every=5,
    batch_size=32,
    subset_size=1024,
    world_size=2,
    epochs=3,
    seed=1337,
)

# Fields that must match across runs. `timestamp` and `pid` are deliberately
# excluded — they differ by construction, and demanding they match would be
# testing the clock rather than the injector.
COMPARED_FIELDS = ("step", "worker", "seed", "failure_rate", "workers",
                   "schedule_fingerprint", "epoch", "step_in_epoch", "loss")


class ReproducibilityError(AssertionError):
    """Raised when two same-seed runs did not fail identically."""


def _run(label: str, args: argparse.Namespace, work: str, tag: str) -> int:
    cmd = [
        sys.executable, "-u", TRAIN_SCRIPT,
        "--deterministic",
        "--epochs", str(args.epochs),
        "--world-size", str(args.world_size),
        "--batch-size", str(args.batch_size),
        "--subset-size", str(args.subset_size),
        "--seed", str(args.seed),
        "--max-steps", str(args.steps),
        "--checkpoint-every", str(args.checkpoint_every),
        # Separate dirs: if both runs shared one, the second would resume
        # from the first's checkpoints and never reach the same step numbers.
        "--checkpoint-dir", os.path.join(work, f"ckpt_{tag}"),
        "--no-resume",
        "--failure-rate", str(args.failure_rate),
        "--failure-seed", str(args.failure_seed),
        "--failure-log", os.path.join(work, f"failures_{tag}.jsonl"),
        "--log-every", "1000",
    ]
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if any(k in line for k in ("failure injection ON", "scheduled failures",
                                   "INJECTED FAILURE", "is being killed",
                                   "checkpoint @", "DONE", "max-steps")):
            print(f"  | {line}")
    print(f"  -> {label} exited {proc.returncode}")
    if proc.returncode == 0:
        print("  (no failure fired — the run completed normally)")
    return proc.returncode


def read_failures(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _key(record: dict) -> Tuple:
    return tuple(record.get(field) for field in COMPARED_FIELDS)


def verify(args: argparse.Namespace) -> dict:
    work = os.path.join(REPO_ROOT, "results", "repro")
    if os.path.exists(work):
        shutil.rmtree(work)   # stale logs would fake a match
    os.makedirs(work)

    expected = FailureSchedule(args.failure_rate, args.world_size,
                               args.failure_seed, args.steps)

    print("=" * 78)
    print("FAILURE INJECTION REPRODUCIBILITY CHECK")
    print(f"  failure_rate={args.failure_rate}  failure_seed={args.failure_seed}  "
          f"world_size={args.world_size}  steps={args.steps}")
    print(f"  schedule predicted up front: "
          f"{[(f.step, f.worker) for f in expected.failures]}")
    print(f"  schedule fingerprint: {expected.fingerprint()}")
    print("=" * 78)

    if not expected.failures:
        raise ReproducibilityError(
            f"this config schedules no failures in {args.steps} steps, so the "
            f"check would pass vacuously — raise --failure-rate or --steps")

    print("\n[1/2] RUN A")
    rc_a = _run("run A", args, work, "a")
    print("\n[2/2] RUN B (identical seed)")
    rc_b = _run("run B", args, work, "b")

    log_a = read_failures(os.path.join(work, "failures_a.jsonl"))
    log_b = read_failures(os.path.join(work, "failures_b.jsonl"))

    print(f"\nRun A logged {len(log_a)} injected failure(s); "
          f"run B logged {len(log_b)}")

    if not log_a or not log_b:
        raise ReproducibilityError(
            f"expected both runs to log a failure, got {len(log_a)} and "
            f"{len(log_b)} — the schedule never reached the training loop")

    # Each run dies at its first scheduled failure, so one record each.
    first_expected = expected.failures[0]

    print(f"\n  {'':<8} {'step':>6}  {'worker':>7}  {'fingerprint':>18}  "
          f"{'timestamp (differs)':>22}")
    for label, log in (("run A", log_a), ("run B", log_b)):
        rec = log[0]
        print(f"  {label:<8} {rec['step']:>6}  {rec['worker']:>7}  "
              f"{rec['schedule_fingerprint']:>18}  {rec['timestamp_iso']:>22}")
    print(f"  {'predicted':<8} {first_expected.step:>6}  "
          f"{first_expected.worker:>7}  {expected.fingerprint():>18}")

    keys_a = [_key(r) for r in log_a]
    keys_b = [_key(r) for r in log_b]

    if keys_a != keys_b:
        raise ReproducibilityError(
            f"the two runs failed differently:\n  A: {keys_a}\n  B: {keys_b}")

    observed = log_a[0]
    if observed["step"] != first_expected.step:
        raise ReproducibilityError(
            f"runs died at step {observed['step']} but the schedule predicted "
            f"{first_expected.step} — the injector is not honouring its own plan")
    if observed["worker"] != first_expected.worker:
        raise ReproducibilityError(
            f"killed worker {observed['worker']} but the schedule predicted "
            f"{first_expected.worker}")
    if rc_a != rc_b:
        raise ReproducibilityError(
            f"exit codes differ: run A {rc_a}, run B {rc_b}")

    summary = dict(
        failure_rate=args.failure_rate,
        failure_seed=args.failure_seed,
        failed_at_step=observed["step"],
        worker_killed=observed["worker"],
        fingerprint=observed["schedule_fingerprint"],
        failures_logged=len(log_a),
        exit_code=rc_a,
    )

    print("\n" + "=" * 78)
    print("PASS — same seed produced the same failure at the same step")
    print(f"  both runs died at step   {summary['failed_at_step']} "
          f"(predicted {first_expected.step})")
    print(f"  both runs killed worker  {summary['worker_killed']}")
    print(f"  schedule fingerprint     {summary['fingerprint']} (identical)")
    print(f"  both runs exited         {summary['exit_code']}")
    print("=" * 78)

    with open(os.path.join(work, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    return summary


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, val in DEFAULTS.items():
        p.add_argument(f"--{key.replace('_', '-')}", type=type(val), default=val)
    return p.parse_args(argv)


def main(argv=None) -> int:
    try:
        verify(parse_args(argv))
    except ReproducibilityError as exc:
        print(f"\nFAIL — {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
