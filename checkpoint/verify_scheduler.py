"""
Proof that the adaptive scheduler responds to the observed failure rate
(Phase 5 milestone).

Runs the *same command twice*, changing only `--failure-rate` (and the
failure seed, so the two scenarios don't get identical schedules by
coincidence), and reports the checkpoint interval the scheduler chose in
each. Nothing in checkpoint/scheduler.py can see the configured failure
rate — it only ever observes measured write durations, step durations, and
failure timestamps read back from the job's log. So a difference in the
chosen interval is an emergent consequence of measurement, not a branch.

The invariant that makes this honest is asserted, not assumed: the two
worker command lines are diffed, and anything other than the failure rate
and seed differing fails the check.

Expected direction, from T* = sqrt(2 * d * M):

    high failure rate  -> small M -> short interval  (protect work)
    low failure rate   -> large M -> long interval   (avoid write overhead)

Usage:
    python checkpoint/verify_scheduler.py
    python checkpoint/verify_scheduler.py --low-rate 0.0 --high-rate 0.4
"""
import argparse
import json
import os
import shutil
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.recovery import (
    RecoveryController,
    parse_args as controller_args,
    train_flags_from,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = dict(
    low_rate=0.01,
    high_rate=0.25,
    low_seed=32,
    high_seed=2,
    # 96 steps rather than 60: the calm scenario's MTBF estimate is bounded
    # by how long the run actually lasts, and at intervals of 2-4 steps
    # integer rounding compresses any ratio into invisibility. A longer
    # horizon lets the calm case earn a genuinely long interval.
    steps=96,
    initial_interval=4,
    world_size=2,
    batch_size=32,
    subset_size=1024,
    # Headroom: with subset_size/world_size/batch_size below there are 16
    # steps per epoch, so 8 epochs allows 128 steps. --max-steps has to be
    # the binding constraint — if the epoch budget runs out first the run
    # stops early and silently, and the header reports a step count the run
    # never reached.
    epochs=8,
    seed=1337,
    # Must exceed the number of failures the high-rate scenario schedules
    # (32 at rate 0.25 over 150 steps), or the job aborts on restart
    # limit and the comparison is drawn from a run that never finished.
    max_restarts=60,
)


class SchedulerCheckError(AssertionError):
    """Raised when the scheduler did not visibly react to the failure rate."""


def read_decisions(job_dir: str) -> List[dict]:
    path = os.path.join(job_dir, "scheduler_log.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_scenario(label: str, rate: float, failure_seed: int,
                 args: argparse.Namespace, work: str) -> dict:
    job_dir = os.path.join(work, label)
    controller_argv = [
        "--job-dir", job_dir,
        "--world-size", str(args.world_size),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--subset-size", str(args.subset_size),
        "--max-steps", str(args.steps),
        "--checkpoint-every", str(args.initial_interval),
        "--seed", str(args.seed),
        "--failure-rate", str(rate),
        "--failure-seed", str(failure_seed),
        "--max-restarts", str(args.max_restarts),
        "--adaptive-checkpointing",
        "--quiet",
    ]
    cargs = controller_args(controller_argv)
    flags = train_flags_from(cargs)

    print(f"\n[{label}] failure_rate={rate} failure_seed={failure_seed}")
    controller = RecoveryController(
        job_dir=job_dir, world_size=cargs.world_size, train_flags=flags,
        max_restarts=cargs.max_restarts, quiet=True)
    summary = controller.run()

    decisions = read_decisions(job_dir)
    intervals = [d["interval_steps"] for d in decisions]

    reached = max((d["global_step"] for d in decisions), default=0)
    print(f"[{label}] outcome={summary['outcome']} "
          f"recoveries={summary['recoveries']} "
          f"decisions={len(decisions)} reached_step={reached}")

    return {
        "label": label,
        "failure_rate": rate,
        "failure_seed": failure_seed,
        "job_dir": job_dir,
        "worker_flags": flags,
        "outcome": summary["outcome"],
        "recoveries": summary["recoveries"],
        "decisions": decisions,
        "intervals": intervals,
        "final_interval": intervals[-1] if intervals else None,
        "mean_interval": (sum(intervals) / len(intervals)) if intervals else None,
        "min_interval": min(intervals) if intervals else None,
        "max_interval": max(intervals) if intervals else None,
        "recovery_stats": summary.get("recovery_stats", {}),
        "reached_step": reached,
    }


def assert_only_failure_config_differs(low: dict, high: dict) -> None:
    """
    The scheduler must not be reacting to anything but measurements.

    If the two scenarios differed in, say, checkpoint-every, the interval
    difference would prove nothing. Diff the actual command lines.
    """
    def as_map(flags: List[str]) -> Dict[str, str]:
        out, i = {}, 0
        while i < len(flags):
            if flags[i].startswith("--"):
                if i + 1 < len(flags) and not flags[i + 1].startswith("--"):
                    out[flags[i]] = flags[i + 1]
                    i += 2
                else:
                    out[flags[i]] = "<set>"
                    i += 1
            else:
                i += 1
        return out

    a, b = as_map(low["worker_flags"]), as_map(high["worker_flags"])
    differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    allowed = {"--failure-rate", "--failure-seed"}

    if differing - allowed:
        raise SchedulerCheckError(
            f"scenarios differ in more than the failure config: "
            f"{sorted(differing - allowed)} — the comparison would not "
            f"isolate the scheduler's response to failures")
    print(f"\ninvariant OK: worker command lines differ only in "
          f"{sorted(differing)}")


def summarize_scenario(row: dict) -> None:
    print(f"\n  {row['label']}  (failure_rate={row['failure_rate']}, "
          f"{row['recoveries']} recovery(ies))")
    if not row["decisions"]:
        print("    no scheduler decisions recorded")
        return
    print(f"    {'step':>5}  {'d (s)':>7}  {'M (s)':>8}  {'R (s)':>7}  "
          f"{'steps/s':>8}  {'T* (s)':>7}  {'interval':>8}  {'E[waste]':>9}")
    for d in row["decisions"]:
        print(f"    {d['global_step']:>5}  {d['checkpoint_cost']:>7.3f}  "
              f"{d['mtbf']:>8.2f}  {d['recovery_cost']:>7.2f}  "
              f"{d['throughput']:>8.2f}  {d['optimal_seconds']:>7.2f}  "
              f"{d['interval_steps']:>8}  {d['expected_waste']:>9.3f}")
    print(f"    interval: min={row['min_interval']} max={row['max_interval']} "
          f"mean={row['mean_interval']:.2f} final={row['final_interval']}")


def verify(args: argparse.Namespace) -> dict:
    work = os.path.join(REPO_ROOT, "results", "scheduler")
    if os.path.exists(work):
        shutil.rmtree(work)      # stale logs would prime the scheduler
    os.makedirs(work)

    print("=" * 84)
    print("ADAPTIVE CHECKPOINT SCHEDULER — RESPONSE TO FAILURE RATE")
    print(f"  identical config both runs except failure rate/seed:")
    print(f"    steps={args.steps} world_size={args.world_size} "
          f"initial_interval={args.initial_interval} "
          f"subset={args.subset_size} batch={args.batch_size}")
    print("=" * 84)

    low = run_scenario("low_failure_rate", args.low_rate, args.low_seed,
                       args, work)
    high = run_scenario("high_failure_rate", args.high_rate, args.high_seed,
                        args, work)

    assert_only_failure_config_differs(low, high)

    summarize_scenario(low)
    summarize_scenario(high)

    if not low["decisions"] or not high["decisions"]:
        raise SchedulerCheckError(
            "one scenario recorded no scheduler decisions — nothing to compare")

    # A scenario that died early, or ran out of epochs before reaching
    # --max-steps, gives a truncated view of the interval it would have
    # settled on. Caught explicitly because it fails silently otherwise:
    # the job still exits 0 and still reports "completed".
    for row in (low, high):
        if row["outcome"] != "completed":
            raise SchedulerCheckError(
                f"{row['label']} did not complete (outcome="
                f"{row['outcome']}) — its intervals are not representative")
        # The last decision lands within one interval of the end, not on it.
        if row["reached_step"] < args.steps - max(row["intervals"]) - 1:
            raise SchedulerCheckError(
                f"{row['label']} only reached step {row['reached_step']} of "
                f"{args.steps} — check that --epochs allows that many steps "
                f"(epoch budget binds before --max-steps otherwise)")

    # The headline comparison. Mean over the run rather than the final value:
    # a single final reading could be an artefact of where the run happened
    # to stop.
    low_mean, high_mean = low["mean_interval"], high["mean_interval"]

    print("\n" + "=" * 84)
    if high_mean >= low_mean:
        raise SchedulerCheckError(
            f"scheduler did NOT shorten the interval under the higher failure "
            f"rate: mean interval was {high_mean:.2f} steps at rate "
            f"{args.high_rate} vs {low_mean:.2f} steps at rate {args.low_rate}")

    ratio = low_mean / high_mean if high_mean else float("inf")

    def mean_of(row, key):
        return sum(d[key] for d in row["decisions"]) / len(row["decisions"])

    low_t, high_t = mean_of(low, "optimal_seconds"), mean_of(high, "optimal_seconds")
    low_m, high_m = mean_of(low, "mtbf"), mean_of(high, "mtbf")

    print("PASS — the scheduler chose a visibly shorter interval under the "
          "higher failure rate")
    print(f"  low  rate {args.low_rate:<5} -> mean interval "
          f"{low_mean:>6.2f} steps  (observed M up to "
          f"{max(d['mtbf'] for d in low['decisions']):.1f}s, "
          f"{low['recoveries']} recoveries)")
    print(f"  high rate {args.high_rate:<5} -> mean interval "
          f"{high_mean:>6.2f} steps  (observed M down to "
          f"{min(d['mtbf'] for d in high['decisions']):.1f}s, "
          f"{high['recoveries']} recoveries)")
    print(f"  ratio      {ratio:.2f}x shorter under load")
    print()
    print(f"  {'':<22}{'low rate':>12}{'high rate':>12}{'ratio':>9}")
    print(f"  {'mean observed M':<22}{low_m:>11.1f}s{high_m:>11.1f}s"
          f"{low_m / high_m:>8.2f}x")
    print(f"  {'mean T* (continuous)':<22}{low_t:>11.2f}s{high_t:>11.2f}s"
          f"{low_t / high_t:>8.2f}x")
    print(f"  {'mean interval (steps)':<22}{low_mean:>12.2f}{high_mean:>12.2f}"
          f"{ratio:>8.2f}x")
    print(f"\n  T* scales as sqrt(M), so a {low_m / high_m:.1f}x difference in "
          f"MTBF is expected to give ~{(low_m / high_m) ** 0.5:.2f}x in "
          f"interval; observed {low_t / high_t:.2f}x.")
    print(f"  neither scenario hardcodes an interval; both got "
          f"--checkpoint-every {args.initial_interval} as a starting point")
    print("=" * 84)

    summary = {
        "low": {k: v for k, v in low.items() if k != "decisions"},
        "high": {k: v for k, v in high.items() if k != "decisions"},
        "ratio": ratio,
    }
    with open(os.path.join(work, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, val in DEFAULTS.items():
        p.add_argument(f"--{key.replace('_', '-')}", type=type(val), default=val)
    return p.parse_args(argv)


def main(argv=None) -> int:
    try:
        verify(parse_args(argv))
    except SchedulerCheckError as exc:
        print(f"\nFAIL — {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
