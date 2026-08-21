"""
Shared machinery for the Phase 6 experiments.

`fixed_interval.py` and `adaptive.py` are the two strategies under test and
`benchmark.py` orchestrates them; all three need the same notion of a
scenario and the same job runner, so it lives here rather than being
duplicated or imported sideways between the strategy modules.

WHAT MAKES THE COMPARISON FAIR
------------------------------

Every strategy in a scenario must face the *identical* failure sequence,
otherwise a strategy can win by having gotten lucky. Phase 3 gives that for
free: failures are a pure function of (seed, step), so two runs over the same
step range meet the same failures at the same steps regardless of how often
they checkpoint. `benchmark.py` asserts it after the fact rather than
trusting it.

The only flags that differ between strategies are `--checkpoint-every` and
`--adaptive-checkpointing`. Same seed, same data, same world size, same step
budget.

WHY EVERY SCENARIO NEEDS SEVERAL FAILURES
-----------------------------------------

The controller writes its `training_resumed` event only after it observes the
new generation's first training step — which is necessarily after that
generation's worker already read the log at startup. So a worker's estimate
of recovery cost R lags by one recovery, and a job with exactly one recovery
never sees a non-zero R at all. Benchmarking a scenario with a single failure
would therefore understate R's contribution for every strategy. Scenarios are
chosen to fire several failures, and `benchmark.py` refuses to report on one
that didn't.
"""
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.recovery import RecoveryController, train_flags_from
from controller.recovery import parse_args as controller_parse_args
from failures.injector import FailureSchedule
from metrics.job_metrics import JobMetrics, collect_job_metrics

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(REPO_ROOT, "results", "benchmark")


@dataclass(frozen=True)
class Scenario:
    """A failure environment every strategy is measured against."""
    name: str
    failure_rate: float
    failure_seed: int

    # Held identical across strategies and scenarios.
    steps: int = 60
    world_size: int = 2
    batch_size: int = 32
    subset_size: int = 1024
    epochs: int = 8
    seed: int = 1337
    max_restarts: int = 60

    def expected_failures(self) -> List[tuple]:
        """
        The failures this scenario will inject, known before running.

        Lets the benchmark verify up front that a scenario actually exercises
        recovery, instead of discovering afterwards that it fired none.
        """
        schedule = FailureSchedule(self.failure_rate, self.world_size,
                                   self.failure_seed, self.steps)
        return [(f.step, f.worker) for f in schedule.failures]


def run_job(scenario: Scenario, strategy: str, checkpoint_every: int,
            adaptive: bool, results_root: str = RESULTS_ROOT,
            quiet: bool = True) -> JobMetrics:
    """
    Run one (strategy, scenario) job under the Phase 4 controller and cost it.

    Wall clock is measured around the controller, so it includes everything a
    user would wait for: training, checkpoint writes, crashes, detection,
    restarts and checkpoint reloads.
    """
    job_dir = os.path.join(results_root, scenario.name, strategy)
    os.makedirs(job_dir, exist_ok=True)

    argv = [
        "--job-dir", job_dir,
        "--world-size", str(scenario.world_size),
        "--epochs", str(scenario.epochs),
        "--batch-size", str(scenario.batch_size),
        "--subset-size", str(scenario.subset_size),
        "--max-steps", str(scenario.steps),
        "--seed", str(scenario.seed),
        "--checkpoint-every", str(checkpoint_every),
        "--failure-rate", str(scenario.failure_rate),
        "--failure-seed", str(scenario.failure_seed),
        "--max-restarts", str(scenario.max_restarts),
        "--quiet",
    ]
    if adaptive:
        argv.append("--adaptive-checkpointing")

    args = controller_parse_args(argv)
    controller = RecoveryController(
        job_dir=job_dir, world_size=args.world_size,
        train_flags=train_flags_from(args),
        max_restarts=args.max_restarts, quiet=quiet)

    print(f"  running {strategy:<16} on {scenario.name:<18} "
          f"(checkpoint_every={checkpoint_every}"
          f"{', adaptive' if adaptive else ''}) ... ", end="", flush=True)

    started = time.perf_counter()
    summary = controller.run()
    wall_clock = time.perf_counter() - started

    metrics = collect_job_metrics(
        job_dir=job_dir, strategy=strategy, scenario=scenario.name,
        failure_rate=scenario.failure_rate, wall_clock_seconds=wall_clock,
        controller_summary=summary)

    print(f"{wall_clock:6.1f}s  "
          f"{metrics.recoveries} recoveries, "
          f"{metrics.checkpoints_written} checkpoints, "
          f"waste {metrics.wasted_pct:.1f}%")

    with open(os.path.join(job_dir, "metrics.json"), "w") as fh:
        json.dump(metrics.to_row(), fh, indent=2)

    return metrics
