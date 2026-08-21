"""
Fixed-interval checkpointing — the control group.

This is Phase 2's behaviour: checkpoint every N steps, N chosen up front and
never revisited. It is the thing the adaptive scheduler has to beat, and it
needs to be represented honestly — by a *range* of N, not one strawman.

Three intervals are run, because "fixed interval" isn't a single strategy but
a family whose members fail in opposite directions:

  * **short** — cheap crashes, expensive steady state. Pays the write cost
    constantly whether or not anything is failing.
  * **medium** — a reasonable hand-picked guess, the honest baseline.
  * **long** — cheap steady state, expensive crashes. Loses more work per
    failure, and pays more to fast-forward the dataloader on resume.

The comparison to make is against the *best* of these per scenario, not the
average. Beating a badly-chosen fixed interval proves nothing; the claim
worth making is beating the best one you could have picked with hindsight —
which the adaptive scheduler has to do without the hindsight.

Usage:
    python experiments/fixed_interval.py --interval 8 --failure-rate 0.1
"""
import argparse
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.harness import RESULTS_ROOT, Scenario, run_job
from metrics.job_metrics import JobMetrics

# Step counts, not wall-clock minutes. The project plan suggests 1/5/15
# minutes, which assumes a cluster; on this CPU testbed a step takes about a
# second and a whole run is ~60 steps, so minute-scale intervals would mean
# "checkpoint once, at the end" for all three. These are the same idea at the
# testbed's scale, and the benchmark reports each one's wall-clock equivalent
# at measured throughput so the mapping stays visible.
FIXED_INTERVALS: Dict[str, int] = {
    "fixed_short": 2,
    "fixed_medium": 8,
    "fixed_long": 24,
}


def run_fixed(scenario: Scenario, interval: int, label: str = None,
              results_root: str = RESULTS_ROOT) -> JobMetrics:
    """Run one fixed-interval job."""
    return run_job(scenario=scenario,
                   strategy=label or f"fixed_{interval}",
                   checkpoint_every=interval, adaptive=False,
                   results_root=results_root)


def run_all_fixed(scenario: Scenario,
                  results_root: str = RESULTS_ROOT) -> List[JobMetrics]:
    """Run every fixed interval against one scenario."""
    return [run_fixed(scenario, interval, label, results_root)
            for label, interval in FIXED_INTERVALS.items()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interval", type=int, default=None,
                   help="Checkpoint every N steps (default: run all three)")
    p.add_argument("--failure-rate", type=float, default=0.1)
    p.add_argument("--failure-seed", type=int, default=2)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--name", default="adhoc")
    args = p.parse_args(argv)

    scenario = Scenario(name=args.name, failure_rate=args.failure_rate,
                        failure_seed=args.failure_seed, steps=args.steps)
    print(f"scenario {scenario.name}: failures scheduled at "
          f"{scenario.expected_failures()}")

    results = ([run_fixed(scenario, args.interval)] if args.interval
               else run_all_fixed(scenario))

    print()
    for m in results:
        print(f"  {m.strategy:<16} wall {m.wall_clock_seconds:6.1f}s  "
              f"waste {m.wasted_pct:5.1f}%  "
              f"ckpt {m.checkpoint_overhead_pct:5.1f}%  "
              f"redundant steps {m.steps_redundant}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
