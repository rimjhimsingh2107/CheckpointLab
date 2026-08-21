"""
Adaptive checkpointing — the treatment group.

Runs the same scenarios as `fixed_interval.py` with the Phase 5 scheduler
deciding the interval from measured checkpoint cost, observed MTBF and
throughput (T* = sqrt(2dM)) instead of holding it fixed.

`--checkpoint-every` is still passed, but here it is only the *starting*
interval used until enough has been measured to compute one. It is set to the
same value as the medium fixed strategy on purpose: if the adaptive run began
from a different starting point than any baseline, part of any difference in
the result would be attributable to the starting point rather than to the
adaptation.

Usage:
    python experiments/adaptive.py --failure-rate 0.1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.fixed_interval import FIXED_INTERVALS
from experiments.harness import RESULTS_ROOT, Scenario, run_job
from metrics.job_metrics import JobMetrics

# Deliberately the medium fixed interval — see module docstring.
INITIAL_INTERVAL = FIXED_INTERVALS["fixed_medium"]

STRATEGY = "adaptive"


def run_adaptive(scenario: Scenario, initial_interval: int = INITIAL_INTERVAL,
                 results_root: str = RESULTS_ROOT) -> JobMetrics:
    """Run one adaptive job."""
    return run_job(scenario=scenario, strategy=STRATEGY,
                   checkpoint_every=initial_interval, adaptive=True,
                   results_root=results_root)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--failure-rate", type=float, default=0.1)
    p.add_argument("--failure-seed", type=int, default=2)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--initial-interval", type=int, default=INITIAL_INTERVAL)
    p.add_argument("--name", default="adhoc")
    args = p.parse_args(argv)

    scenario = Scenario(name=args.name, failure_rate=args.failure_rate,
                        failure_seed=args.failure_seed, steps=args.steps)
    print(f"scenario {scenario.name}: failures scheduled at "
          f"{scenario.expected_failures()}")

    m = run_adaptive(scenario, args.initial_interval)
    print()
    print(f"  {m.strategy:<16} wall {m.wall_clock_seconds:6.1f}s  "
          f"waste {m.wasted_pct:5.1f}%  "
          f"ckpt {m.checkpoint_overhead_pct:5.1f}%  "
          f"mean interval {m.mean_interval_steps:.2f} steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
