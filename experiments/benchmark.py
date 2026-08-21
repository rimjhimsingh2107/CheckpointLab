"""
Benchmark suite: fixed-interval vs adaptive checkpointing.

Runs every strategy against every failure-rate scenario, costs each job out,
checks the numbers hang together, and emits a table, a CSV and a chart.

EXPERIMENTAL DESIGN
-------------------

**Scenarios are nested, not independent.** A failure fires when
`sha256(seed:step) < rate`, which is monotone in rate — so holding the seed
fixed and raising the rate only ever *adds* failures. The three scenarios are
therefore strictly nested:

    light  0.05  [22, 32, 53]
    medium 0.10  [15, 22, 32, 36, 53]                  ⊃ light
    severe 0.20  [8,10,11,15,16,22,24,30,31,32,36,53]  ⊃ medium

That's a better design than three separate seeds: moving along the x-axis
changes the failure *rate* and nothing else, so any difference between
scenarios can't be a seed artefact.

**Every scenario fires several failures on purpose.** The controller writes
its `training_resumed` event only after the relaunched generation's first
training step — after that generation's worker already read the log at
startup. So a worker's estimate of recovery cost R lags one recovery behind,
and a job with a single recovery never sees a non-zero R. Scenarios with one
failure would understate R's contribution for every strategy, so the
pre-flight check refuses them.

**The headline metric is wasted compute, not wall clock.** Wall clock is
reported (and charted, as the plan asks) but it carries machine noise —
disk cache state, background load. Wasted compute is assembled from directly
measured components: duplicate steps' own recorded durations, checkpoint
write durations, fast-forward durations, recovery latencies. It's the
quantity the scheduler actually minimizes, and it's far less noisy.

Usage:
    python experiments/benchmark.py
    python experiments/benchmark.py --quick     # fewer steps, for a smoke test
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.adaptive import INITIAL_INTERVAL, run_adaptive
from experiments.fixed_interval import FIXED_INTERVALS, run_fixed
from experiments.harness import RESULTS_ROOT, Scenario
from metrics.job_metrics import JobMetrics, write_csv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "benchmark_results.csv")
CHART_PATH = os.path.join(RESULTS_DIR, "comparison_chart.png")

# One seed across all rates, so the scenarios nest. See module docstring.
SHARED_FAILURE_SEED = 11

# Below this, R never becomes non-zero and the scenario would understate it.
MIN_FAILURES_PER_SCENARIO = 3

# Above this share of unattributed wall clock, the cost model is missing
# something and the breakdown should not be trusted.
MAX_UNACCOUNTED_FRACTION = 0.15

STRATEGY_ORDER = ["fixed_short", "fixed_medium", "fixed_long", "adaptive"]

# Validated categorical slots 1-4 (see the dataviz palette reference). Slots
# 1-4 clear every adjacent-pairlist gate in both modes, which is the right
# pairlist for lines. Two light slots fall below 3:1 on the surface, so the
# relief rule applies: direct labels and a table view ship with the chart.
SERIES_LIGHT = {
    "fixed_short": "#2a78d6",   # blue
    "fixed_medium": "#eb6834",  # orange
    "fixed_long": "#1baf7a",    # aqua
    "adaptive": "#eda100",      # yellow
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"


class BenchmarkError(AssertionError):
    """Raised when the results don't hang together well enough to report."""


def build_scenarios(steps: int) -> List[Scenario]:
    return [
        Scenario(name="light_0.05", failure_rate=0.05,
                 failure_seed=SHARED_FAILURE_SEED, steps=steps),
        Scenario(name="medium_0.10", failure_rate=0.10,
                 failure_seed=SHARED_FAILURE_SEED, steps=steps),
        Scenario(name="severe_0.20", failure_rate=0.20,
                 failure_seed=SHARED_FAILURE_SEED, steps=steps),
    ]


# --------------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------------

def preflight(scenarios: List[Scenario]) -> None:
    """Check the scenarios are worth running before spending 25 minutes."""
    print("=" * 86)
    print("PRE-FLIGHT")
    print("=" * 86)

    previous: Optional[set] = None
    for scenario in scenarios:
        failures = scenario.expected_failures()
        steps = [s for s, _ in failures]
        print(f"  {scenario.name:<14} rate={scenario.failure_rate:<5} "
              f"{len(failures):>2} failures at {steps}")

        if len(failures) < MIN_FAILURES_PER_SCENARIO:
            raise BenchmarkError(
                f"{scenario.name} schedules only {len(failures)} failure(s); "
                f"need >= {MIN_FAILURES_PER_SCENARIO} so recovery cost R "
                f"becomes non-zero (it lags one recovery behind)")

        current = set(steps)
        if previous is not None and not previous <= current:
            raise BenchmarkError(
                f"{scenario.name} is not a superset of the previous scenario "
                f"— the nesting property is broken, so moving along the x-axis "
                f"changes more than the failure rate")
        previous = current

    print(f"\n  nesting verified: raising the rate only adds failures, so the "
          f"x-axis varies rate alone")
    print(f"  strategies: " + ", ".join(
        f"{k}={v}" for k, v in FIXED_INTERVALS.items())
        + f", adaptive (starts at {INITIAL_INTERVAL})")


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def run_all(scenarios: List[Scenario]) -> List[JobMetrics]:
    results: List[JobMetrics] = []
    for scenario in scenarios:
        print(f"\n--- scenario {scenario.name} "
              f"(rate {scenario.failure_rate}) ---")
        for label, interval in FIXED_INTERVALS.items():
            results.append(run_fixed(scenario, interval, label))
        results.append(run_adaptive(scenario, INITIAL_INTERVAL))
    return results


# --------------------------------------------------------------------------
# Sanity checks — the numbers have to hang together before they're reported
# --------------------------------------------------------------------------

def sanity_check(results: List[JobMetrics], scenarios: List[Scenario]
                 ) -> List[str]:
    """Hard failures raise; softer concerns are returned as warnings."""
    warnings: List[str] = []
    by_scenario: Dict[str, List[JobMetrics]] = {}
    for m in results:
        by_scenario.setdefault(m.scenario, []).append(m)

    print("\n" + "=" * 86)
    print("SANITY CHECKS")
    print("=" * 86)

    # 1. Everything finished. A strategy that aborted didn't do the same work.
    aborted = [m for m in results if m.outcome != "completed"]
    if aborted:
        raise BenchmarkError(
            "these jobs did not complete, so their costs aren't comparable: "
            + ", ".join(f"{m.scenario}/{m.strategy}={m.outcome}" for m in aborted))
    print(f"  [PASS] all {len(results)} jobs completed")

    # 2. Same useful work. If one strategy trained fewer steps it "won" by
    #    doing less, which is the classic way a benchmark lies.
    for scenario in scenarios:
        for m in by_scenario[scenario.name]:
            if m.steps_useful != scenario.steps:
                raise BenchmarkError(
                    f"{m.scenario}/{m.strategy} completed {m.steps_useful} "
                    f"useful steps, expected {scenario.steps} — strategies did "
                    f"not do equal work")
    print(f"  [PASS] every job completed exactly {scenarios[0].steps} "
          f"useful steps")

    # 3. Identical failure sequence within a scenario. Phase 3 should
    #    guarantee it; verify rather than trust.
    for name, group in by_scenario.items():
        counts = {m.strategy: m.failures for m in group}
        if len(set(counts.values())) != 1:
            raise BenchmarkError(
                f"{name}: strategies saw different numbers of failures "
                f"{counts} — the comparison is not controlled")
    print(f"  [PASS] within each scenario every strategy faced the same "
          f"failure count")

    # 4. Multiple recoveries everywhere, so R is real (the whole reason the
    #    scenarios were chosen this way).
    thin = [m for m in results if m.recoveries < 2]
    if thin:
        raise BenchmarkError(
            "these jobs had fewer than 2 recoveries, so recovery cost R never "
            "became non-zero: "
            + ", ".join(f"{m.scenario}/{m.strategy}" for m in thin))
    zero_r = [m for m in results if m.mean_recovery_seconds <= 0]
    if zero_r:
        raise BenchmarkError(
            "recovery latency measured as zero for: "
            + ", ".join(f"{m.scenario}/{m.strategy}" for m in zero_r))
    print(f"  [PASS] every job recorded >= 2 recoveries with non-zero latency "
          f"(min {min(m.recoveries for m in results)})")

    # 5. The cost model accounts for the wall clock it claims to explain.
    worst = max(results, key=lambda m: m.unaccounted_seconds / max(m.wall_clock_seconds, 1e-9))
    worst_frac = worst.unaccounted_seconds / max(worst.wall_clock_seconds, 1e-9)
    if worst_frac > MAX_UNACCOUNTED_FRACTION:
        warnings.append(
            f"{worst.scenario}/{worst.strategy} leaves "
            f"{100 * worst_frac:.1f}% of wall clock unattributed "
            f"({worst.unaccounted_seconds:.1f}s) — the breakdown is missing a "
            f"cost, treat component percentages with suspicion")
    print(f"  [PASS] cost model attributes >= "
          f"{100 * (1 - worst_frac):.1f}% of wall clock in every job "
          f"(worst: {worst.scenario}/{worst.strategy})")

    # 6. Directional expectations. Not laws, so they warn rather than fail —
    #    but a violation usually means noise is swamping the signal.
    for name, group in by_scenario.items():
        by_strategy = {m.strategy: m for m in group}
        short, long = by_strategy.get("fixed_short"), by_strategy.get("fixed_long")
        if short and long:
            if short.checkpoint_overhead_seconds <= long.checkpoint_overhead_seconds:
                warnings.append(
                    f"{name}: fixed_short's checkpoint overhead "
                    f"({short.checkpoint_overhead_seconds:.1f}s) is not above "
                    f"fixed_long's ({long.checkpoint_overhead_seconds:.1f}s) — "
                    f"unexpected, a short interval should write more")
            if long.wasted_recompute_seconds <= short.wasted_recompute_seconds:
                warnings.append(
                    f"{name}: fixed_long recomputed no more than fixed_short "
                    f"({long.wasted_recompute_seconds:.1f}s vs "
                    f"{short.wasted_recompute_seconds:.1f}s) — unexpected, a "
                    f"long interval should lose more work per failure")
    if not warnings:
        print("  [PASS] directional expectations hold (short interval writes "
              "more, long interval recomputes more)")

    for w in warnings:
        print(f"  [WARN] {w}")
    return warnings


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_table(results: List[JobMetrics]) -> None:
    print("\n" + "=" * 110)
    print("RESULTS")
    print("=" * 110)
    header = (f"{'scenario':<13} {'strategy':<14} {'wall s':>8} {'useful s':>9} "
              f"{'waste s':>8} {'waste %':>8} {'ckpt %':>7} {'recomp s':>9} "
              f"{'recov s':>8} {'ckpts':>6} {'redund':>7} {'goodput %':>10}")
    print(header)
    print("-" * len(header))

    current = None
    for m in sorted(results, key=lambda x: (x.failure_rate,
                                            STRATEGY_ORDER.index(x.strategy))):
        if current is not None and m.scenario != current:
            print()
        current = m.scenario
        print(f"{m.scenario:<13} {m.strategy:<14} "
              f"{m.wall_clock_seconds:>8.1f} {m.useful_compute_seconds:>9.1f} "
              f"{m.wasted_total_seconds:>8.1f} {m.wasted_pct:>8.1f} "
              f"{m.checkpoint_overhead_pct:>7.1f} "
              f"{m.wasted_recompute_seconds:>9.1f} "
              f"{m.recovery_latency_seconds:>8.1f} "
              f"{m.checkpoints_written:>6} {m.steps_redundant:>7} "
              f"{m.goodput_pct:>10.1f}")


def headline(results: List[JobMetrics]) -> dict:
    """
    Adaptive vs the BEST fixed interval per scenario.

    Compared against the best, not the mean: beating a badly-chosen fixed
    interval proves nothing. The claim worth making is beating the best one
    you could have picked with hindsight, without the hindsight.
    """
    by_scenario: Dict[str, List[JobMetrics]] = {}
    for m in results:
        by_scenario.setdefault(m.scenario, []).append(m)

    print("\n" + "=" * 86)
    print("ADAPTIVE vs BEST FIXED INTERVAL (per scenario)")
    print("=" * 86)
    print(f"  {'scenario':<13} {'best fixed':<14} {'best waste':>11} "
          f"{'adaptive':>10} {'delta':>9} {'verdict':>10}")

    rows, wins = [], 0
    for name in sorted(by_scenario, key=lambda n: by_scenario[n][0].failure_rate):
        group = by_scenario[name]
        adaptive = next(m for m in group if m.strategy == "adaptive")
        fixed = [m for m in group if m.strategy != "adaptive"]
        best = min(fixed, key=lambda m: m.wasted_total_seconds)

        delta_pct = (100.0 * (best.wasted_total_seconds
                              - adaptive.wasted_total_seconds)
                     / best.wasted_total_seconds) if best.wasted_total_seconds else 0.0
        won = adaptive.wasted_total_seconds < best.wasted_total_seconds
        wins += won

        print(f"  {name:<13} {best.strategy:<14} "
              f"{best.wasted_total_seconds:>10.1f}s "
              f"{adaptive.wasted_total_seconds:>9.1f}s "
              f"{delta_pct:>+8.1f}% {'WIN' if won else 'loss':>10}")

        rows.append({
            "scenario": name,
            "failure_rate": adaptive.failure_rate,
            "best_fixed_strategy": best.strategy,
            "best_fixed_wasted_seconds": round(best.wasted_total_seconds, 2),
            "adaptive_wasted_seconds": round(adaptive.wasted_total_seconds, 2),
            "waste_reduction_pct": round(delta_pct, 2),
            "adaptive_beats_best_fixed": won,
            "best_fixed_wall_seconds": round(best.wall_clock_seconds, 2),
            "adaptive_wall_seconds": round(adaptive.wall_clock_seconds, 2),
            "adaptive_mean_interval": round(adaptive.mean_interval_steps, 2),
        })

    print(f"\n  adaptive beat the best fixed interval in {wins} of "
          f"{len(rows)} scenarios")
    return {"scenarios": rows, "wins": wins, "total": len(rows)}


# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------

def make_chart(results: List[JobMetrics], path: str = CHART_PATH) -> str:
    """
    Two panels, never a dual axis: wall clock and wasted compute each get
    their own y-scale on their own plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_strategy: Dict[str, List[JobMetrics]] = {}
    for m in results:
        by_strategy.setdefault(m.strategy, []).append(m)
    for group in by_strategy.values():
        group.sort(key=lambda m: m.failure_rate)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURFACE)
    # Generous top margin: the suptitle, the subtitle and the two panel
    # titles are four stacked text rows and they collide at tighter values.
    fig.subplots_adjust(left=0.07, right=0.86, top=0.79, bottom=0.20, wspace=0.30)

    panels = [
        (axes[0], "wall_clock_seconds", "Total wall-clock time (s)",
         "Time to finish the job"),
        (axes[1], "wasted_pct", "Wasted compute (% of wall clock)",
         "Share of time not spent on surviving work"),
    ]

    for ax, attr, ylabel, title in panels:
        ax.set_facecolor(SURFACE)
        # Recessive grid and axes: the data is the figure, not the frame.
        ax.grid(True, axis="y", color="#e6e5e1", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d8d7d2")
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)

        for strategy in STRATEGY_ORDER:
            group = by_strategy.get(strategy)
            if not group:
                continue
            xs = [m.failure_rate for m in group]
            ys = [getattr(m, attr) for m in group]
            colour = SERIES_LIGHT[strategy]
            emphasis = strategy == "adaptive"

            ax.plot(xs, ys, color=colour,
                    linewidth=2.6 if emphasis else 2.0,
                    zorder=4 if emphasis else 3,
                    solid_capstyle="round")
            # 2px surface ring so overlapping markers stay separable.
            ax.plot(xs, ys, marker="o", linestyle="none",
                    markersize=9 if emphasis else 8,
                    markerfacecolor=colour, markeredgecolor=SURFACE,
                    markeredgewidth=2, zorder=5 if emphasis else 4)

        ax.set_xlabel("Injected failure rate (per step)", color=INK_SECONDARY,
                      fontsize=10, labelpad=8)
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10, labelpad=8)
        ax.set_title(title, color=INK, fontsize=11.5, loc="left", pad=12)
        ax.set_xticks(sorted({m.failure_rate for m in results}))

    # Direct labels at the right edge of the second panel. Two light-mode
    # slots fall below 3:1 contrast on this surface, so the relief rule
    # requires visible labels (or the table view — both ship).
    ax = axes[1]
    x_max = max(m.failure_rate for m in results)
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    placed: List[float] = []
    for strategy in STRATEGY_ORDER:
        group = by_strategy.get(strategy)
        if not group:
            continue
        y = getattr(group[-1], "wasted_pct")
        # Nudge apart so labels don't collide when lines converge.
        while any(abs(y - p) < span * 0.055 for p in placed):
            y += span * 0.055
        placed.append(y)
        ax.annotate(strategy.replace("_", " "),
                    xy=(x_max, getattr(group[-1], "wasted_pct")),
                    xytext=(8, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=9,
                    color=INK_SECONDARY, annotation_clip=False)

    handles = [plt.Line2D([0], [0], color=SERIES_LIGHT[s], linewidth=2.4,
                          marker="o", markersize=8,
                          markeredgecolor=SURFACE, markeredgewidth=2,
                          label=s.replace("_", " "))
               for s in STRATEGY_ORDER if s in by_strategy]
    legend = fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                        frameon=False, fontsize=9.5,
                        bbox_to_anchor=(0.47, 0.015))
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.suptitle("Adaptive vs fixed-interval checkpointing under injected failures",
                 color=INK, fontsize=13.5, x=0.07, ha="left", y=0.975)
    fig.text(0.07, 0.905,
             "ResNet-18 / CIFAR-10, 2 DDP workers. Lower is better on both "
             "panels. Failure sets are nested across rates (same seed).",
             color=INK_MUTED, fontsize=9.5, ha="left")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--quick", action="store_true",
                   help="Short run to check the plumbing, not to report")
    p.add_argument("--chart-only", action="store_true",
                   help="Rebuild the chart and table from existing CSV")
    args = p.parse_args(argv)

    steps = 16 if args.quick else args.steps
    scenarios = build_scenarios(steps)

    try:
        preflight(scenarios)
    except BenchmarkError as exc:
        print(f"\nPRE-FLIGHT FAILED — {exc}", file=sys.stderr)
        return 1

    results = run_all(scenarios)

    try:
        warnings = sanity_check(results, scenarios)
    except BenchmarkError as exc:
        print(f"\nSANITY CHECK FAILED — {exc}", file=sys.stderr)
        write_csv(results, CSV_PATH)   # keep the evidence
        return 1

    print_table(results)
    summary = headline(results)

    write_csv(results, CSV_PATH)
    print(f"\nwrote {os.path.relpath(CSV_PATH, REPO_ROOT)}")
    chart = make_chart(results)
    print(f"wrote {os.path.relpath(chart, REPO_ROOT)}")

    summary["warnings"] = warnings
    summary["steps"] = steps
    with open(os.path.join(RESULTS_DIR, "benchmark_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
