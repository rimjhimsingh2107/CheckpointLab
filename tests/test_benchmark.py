"""
Tests for the Phase 6 benchmark suite.

A benchmark is a measuring instrument, and an instrument that quietly reports
a wrong number is worse than one that breaks. So most of these tests are
about the *guards*: the checks that refuse to report results which aren't
comparable. Each one corresponds to a specific way this benchmark could
flatter a strategy.

The cost accounting is tested against hand-built log files, so the expected
values are arithmetic rather than "whatever the code produced".
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import benchmark as B
from experiments.adaptive import INITIAL_INTERVAL, STRATEGY
from experiments.fixed_interval import FIXED_INTERVALS
from experiments.harness import Scenario
from metrics.job_metrics import JobMetrics, collect_job_metrics, write_csv


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _job(strategy="adaptive", scenario="s", rate=0.1, wall=100.0, **kw):
    m = JobMetrics(strategy=strategy, scenario=scenario, failure_rate=rate,
                   wall_clock_seconds=wall)
    m.steps_useful = kw.pop("steps_useful", 60)
    m.outcome = kw.pop("outcome", "completed")
    m.failures = kw.pop("failures", 3)
    m.recoveries = kw.pop("recoveries", 3)
    m.mean_recovery_seconds = kw.pop("mean_recovery_seconds", 3.7)
    m.useful_compute_seconds = kw.pop("useful_compute_seconds", 66.0)
    m.wasted_recompute_seconds = kw.pop("wasted_recompute_seconds", 10.0)
    m.checkpoint_overhead_seconds = kw.pop("checkpoint_overhead_seconds", 5.0)
    m.fast_forward_seconds = kw.pop("fast_forward_seconds", 0.2)
    m.recovery_latency_seconds = kw.pop("recovery_latency_seconds", 11.1)
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def _matrix(**overrides):
    """A full 4-strategy x 1-scenario set that passes every check."""
    jobs = []
    for strategy in B.STRATEGY_ORDER:
        jobs.append(_job(strategy=strategy, scenario="light_0.05", rate=0.05))
    # Directional expectations: short writes most, long recomputes most.
    by = {j.strategy: j for j in jobs}
    by["fixed_short"].checkpoint_overhead_seconds = 12.0
    by["fixed_short"].wasted_recompute_seconds = 3.0
    by["fixed_long"].checkpoint_overhead_seconds = 1.0
    by["fixed_long"].wasted_recompute_seconds = 40.0
    for strategy, changes in overrides.items():
        for k, v in changes.items():
            setattr(by[strategy], k, v)
    return jobs


SCENARIOS = [Scenario(name="light_0.05", failure_rate=0.05, failure_seed=11,
                      steps=60)]


# --------------------------------------------------------------------------
# Cost accounting
# --------------------------------------------------------------------------

def test_wasted_recompute_comes_from_duplicate_steps(tmp_path):
    """
    A global_step appearing twice was computed twice; the earlier record's own
    duration is exactly what that redundant work cost. No estimation.
    """
    job = str(tmp_path / "job")
    _write_jsonl(os.path.join(job, "loss_log.jsonl"), [
        {"global_step": 1, "step_seconds": 1.0},
        {"global_step": 2, "step_seconds": 1.0},
        {"global_step": 3, "step_seconds": 1.0},   # lost to a crash
        {"global_step": 3, "step_seconds": 2.0},   # recomputed, survived
        {"global_step": 4, "step_seconds": 1.5},
    ])
    m = collect_job_metrics(job, "adaptive", "s", 0.1, wall_clock_seconds=20.0)

    assert m.steps_computed == 5
    assert m.steps_useful == 4
    assert m.steps_redundant == 1
    # Surviving: 1.0 + 1.0 + 2.0 + 1.5 ; redundant: the first step-3 record.
    assert m.useful_compute_seconds == pytest.approx(5.5)
    assert m.wasted_recompute_seconds == pytest.approx(1.0)


def test_checkpoint_and_fast_forward_costs_are_separated(tmp_path):
    job = str(tmp_path / "job")
    _write_jsonl(os.path.join(job, "loss_log.jsonl"),
                 [{"global_step": 1, "step_seconds": 1.0}])
    _write_jsonl(os.path.join(job, "checkpoint_log.jsonl"), [
        {"event": "checkpoint", "duration_seconds": 0.4, "interval_used": 4},
        {"event": "checkpoint", "duration_seconds": 0.6, "interval_used": 6},
        {"event": "fast_forward", "duration_seconds": 0.25, "steps_skipped": 9},
    ])
    m = collect_job_metrics(job, "adaptive", "s", 0.1, wall_clock_seconds=10.0)

    assert m.checkpoints_written == 2
    assert m.checkpoint_overhead_seconds == pytest.approx(1.0)
    assert m.mean_checkpoint_seconds == pytest.approx(0.5)
    assert m.mean_interval_steps == pytest.approx(5.0)
    # Fast-forward must NOT be folded into checkpoint overhead: it is caused
    # by the interval but is a different cost, and omitting it would subsidise
    # the long-interval strategies.
    assert m.fast_forward_seconds == pytest.approx(0.25)
    assert m.fast_forward_steps == 9


def test_wasted_total_is_the_sum_of_its_parts(tmp_path):
    m = _job(wasted_recompute_seconds=10.0, checkpoint_overhead_seconds=5.0,
             fast_forward_seconds=0.5, recovery_latency_seconds=11.0)
    assert m.wasted_total_seconds == pytest.approx(26.5)
    assert m.accounted_seconds == pytest.approx(66.0 + 26.5)


def test_unaccounted_is_the_residual_and_never_negative():
    m = _job(wall=100.0)
    assert m.unaccounted_seconds == pytest.approx(100.0 - m.accounted_seconds)
    # An over-attributing model must clamp rather than report negative time.
    tight = _job(wall=1.0)
    assert tight.unaccounted_seconds == 0.0


def test_percentages_are_shares_of_wall_clock():
    m = _job(wall=200.0, useful_compute_seconds=100.0,
             checkpoint_overhead_seconds=20.0, wasted_recompute_seconds=0.0,
             fast_forward_seconds=0.0, recovery_latency_seconds=0.0)
    assert m.goodput_pct == pytest.approx(50.0)
    assert m.checkpoint_overhead_pct == pytest.approx(10.0)
    assert m.wasted_pct == pytest.approx(10.0)


def test_zero_wall_clock_does_not_divide_by_zero():
    m = _job(wall=0.0)
    assert m.goodput_pct == 0.0
    assert m.throughput_steps_per_s == 0.0


def test_metrics_survive_missing_logs(tmp_path):
    """A job whose logs are absent should read as zeros, not crash."""
    m = collect_job_metrics(str(tmp_path / "nothing"), "adaptive", "s", 0.1,
                            wall_clock_seconds=5.0)
    assert m.steps_useful == 0
    assert m.wasted_total_seconds == 0.0


def test_metrics_survive_a_truncated_final_line(tmp_path):
    job = str(tmp_path / "job")
    os.makedirs(job)
    with open(os.path.join(job, "loss_log.jsonl"), "w") as fh:
        fh.write(json.dumps({"global_step": 1, "step_seconds": 1.0}) + "\n")
        fh.write('{"global_step": 2, "step_sec')      # killed mid-write
    m = collect_job_metrics(job, "adaptive", "s", 0.1, wall_clock_seconds=5.0)
    assert m.steps_useful == 1


def test_csv_round_trip(tmp_path):
    path = str(tmp_path / "r.csv")
    write_csv(_matrix(), path)
    lines = open(path).read().strip().splitlines()
    assert len(lines) == 5                       # header + 4 strategies
    header = lines[0].split(",")
    for column in ("strategy", "scenario", "wall_clock_seconds",
                   "wasted_total_seconds", "checkpoint_overhead_pct",
                   "goodput_pct", "unaccounted_seconds"):
        assert column in header


# --------------------------------------------------------------------------
# Pre-flight guards
# --------------------------------------------------------------------------

def test_preflight_rejects_a_scenario_with_too_few_failures():
    """
    R lags one recovery behind, so a scenario with one failure never produces
    a non-zero recovery cost and would understate it for every strategy.
    """
    thin = [Scenario(name="too_calm", failure_rate=0.001, failure_seed=11,
                     steps=20)]
    assert len(thin[0].expected_failures()) < B.MIN_FAILURES_PER_SCENARIO
    with pytest.raises(B.BenchmarkError, match="recovery cost R"):
        B.preflight(thin)


def test_preflight_accepts_the_real_scenarios():
    scenarios = B.build_scenarios(60)
    B.preflight(scenarios)                       # must not raise
    for s in scenarios:
        assert len(s.expected_failures()) >= B.MIN_FAILURES_PER_SCENARIO


def test_scenarios_are_nested_so_the_x_axis_varies_rate_alone():
    """
    A failure fires when hash(seed:step) < rate, which is monotone in rate.
    Holding the seed fixed, raising the rate only ADDS failures — so moving
    along the x-axis can't be a seed artefact.
    """
    scenarios = B.build_scenarios(60)
    sets = [{step for step, _ in s.expected_failures()} for s in scenarios]
    for smaller, larger in zip(sets, sets[1:]):
        assert smaller < larger, "scenarios must strictly nest"
    assert len({s.failure_seed for s in scenarios}) == 1


def test_preflight_rejects_non_nested_scenarios():
    scenarios = [
        Scenario(name="a", failure_rate=0.2, failure_seed=11, steps=60),
        Scenario(name="b", failure_rate=0.2, failure_seed=7, steps=60),
    ]
    with pytest.raises(B.BenchmarkError, match="nesting property"):
        B.preflight(scenarios)


# --------------------------------------------------------------------------
# Sanity-check guards — each is a way the benchmark could lie
# --------------------------------------------------------------------------

def test_sanity_passes_on_a_clean_matrix():
    assert B.sanity_check(_matrix(), SCENARIOS) == []


def test_sanity_rejects_an_aborted_job():
    with pytest.raises(B.BenchmarkError, match="did not complete"):
        B.sanity_check(_matrix(adaptive={"outcome": "max_restarts_exceeded"}),
                       SCENARIOS)


def test_sanity_rejects_unequal_useful_work():
    """The classic way a benchmark lies: one strategy 'wins' by doing less."""
    with pytest.raises(B.BenchmarkError, match="not do equal work"):
        B.sanity_check(_matrix(adaptive={"steps_useful": 40}), SCENARIOS)


def test_sanity_rejects_differing_failure_counts():
    """Every strategy must face the identical failure sequence."""
    with pytest.raises(B.BenchmarkError, match="different numbers of failures"):
        B.sanity_check(_matrix(adaptive={"failures": 1}), SCENARIOS)


def test_sanity_rejects_a_job_with_too_few_recoveries():
    with pytest.raises(B.BenchmarkError, match="fewer than 2 recoveries"):
        B.sanity_check(_matrix(fixed_long={"recoveries": 1}), SCENARIOS)


def test_sanity_rejects_zero_measured_recovery_latency():
    with pytest.raises(B.BenchmarkError, match="recovery latency measured as zero"):
        B.sanity_check(_matrix(adaptive={"mean_recovery_seconds": 0.0}),
                       SCENARIOS)


def test_sanity_warns_when_too_much_wall_clock_is_unattributed():
    """A large residual means the cost model is missing something."""
    warnings = B.sanity_check(
        _matrix(adaptive={"wall_clock_seconds": 400.0}), SCENARIOS)
    assert any("unattributed" in w for w in warnings)


def test_sanity_warns_on_inverted_directional_expectations():
    """Short interval should write more; long should recompute more."""
    warnings = B.sanity_check(
        _matrix(fixed_short={"checkpoint_overhead_seconds": 0.1},
                fixed_long={"wasted_recompute_seconds": 0.1}), SCENARIOS)
    assert any("checkpoint overhead" in w for w in warnings)
    assert any("recomputed no more" in w for w in warnings)


# --------------------------------------------------------------------------
# Headline comparison
# --------------------------------------------------------------------------

def test_headline_compares_against_the_best_fixed_not_the_mean():
    """
    Beating a badly-chosen fixed interval proves nothing. The comparison has
    to be against the best interval you could have picked with hindsight.
    """
    jobs = _matrix()
    by = {j.strategy: j for j in jobs}
    by["fixed_short"].wasted_recompute_seconds = 0.0     # best fixed
    by["fixed_short"].checkpoint_overhead_seconds = 5.0
    by["fixed_medium"].wasted_recompute_seconds = 100.0  # terrible
    by["fixed_long"].wasted_recompute_seconds = 200.0    # worse
    by["adaptive"].wasted_recompute_seconds = 0.0
    by["adaptive"].checkpoint_overhead_seconds = 4.0     # narrowly better

    summary = B.headline(jobs)
    row = summary["scenarios"][0]
    assert row["best_fixed_strategy"] == "fixed_short"
    assert row["adaptive_beats_best_fixed"] is True
    assert 0 < row["waste_reduction_pct"] < 100


def test_headline_reports_a_loss_honestly():
    jobs = _matrix(adaptive={"wasted_recompute_seconds": 500.0})
    summary = B.headline(jobs)
    assert summary["wins"] == 0
    assert summary["scenarios"][0]["adaptive_beats_best_fixed"] is False
    assert summary["scenarios"][0]["waste_reduction_pct"] < 0


# --------------------------------------------------------------------------
# Strategy configuration
# --------------------------------------------------------------------------

def test_three_distinct_fixed_intervals_are_covered():
    assert len(set(FIXED_INTERVALS.values())) == 3
    intervals = list(FIXED_INTERVALS.values())
    assert intervals == sorted(intervals), "short < medium < long"


def test_adaptive_starts_from_the_medium_fixed_interval():
    """
    Otherwise part of any difference would be attributable to a different
    starting point rather than to the adaptation.
    """
    assert INITIAL_INTERVAL == FIXED_INTERVALS["fixed_medium"]


def test_strategy_order_covers_every_strategy():
    assert set(B.STRATEGY_ORDER) == set(FIXED_INTERVALS) | {STRATEGY}


def test_every_series_has_a_validated_colour():
    for strategy in B.STRATEGY_ORDER:
        assert strategy in B.SERIES_LIGHT
    assert len(set(B.SERIES_LIGHT.values())) == len(B.STRATEGY_ORDER), \
        "each series needs its own hue; colour must never be reused"


# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------

def test_chart_renders_to_a_png(tmp_path):
    path = str(tmp_path / "chart.png")
    jobs = []
    for rate, name in ((0.05, "light_0.05"), (0.10, "medium_0.10"),
                       (0.20, "severe_0.20")):
        for strategy in B.STRATEGY_ORDER:
            jobs.append(_job(strategy=strategy, scenario=name, rate=rate))
    B.make_chart(jobs, path)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 10_000
    with open(path, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"
