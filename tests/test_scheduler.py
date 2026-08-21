"""
Tests for the Phase 5 adaptive checkpoint scheduler.

The load-bearing test here is `test_closed_form_is_the_argmin_of_the_waste_
function`: it grid-searches W(T) numerically and asserts sqrt(2dM) really is
where the minimum sits. That's what keeps the docstring's derivation and the
code honest with each other — if someone "improves" the formula without
updating the objective it claims to minimize, that test fails.

The rest checks the estimators (which is where the subtle bugs live: censored
intervals, empty windows, divide-by-zero) and the monotonicity properties the
whole design rests on.
"""
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkpoint.scheduler import (
    CheckpointScheduler,
    failure_times_from_log,
    scheduler_from_job,
)
from metrics.recovery_timing import (
    PHASES,
    RecoveryTiming,
    load_recovery_timings,
    summarize,
)


def _primed(d=0.4, step=1.0, window=20, initial=5, started_at=0.0, **kw):
    """A scheduler with a known checkpoint cost and throughput."""
    s = CheckpointScheduler(initial_interval_steps=initial, window=window,
                            started_at=started_at, **kw)
    s.record_checkpoint(d)
    s.record_step(step)
    return s


# --------------------------------------------------------------------------
# The math
# --------------------------------------------------------------------------

@pytest.mark.parametrize("d", [0.05, 0.4, 1.0, 3.0])
@pytest.mark.parametrize("m", [2.0, 5.0, 30.0, 300.0, 3600.0])
@pytest.mark.parametrize("r", [0.0, 4.0, 60.0])
def test_closed_form_is_the_argmin_of_the_waste_function(d, m, r):
    """
    T* = sqrt(2dM) must be where W(T) = d/T + T/(2M) + R/M is minimized.

    Grid-searched rather than differentiated, so this checks the *code*, not
    the algebra a second time.
    """
    s = _primed(d=d)
    s.record_recovery(r)
    now = m                       # no failures => censored gap == m
    assert s.mtbf(now) == pytest.approx(m)

    analytic = s.optimal_interval_seconds(now)

    lo, hi, n = analytic * 0.2, analytic * 5.0, 20000
    best_t, best_w = None, math.inf
    for i in range(1, n + 1):
        t = lo + (hi - lo) * i / n
        w = s.expected_waste(t, now)
        if w < best_w:
            best_w, best_t = w, t

    assert best_t == pytest.approx(analytic, rel=1e-3)
    # And the analytic point is genuinely no worse than the grid's best.
    assert s.expected_waste(analytic, now) <= best_w * (1 + 1e-9)


@pytest.mark.parametrize("r", [0.0, 4.0, 60.0, 600.0])
def test_recovery_cost_shifts_waste_but_not_the_optimum(r):
    """
    R/M is constant in T, so a slow restart makes failures more expensive
    without arguing for more frequent checkpoints. Worth pinning down because
    it's the most counter-intuitive consequence of the model.
    """
    s = _primed(d=0.4)
    s.record_recovery(r)
    assert s.optimal_interval_seconds(30.0) == pytest.approx(math.sqrt(2 * 0.4 * 30))
    # ...but the reported waste does rise with R.
    assert s.expected_waste(5.0, 30.0) == pytest.approx(
        0.4 / 5.0 + 5.0 / 60.0 + r / 30.0)


def test_waste_function_matches_the_documented_formula():
    s = _primed(d=0.5)
    s.record_recovery(3.0)
    d, m, r, t = 0.5, 20.0, 3.0, 7.0
    assert s.mtbf(20.0) == pytest.approx(m)
    assert s.expected_waste(t, 20.0) == pytest.approx(d / t + t / (2 * m) + r / m)


def test_interval_scales_as_sqrt_of_mtbf():
    """100x the MTBF should give 10x the interval, not 100x."""
    s = _primed(d=0.4)
    short = s.optimal_interval_seconds(10.0)
    long = s.optimal_interval_seconds(1000.0)
    assert long / short == pytest.approx(10.0, rel=1e-6)


def test_interval_scales_as_sqrt_of_checkpoint_cost():
    cheap = _primed(d=0.01).optimal_interval_seconds(100.0)
    dear = _primed(d=1.0).optimal_interval_seconds(100.0)
    assert dear / cheap == pytest.approx(10.0, rel=1e-6)


# --------------------------------------------------------------------------
# Direction of adaptation — the milestone property, unit-tested
# --------------------------------------------------------------------------

def test_higher_failure_rate_gives_a_shorter_interval():
    """
    The Phase 5 milestone in miniature: same checkpoint cost and throughput,
    different observed failure history, shorter interval under load.
    """
    calm = _primed(d=0.4, step=1.0)
    calm.observe_failures([0.0, 100.0, 200.0])          # every ~100s

    storm = _primed(d=0.4, step=1.0)
    storm.observe_failures([0.0, 3.0, 6.0, 9.0])        # every ~3s

    now = 300.0
    assert storm.mtbf(now) < calm.mtbf(now)
    assert (storm.decide(now).interval_steps
            < calm.decide(now).interval_steps)


def test_interval_lengthens_as_a_failure_free_stretch_grows():
    """
    Stability has to be rewarded, or a job that stabilised keeps paying the
    write cost of its worst moment forever.
    """
    s = _primed(d=0.4, step=1.0)
    s.observe_failures([0.0, 10.0])

    intervals = [s.decide(now).interval_steps for now in (20, 60, 300, 3000)]
    assert intervals == sorted(intervals)
    assert intervals[-1] > intervals[0]


def test_expensive_checkpoints_lengthen_the_interval():
    now = 100.0
    cheap = _primed(d=0.02, step=1.0).decide(now).interval_steps
    dear = _primed(d=2.0, step=1.0).decide(now).interval_steps
    assert dear > cheap


def test_scheduler_never_reads_the_configured_failure_rate():
    """
    The scheduler's whole claim is that adaptation is emergent. It should be
    impossible to tell it what the failure rate is.

    Checked over identifiers parsed from the AST rather than raw text: the
    module docstring legitimately *discusses* failure_rate when quoting the
    project plan's formula, and a substring search over source would flag
    that prose as a violation.
    """
    import ast
    import inspect

    import checkpoint.scheduler as mod

    tree = ast.parse(inspect.getsource(mod))
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            identifiers.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            identifiers.add(node.name)

    offenders = {name for name in identifiers if "failure_rate" in name}
    assert not offenders, \
        f"scheduler reads a configured failure rate via {sorted(offenders)}"

    params = inspect.signature(CheckpointScheduler.__init__).parameters
    assert not any("rate" in p for p in params)

    # It does still need failure *timestamps* — that's an observation, not a
    # configured parameter, and it's the whole basis for the adaptation.
    assert "observe_failures" in dir(CheckpointScheduler)


# --------------------------------------------------------------------------
# Estimators and edge cases
# --------------------------------------------------------------------------

def test_no_observations_falls_back_to_the_initial_interval():
    """With nothing measured it must behave exactly like Phase 2."""
    s = CheckpointScheduler(initial_interval_steps=7, started_at=0.0)
    decision = s.decide(10.0)
    assert decision.interval_steps == 7
    assert "no checkpoint cost" in decision.reason


def test_throughput_measured_but_no_checkpoint_cost_uses_initial():
    s = CheckpointScheduler(initial_interval_steps=9, started_at=0.0)
    s.record_step(1.0)
    assert s.decide(10.0).interval_steps == 9


def test_checkpoint_cost_but_no_throughput_uses_initial():
    s = CheckpointScheduler(initial_interval_steps=11, started_at=0.0)
    s.record_checkpoint(0.4)
    decision = s.decide(10.0)
    assert decision.interval_steps == 11
    assert "throughput" in decision.reason


def test_zero_checkpoint_cost_is_not_a_divide_by_zero():
    s = CheckpointScheduler(initial_interval_steps=5, started_at=0.0)
    s.record_checkpoint(0.0)
    s.record_step(1.0)
    assert s.optimal_interval_seconds(10.0) == 0.0
    assert s.decide(10.0).interval_steps >= 1


def test_mtbf_uses_gaps_plus_the_censored_interval():
    s = _primed()
    s.observe_failures([0.0, 10.0, 20.0])   # gaps: 10, 10
    # censored = now - last = 30 - 20 = 10 -> mean([10, 10, 10]) == 10
    assert s.mtbf(30.0) == pytest.approx(10.0)
    # A long quiet stretch pulls the mean up: mean([10, 10, 100])
    assert s.mtbf(120.0) == pytest.approx(40.0)


def test_mtbf_with_no_failures_uses_time_since_start():
    s = _primed(started_at=5.0)
    assert s.failures_observed == 0
    assert s.mtbf(65.0) == pytest.approx(60.0)


def test_mtbf_respects_its_floor():
    """
    A whole world dying at once can log several failures microseconds apart.
    Without a floor that implies an MTBF near zero and an interval of "every
    step forever".
    """
    s = _primed(mtbf_floor=0.5)
    s.observe_failures([100.0, 100.000001, 100.000002])
    assert s.mtbf(100.000003) == pytest.approx(0.5)


def test_mtbf_ignores_non_positive_gaps():
    s = _primed()
    s.observe_failures([10.0, 10.0, 20.0])   # duplicate timestamp -> zero gap
    assert s.mtbf(30.0) > 0


def test_rolling_window_forgets_old_observations():
    s = CheckpointScheduler(initial_interval_steps=5, window=3, started_at=0.0)
    for cost in (10.0, 10.0, 10.0, 0.1, 0.1, 0.1):
        s.record_checkpoint(cost)
    assert s.checkpoint_cost() == pytest.approx(0.1)


def test_throughput_is_mean_step_rate():
    s = CheckpointScheduler(initial_interval_steps=5, started_at=0.0)
    for duration in (0.5, 1.0, 1.5):
        s.record_step(duration)
    assert s.throughput() == pytest.approx(1.0 / 1.0)


def test_negative_and_none_observations_are_ignored():
    s = CheckpointScheduler(initial_interval_steps=5, started_at=0.0)
    s.record_checkpoint(None)
    s.record_checkpoint(-1.0)
    s.record_step(None)
    s.record_step(-1.0)
    s.record_step(0.0)
    assert s.checkpoint_cost() == 0.0
    assert s.throughput() == 0.0


def test_interval_is_clamped_to_bounds():
    low = _primed(d=0.4, step=1.0, min_interval_steps=8)
    low.observe_failures([0.0, 0.6, 1.2])         # frantic -> wants < 8
    decision = low.decide(1.5)
    assert decision.interval_steps == 8
    assert decision.clamped == "min"

    high = _primed(d=0.4, step=1.0, max_interval_steps=3)
    decision = high.decide(1e6)                   # serene -> wants > 3
    assert decision.interval_steps == 3
    assert decision.clamped == "max"


@pytest.mark.parametrize("kwargs", [
    dict(initial_interval_steps=0),
    dict(initial_interval_steps=5, min_interval_steps=0),
    dict(initial_interval_steps=5, min_interval_steps=10, max_interval_steps=2),
])
def test_invalid_construction_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CheckpointScheduler(**kwargs)


def test_decision_records_every_input():
    """A chosen interval must be auditable from the log alone."""
    s = _primed(d=0.4, step=1.0)
    s.record_recovery(3.0)
    s.observe_failures([0.0, 20.0])
    record = s.decide(40.0, global_step=17).as_dict()

    for key in ("global_step", "interval_steps", "optimal_seconds",
                "checkpoint_cost", "mtbf", "recovery_cost", "throughput",
                "expected_waste", "failures_observed", "reason"):
        assert key in record
    assert record["global_step"] == 17
    assert record["failures_observed"] == 2
    json.dumps(record)   # must be loggable as-is


# --------------------------------------------------------------------------
# Rebuilding state from a crashed job's logs
# --------------------------------------------------------------------------

def test_failure_times_from_log(tmp_path):
    log = tmp_path / "failure_log.jsonl"
    log.write_text(
        json.dumps({"event": "injected_failure", "timestamp": 30.0}) + "\n"
        + json.dumps({"event": "injected_failure", "timestamp": 10.0}) + "\n"
        + json.dumps({"event": "something_else", "timestamp": 99.0}) + "\n")
    assert failure_times_from_log(str(log)) == [10.0, 30.0]


def test_failure_times_from_log_survives_truncation(tmp_path):
    """The worker that died may not have finished its last line."""
    log = tmp_path / "failure_log.jsonl"
    log.write_text(
        json.dumps({"event": "injected_failure", "timestamp": 10.0}) + "\n"
        + '{"event": "injected_failure", "timesta')
    assert failure_times_from_log(str(log)) == [10.0]


def test_failure_times_from_missing_log_is_empty(tmp_path):
    assert failure_times_from_log(str(tmp_path / "nope.jsonl")) == []


def test_scheduler_from_job_inherits_failures_and_recovery_cost(tmp_path):
    """
    A restarted generation must inherit its predecessors' hard-won
    knowledge. Starting fresh would make every crash-restart look like proof
    of a healthy system.
    """
    job = tmp_path / "job"
    job.mkdir()
    (job / "failure_log.jsonl").write_text(
        json.dumps({"event": "injected_failure", "timestamp": 100.0}) + "\n"
        + json.dumps({"event": "injected_failure", "timestamp": 110.0}) + "\n")
    (job / "recovery_log.jsonl").write_text(
        json.dumps({"event": "training_resumed", "attempt": 1,
                    "failed_rank": 0, "failure_kind": "exit",
                    "recovery_seconds": 4.0,
                    "breakdown": {"terminate_workers": 0.1,
                                  "select_checkpoint": 0.1,
                                  "relaunch_workers": 0.1,
                                  "reach_first_step": 3.7}}) + "\n")

    s = scheduler_from_job(str(job), initial_interval_steps=5)
    assert s.failures_observed == 2
    assert s.recovery_cost() == pytest.approx(4.0)
    # started_at anchors on the first failure so a one-failure job still has
    # an interval to reason about.
    assert s.started_at == pytest.approx(100.0)


def test_scheduler_from_empty_job_dir(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    s = scheduler_from_job(str(job), initial_interval_steps=6)
    assert s.failures_observed == 0
    assert s.decide(10.0).interval_steps == 6


# --------------------------------------------------------------------------
# metrics/recovery_timing.py — reusable for Phase 6
# --------------------------------------------------------------------------

def test_recovery_timing_breakdown_and_total():
    t = RecoveryTiming(attempt=1, failed_rank=0, failure_kind="exit",
                       recovery_seconds=4.0, terminate_workers=0.1,
                       select_checkpoint=0.2, relaunch_workers=0.05,
                       reach_first_step=3.6)
    assert set(t.breakdown) == set(PHASES)
    assert t.accounted_seconds == pytest.approx(3.95)
    assert list(t.breakdown) == list(PHASES), "breakdown must stay ordered"


def test_recovery_timing_round_trips_through_the_event_log(tmp_path):
    original = RecoveryTiming(attempt=2, failed_rank=1, failure_kind="exit",
                              recovery_seconds=3.5, terminate_workers=0.06,
                              select_checkpoint=0.02, relaunch_workers=0.01,
                              reach_first_step=3.41, exit_code=137,
                              step_at_failure=31, restored_to_step=30,
                              steps_lost=1)
    log = tmp_path / "recovery_log.jsonl"
    log.write_text(json.dumps({"event": "training_resumed",
                               **original.as_dict()}) + "\n")

    loaded = load_recovery_timings(str(log))
    assert loaded == [original]


def test_load_recovery_timings_ignores_other_events_and_truncation(tmp_path):
    log = tmp_path / "recovery_log.jsonl"
    log.write_text(
        json.dumps({"event": "job_started"}) + "\n"
        + json.dumps({"event": "training_resumed", "attempt": 1,
                      "failed_rank": 0, "recovery_seconds": 2.0}) + "\n"
        + '{"event": "training_resumed", "att')
    timings = load_recovery_timings(str(log))
    assert len(timings) == 1
    assert timings[0].recovery_seconds == 2.0


def test_summarize_empty_is_zeros_not_an_error():
    """A job that never failed is a valid benchmark data point."""
    stats = summarize([])
    assert stats["recoveries"] == 0
    assert stats["mean_seconds"] == 0.0
    assert set(stats["mean_breakdown"]) == set(PHASES)


def test_summarize_aggregates_totals_and_phases():
    a = RecoveryTiming(attempt=1, failed_rank=0, failure_kind="exit",
                       recovery_seconds=2.0, reach_first_step=2.0, steps_lost=1)
    b = RecoveryTiming(attempt=2, failed_rank=1, failure_kind="exit",
                       recovery_seconds=4.0, reach_first_step=4.0, steps_lost=3)
    stats = summarize([a, b])
    assert stats["recoveries"] == 2
    assert stats["mean_seconds"] == pytest.approx(3.0)
    assert stats["min_seconds"] == pytest.approx(2.0)
    assert stats["max_seconds"] == pytest.approx(4.0)
    assert stats["steps_lost"] == 4
    assert stats["mean_breakdown"]["reach_first_step"] == pytest.approx(3.0)


def test_mtbf_excludes_recovery_downtime():
    """
    A worker cannot fail while it is being restarted, so the recovery cost
    inside each gap is not exposure to failure. Measured on this testbed
    under load, R was ~42% of the wall-clock gap — leaving it in inflates M
    and lengthens the interval exactly when work is most at risk.
    """
    s = _primed(d=0.4, step=1.0)
    s.observe_failures([0.0, 10.0, 20.0])

    without_downtime = s.mtbf(30.0)      # R == 0 so far
    assert without_downtime == pytest.approx(10.0)

    s.record_recovery(4.0)
    with_downtime = s.mtbf(30.0)         # each 10s gap is really 6s of training
    assert with_downtime == pytest.approx(6.0)

    # ...and that shortens the interval, even though R is not a term in T*.
    assert s.optimal_interval_seconds(30.0) < math.sqrt(2 * 0.4 * 10.0)


def test_no_failures_means_no_downtime_to_subtract():
    """Nothing has failed, so nothing has been recovered from."""
    s = _primed(d=0.4, started_at=0.0)
    s.record_recovery(4.0)               # e.g. inherited from a previous job
    assert s.failures_observed == 0
    assert s.mtbf(60.0) == pytest.approx(60.0)


def test_downtime_subtraction_cannot_drive_mtbf_negative():
    """A recovery longer than the gap must not produce a negative hazard."""
    s = _primed(d=0.4, mtbf_floor=0.5)
    s.observe_failures([0.0, 2.0, 4.0])
    s.record_recovery(30.0)              # absurdly slow restarts
    assert s.mtbf(5.0) == pytest.approx(0.5)
    assert s.optimal_interval_seconds(5.0) > 0


# --------------------------------------------------------------------------
# The two-scenario proof's own invariant
# --------------------------------------------------------------------------

def test_scenario_comparison_rejects_configs_differing_beyond_failure_rate():
    """
    The whole proof rests on the two scenarios being otherwise identical. If
    they differed in, say, --checkpoint-every, an interval difference would
    prove nothing, so the comparison refuses to draw a conclusion from it.
    """
    from checkpoint.verify_scheduler import (
        SchedulerCheckError,
        assert_only_failure_config_differs,
    )

    base = ["--epochs", "8", "--checkpoint-every", "4",
            "--failure-rate", "0.01", "--failure-seed", "32"]
    ok_high = ["--epochs", "8", "--checkpoint-every", "4",
               "--failure-rate", "0.25", "--failure-seed", "2"]

    # Legitimate: only the failure config differs.
    assert_only_failure_config_differs({"worker_flags": base},
                                       {"worker_flags": ok_high})

    # Illegitimate: the interval itself was changed between scenarios.
    rigged = ["--epochs", "8", "--checkpoint-every", "40",
              "--failure-rate", "0.25", "--failure-seed", "2"]
    with pytest.raises(SchedulerCheckError, match="more than the failure config"):
        assert_only_failure_config_differs({"worker_flags": base},
                                           {"worker_flags": rigged})


def test_scheduler_from_job_inherits_checkpoint_cost_and_throughput(tmp_path):
    """
    d and throughput are per-PROCESS measurements, and the process that made
    them is the one that died. If they aren't rebuilt from the job's logs,
    decide() has no choice but to return the launch-time default, so every
    restarted generation checkpoints at that default until it has written
    twice — and under a high failure rate generations are too short for the
    learned interval to ever apply. That silently turns "adaptive" into
    "fixed default", which is exactly what the Phase 6 benchmark caught.
    """
    job = tmp_path / "job"
    job.mkdir()
    (job / "checkpoint_log.jsonl").write_text("\n".join(
        json.dumps({"event": "checkpoint", "duration_seconds": 0.4,
                    "global_step": s}) for s in (2, 4, 6)) + "\n")
    (job / "loss_log.jsonl").write_text("\n".join(
        json.dumps({"global_step": s, "step_seconds": 1.0})
        for s in range(1, 7)) + "\n")
    (job / "failure_log.jsonl").write_text(
        json.dumps({"event": "injected_failure", "timestamp": 100.0}) + "\n"
        + json.dumps({"event": "injected_failure", "timestamp": 106.0}) + "\n")

    s = scheduler_from_job(str(job), initial_interval_steps=8)

    assert s.checkpoint_cost() == pytest.approx(0.4)
    assert s.throughput() == pytest.approx(1.0)
    # With d and throughput known, the opening decision is computed, not the
    # launch default.
    decision = s.decide(112.0)
    assert "sqrt" in decision.reason
    assert decision.interval_steps != 8


def test_scheduler_from_job_without_measurement_logs_falls_back(tmp_path):
    """No logs yet (a genuinely fresh job) must still work."""
    job = tmp_path / "job"
    job.mkdir()
    s = scheduler_from_job(str(job), initial_interval_steps=8)
    assert s.decide(10.0).interval_steps == 8


def test_measurement_log_readers_are_windowed_and_truncation_safe(tmp_path):
    from checkpoint.scheduler import (
        checkpoint_costs_from_log,
        step_durations_from_log,
    )

    ck = tmp_path / "checkpoint_log.jsonl"
    ck.write_text("\n".join(
        json.dumps({"event": "checkpoint", "duration_seconds": float(i)})
        for i in range(1, 11))
        + "\n" + json.dumps({"event": "fast_forward", "duration_seconds": 99.0})
        + "\n" + '{"event": "checkpoint", "duration_sec')
    costs = checkpoint_costs_from_log(str(ck), limit=3)
    assert costs == [8.0, 9.0, 10.0], "fast_forward must not count as a write"

    ll = tmp_path / "loss_log.jsonl"
    ll.write_text("\n".join(
        json.dumps({"global_step": i, "step_seconds": float(i)})
        for i in range(1, 11)) + "\n")
    assert step_durations_from_log(str(ll), limit=2) == [9.0, 10.0]
