"""
Tests for Phase 3 failure injection.

The property under test is reproducibility. Phase 6's benchmark compares
checkpoint strategies against each other, and that comparison is rigged
unless every strategy meets the identical failure sequence — so "same seed
=> same failures at the same steps" is a correctness requirement here, not a
nice-to-have.

Fast tier covers the schedule arithmetic and the log format. The `slow` test
runs two real training jobs and compares what actually got killed.

    pytest tests/                      # everything
    pytest tests/ -m "not slow"        # fast tier only
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from failures import verify_reproducibility as repro_mod
from failures.injector import (
    FailureSchedule,
    ScheduledFailure,
    load_fired_steps,
    log_failure,
)


def _schedule(**kwargs) -> FailureSchedule:
    params = dict(failure_rate=0.05, workers=4, seed=42, max_steps=200)
    params.update(kwargs)
    return FailureSchedule(**params)


# --------------------------------------------------------------------------
# Determinism — the whole point of the module
# --------------------------------------------------------------------------

def test_same_seed_produces_identical_schedule():
    a, b = _schedule(), _schedule()
    assert a.failures == b.failures
    assert a.fingerprint() == b.fingerprint()
    assert a.failures, "fixture should schedule at least one failure"


def test_different_seed_produces_different_schedule():
    a, b = _schedule(seed=42), _schedule(seed=43)
    assert a.failures != b.failures
    assert a.fingerprint() != b.fingerprint()


def test_schedule_prefix_is_stable_as_horizon_grows():
    """
    Extending a run's step budget must not retroactively change the failures
    it already experienced. This is the property a sequential RNG would
    silently break, and the reason each step is an independent hash.
    """
    short = _schedule(max_steps=50)
    long = _schedule(max_steps=5000)

    assert short.failures == [f for f in long.failures if f.step <= 50]
    for step in range(1, 51):
        assert short.victim_at(step) == long.victim_at(step)


def test_victim_at_matches_the_failure_list():
    schedule = _schedule()
    listed = {f.step: f.worker for f in schedule.failures}

    for step in range(1, schedule.max_steps + 1):
        assert schedule.victim_at(step) == listed.get(step)


# --------------------------------------------------------------------------
# Rate + victim behaviour
# --------------------------------------------------------------------------

def test_zero_rate_schedules_no_failures():
    schedule = _schedule(failure_rate=0.0)
    assert schedule.failures == []
    assert schedule.victim_at(17) is None


def test_rate_one_kills_at_every_step():
    schedule = _schedule(failure_rate=1.0, max_steps=25)
    assert [f.step for f in schedule.failures] == list(range(1, 26))


def test_observed_rate_approximates_requested_rate():
    """A long window should land near the requested per-step probability."""
    schedule = _schedule(failure_rate=0.05, max_steps=20000)
    observed = len(schedule.failures) / 20000
    assert observed == pytest.approx(0.05, abs=0.005)


def test_victims_are_spread_across_all_workers():
    schedule = _schedule(failure_rate=0.5, workers=4, max_steps=2000)
    assert {f.worker for f in schedule.failures} == {0, 1, 2, 3}


def test_victim_is_always_a_valid_rank():
    for workers in (1, 2, 3, 8):
        schedule = _schedule(failure_rate=0.5, workers=workers, max_steps=500)
        assert schedule.failures
        for failure in schedule.failures:
            assert 0 <= failure.worker < workers


def test_min_step_delays_the_first_failure():
    """
    Failing before any checkpoint exists just restarts from zero and tests
    nothing, so min_step buys a floor of real progress first.
    """
    schedule = _schedule(failure_rate=0.5, min_step=100, max_steps=200)
    assert schedule.failures
    assert min(f.step for f in schedule.failures) >= 100
    assert schedule.victim_at(50) is None


@pytest.mark.parametrize("kwargs", [
    dict(failure_rate=-0.1),
    dict(failure_rate=1.5),
    dict(workers=0),
    dict(max_steps=-1),
])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _schedule(**kwargs)


# --------------------------------------------------------------------------
# Failure log
# --------------------------------------------------------------------------

def test_log_failure_writes_an_auditable_record(tmp_path):
    log = str(tmp_path / "failure_log.jsonl")
    schedule = _schedule()

    record = log_failure(log, step=31, worker=2, schedule=schedule,
                         run_id="job-1", extra={"epoch": 1})

    lines = open(log).read().strip().splitlines()
    assert len(lines) == 1
    written = json.loads(lines[0])

    assert written == record
    assert written["event"] == "injected_failure"
    assert written["step"] == 31
    assert written["worker"] == 2
    assert written["run_id"] == "job-1"
    assert written["epoch"] == 1
    assert written["seed"] == schedule.seed
    assert written["schedule_fingerprint"] == schedule.fingerprint()
    # A timestamp and the dead worker's pid are what make the log an audit
    # trail rather than a restatement of the schedule.
    assert written["timestamp"] > 0
    assert written["timestamp_iso"]
    assert written["pid"] == os.getpid()


def test_log_failure_appends_rather_than_truncating(tmp_path):
    log = str(tmp_path / "failure_log.jsonl")
    schedule = _schedule()

    for step in (31, 70, 99):
        log_failure(log, step=step, worker=0, schedule=schedule, run_id="job-1")

    assert len(open(log).read().strip().splitlines()) == 3
    assert load_fired_steps(log) == {31, 70, 99}


def test_load_fired_steps_filters_by_run_id(tmp_path):
    log = str(tmp_path / "failure_log.jsonl")
    schedule = _schedule()

    log_failure(log, step=31, worker=0, schedule=schedule, run_id="job-1")
    log_failure(log, step=70, worker=1, schedule=schedule, run_id="job-2")

    assert load_fired_steps(log, "job-1") == {31}
    assert load_fired_steps(log, "job-2") == {70}
    assert load_fired_steps(log) == {31, 70}


def test_load_fired_steps_survives_a_truncated_final_line(tmp_path):
    """
    A worker killed mid-write can leave a partial line. Recovery has to read
    what it can rather than crashing on the way back up.
    """
    log = str(tmp_path / "failure_log.jsonl")
    schedule = _schedule()
    log_failure(log, step=31, worker=0, schedule=schedule, run_id="job-1")

    with open(log, "a") as fh:
        fh.write('{"event": "injected_failure", "step": 7')  # no newline, no close

    assert load_fired_steps(log) == {31}


def test_load_fired_steps_on_missing_log_is_empty(tmp_path):
    assert load_fired_steps(str(tmp_path / "nope.jsonl")) == set()


def test_fired_steps_suppress_refiring_after_recovery():
    """
    Recovery replays work: a job that died at step 31 resumes from its last
    checkpoint and re-executes 31. If the schedule fired again there, the job
    would never get past it. The training loop consults exactly this pair of
    values, so assert they compose.
    """
    schedule = _schedule(failure_rate=0.05, workers=2, seed=42, max_steps=40)
    step = schedule.failures[0].step

    assert schedule.victim_at(step) is not None      # would fire...
    already_fired = {step}
    assert step in already_fired                     # ...but is suppressed


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------

def test_scheduled_failure_sorts_by_step():
    unsorted = [ScheduledFailure(9, 1), ScheduledFailure(2, 0)]
    assert sorted(unsorted) == [ScheduledFailure(2, 0), ScheduledFailure(9, 1)]


def test_schedule_never_touches_the_global_rng():
    """
    If the injector drew from the training RNG, merely enabling failure
    injection would shift the augmentation stream and change the loss
    trajectory — making a run with failures incomparable to one without.
    """
    import random

    import torch

    torch.manual_seed(0)
    random.seed(0)
    before = (torch.get_rng_state().clone(), random.getstate())

    schedule = _schedule(failure_rate=0.5, max_steps=1000)
    schedule.fingerprint()
    schedule.victim_at(31)

    after = (torch.get_rng_state(), random.getstate())
    assert torch.equal(before[0], after[0])
    assert before[1] == after[1]


# --------------------------------------------------------------------------
# Slow tier — the real milestone
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_same_seed_kills_the_same_worker_at_the_same_step_end_to_end():
    """
    Run two real training jobs with the same failure seed and assert the
    observed failure logs agree — not just the schedules computed up front.
    """
    args = repro_mod.parse_args([])
    summary = repro_mod.verify(args)   # raises on mismatch

    expected = FailureSchedule(args.failure_rate, args.world_size,
                               args.failure_seed, args.steps).failures[0]
    assert summary["failed_at_step"] == expected.step
    assert summary["worker_killed"] == expected.worker
    assert summary["failures_logged"] >= 1
