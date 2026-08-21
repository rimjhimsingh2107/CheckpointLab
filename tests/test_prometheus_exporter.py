"""
Tests for the Phase 7 Prometheus exporter.

The exporter derives every metric from JSONL on disk, so these tests build
log files and assert the exposition. Two things get specific attention:

  * **The label is `ckpt_job`, not `job`.** `job` is reserved by Prometheus
    for the scrape-job name; an exporter that also exports `job` has its
    version silently renamed to `exported_job` on ingest, which makes every
    dashboard selector match nothing while the target still reads healthy.
    That cost real debugging time, so it is pinned by a test.

  * **Counters must be recomputed, not accumulated.** The workers die and
    restart; a counter held in memory would reset to zero on each generation.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prometheus_client import generate_latest

from metrics.prometheus import (
    PREFIX,
    CheckpointLabCollector,
    JobSnapshot,
    build_registry,
    discover_jobs,
)


def write(job, name, records):
    os.makedirs(job, exist_ok=True)
    with open(os.path.join(job, name), "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def make_job(root, name="j1", *, steps=(1, 2, 3), dup=(), ckpts=(0.4, 0.5),
             failures=1, recovered=1, detected=1, finished=True):
    job = os.path.join(root, name)
    loss = [{"global_step": s, "loss": 1.0, "step_seconds": 1.0} for s in steps]
    loss += [{"global_step": s, "loss": 1.0, "step_seconds": 2.0} for s in dup]
    write(job, "loss_log.jsonl", loss)
    write(job, "checkpoint_log.jsonl",
          [{"event": "checkpoint", "duration_seconds": d, "global_step": i + 1,
            "interval_used": 3} for i, d in enumerate(ckpts)]
          + [{"event": "fast_forward", "duration_seconds": 0.1,
              "steps_skipped": 2}])
    write(job, "failure_log.jsonl",
          [{"event": "injected_failure", "timestamp": 100.0 + i, "step": 5}
           for i in range(failures)])
    recov = [{"event": "job_started"}]
    for i in range(detected):
        recov.append({"event": "failure_detected"})
        recov.append({"event": "generation_launched"})
    for i in range(recovered):
        recov.append({"event": "training_resumed", "attempt": i + 1,
                      "failed_rank": 0, "failure_kind": "exit",
                      "recovery_seconds": 4.0 + i,
                      "breakdown": {"terminate_workers": 0.1,
                                    "select_checkpoint": 0.2,
                                    "relaunch_workers": 0.05,
                                    "reach_first_step": 3.65 + i},
                      "steps_lost": 1})
    if finished:
        recov.append({"event": "job_finished", "outcome": "completed",
                      "generations": detected + 1})
    write(job, "recovery_log.jsonl", recov)
    return job


def exposition(root):
    return generate_latest(build_registry(root)).decode()


def sample(text, metric, **labels):
    """Value of one sample, or None."""
    want = f"{PREFIX}_{metric}"
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(want):
            continue
        head, _, val = line.rpartition(" ")
        if all(f'{k}="{v}"' in head for k, v in labels.items()):
            name = head.split("{")[0]
            if name == want:
                return float(val)
    return None


# --------------------------------------------------------------------------
# The label-collision trap
# --------------------------------------------------------------------------

def test_label_is_ckpt_job_not_job(tmp_path):
    """
    Regression guard. Exporting `job` collides with the scrape-job label
    Prometheus attaches, and the exported one loses -- it becomes
    `exported_job`. Every panel then silently matches nothing.
    """
    make_job(str(tmp_path))
    text = exposition(str(tmp_path))
    assert 'ckpt_job="j1"' in text
    assert 'job="j1"' not in text.replace('ckpt_job="j1"', "")


def test_every_labelled_metric_uses_ckpt_job(tmp_path):
    make_job(str(tmp_path))
    for line in exposition(str(tmp_path)).splitlines():
        if line.startswith("#") or "{" not in line:
            continue
        labels = line[line.index("{") + 1:line.rindex("}")]
        keys = {kv.split("=")[0] for kv in labels.split(",") if "=" in kv}
        assert keys <= {"ckpt_job", "phase"}, f"unexpected label in: {line}"


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def test_discover_jobs_finds_dirs_with_logs(tmp_path):
    make_job(str(tmp_path), "a")
    make_job(str(tmp_path), "b")
    os.makedirs(tmp_path / "not-a-job")                      # no marker files
    (tmp_path / "loose.csv").write_text("x")                 # a file, not a dir
    found = [os.path.basename(p) for p in discover_jobs(str(tmp_path))]
    assert found == ["a", "b"]


def test_discover_jobs_on_missing_root(tmp_path):
    assert discover_jobs(str(tmp_path / "nope")) == []


def test_jobs_are_rediscovered_each_scrape(tmp_path):
    """A job created after startup must appear without restarting the exporter."""
    root = str(tmp_path)
    make_job(root, "first")
    assert sample(exposition(root), "exporter_jobs_discovered") == 1.0
    make_job(root, "second")
    assert sample(exposition(root), "exporter_jobs_discovered") == 2.0


# --------------------------------------------------------------------------
# Metric values
# --------------------------------------------------------------------------

def test_useful_and_recomputed_steps(tmp_path):
    """A repeated global_step is work a crash discarded and the job redid."""
    make_job(str(tmp_path), steps=(1, 2, 3, 4), dup=(3, 4))
    t = exposition(str(tmp_path))
    assert sample(t, "steps_completed_total", ckpt_job="j1") == 4.0
    assert sample(t, "steps_recomputed_total", ckpt_job="j1") == 2.0
    assert sample(t, "last_step", ckpt_job="j1") == 4.0


def test_checkpoint_and_fast_forward_are_separate(tmp_path):
    make_job(str(tmp_path), ckpts=(0.4, 0.6))
    t = exposition(str(tmp_path))
    assert sample(t, "checkpoints_written_total", ckpt_job="j1") == 2.0
    assert sample(t, "checkpoint_seconds_total", ckpt_job="j1") == pytest.approx(1.0)
    assert sample(t, "checkpoint_duration_seconds", ckpt_job="j1") == pytest.approx(0.6)
    assert sample(t, "fast_forward_seconds_total", ckpt_job="j1") == pytest.approx(0.1)


def test_throughput_is_inverse_mean_step_time(tmp_path):
    make_job(str(tmp_path), steps=(1, 2, 3))   # each 1.0s
    assert sample(exposition(str(tmp_path)), "throughput_steps_per_second",
                  ckpt_job="j1") == pytest.approx(1.0)


def test_active_failures_is_detected_minus_resumed(tmp_path):
    """
    Non-zero only while a recovery is in flight -- the alertable signal.
    """
    root = str(tmp_path)
    make_job(root, "mid", detected=2, recovered=1, finished=False)
    assert sample(exposition(root), "active_failures", ckpt_job="mid") == 1.0

    make_job(root, "done", detected=2, recovered=2)
    assert sample(exposition(root), "active_failures", ckpt_job="done") == 0.0


def test_active_failures_never_negative(tmp_path):
    """More resumes than detections shouldn't produce a negative gauge."""
    make_job(str(tmp_path), detected=0, recovered=2, finished=True)
    assert sample(exposition(str(tmp_path)), "active_failures", ckpt_job="j1") == 0.0


def test_job_active_tracks_the_finished_event(tmp_path):
    root = str(tmp_path)
    make_job(root, "running", finished=False)
    make_job(root, "ended", finished=True)
    t = exposition(root)
    assert sample(t, "job_active", ckpt_job="running") == 1.0
    assert sample(t, "job_active", ckpt_job="ended") == 0.0


def test_recovery_totals_and_phase_breakdown(tmp_path):
    make_job(str(tmp_path), detected=2, recovered=2)
    t = exposition(str(tmp_path))
    assert sample(t, "recoveries_total", ckpt_job="j1") == 2.0
    # recovery_seconds 4.0 and 5.0
    assert sample(t, "recovery_seconds_total", ckpt_job="j1") == pytest.approx(9.0)
    assert sample(t, "recovery_duration_seconds", ckpt_job="j1") == pytest.approx(5.0)
    # Phases come from the LAST recovery; reach_first_step = 3.65 + 1
    assert sample(t, "recovery_phase_seconds", ckpt_job="j1",
                  phase="reach_first_step") == pytest.approx(4.65)
    assert sample(t, "recovery_phase_seconds", ckpt_job="j1",
                  phase="terminate_workers") == pytest.approx(0.1)


def test_counters_are_recomputed_from_the_log_not_accumulated(tmp_path):
    """
    Scraping twice without new data must not double anything. The workers
    restart constantly; a counter accumulated in memory would be wrong in
    both directions.
    """
    root = str(tmp_path)
    make_job(root, steps=(1, 2, 3))
    first = sample(exposition(root), "steps_completed_total", ckpt_job="j1")
    second = sample(exposition(root), "steps_completed_total", ckpt_job="j1")
    assert first == second == 3.0


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------

def test_truncated_final_line_is_tolerated(tmp_path):
    """A killed worker can leave a torn line; the scrape must still succeed."""
    job = make_job(str(tmp_path), steps=(1, 2))
    with open(os.path.join(job, "loss_log.jsonl"), "a") as fh:
        fh.write('{"global_step": 3, "step_sec')
    assert sample(exposition(str(tmp_path)), "steps_completed_total",
                  ckpt_job="j1") == 2.0


def test_job_with_only_a_loss_log(tmp_path):
    """Plain `train` runs have no controller, so no recovery log exists."""
    job = os.path.join(str(tmp_path), "bare")
    write(job, "loss_log.jsonl",
          [{"global_step": 1, "loss": 1.0, "step_seconds": 0.5}])
    t = exposition(str(tmp_path))
    assert sample(t, "steps_completed_total", ckpt_job="bare") == 1.0
    assert sample(t, "recoveries_total", ckpt_job="bare") == 0.0
    # No recovery_log at all means the job never started under a controller,
    # so "active" is 0 rather than an unknown.
    assert sample(t, "job_active", ckpt_job="bare") == 0.0


def test_infinite_expected_waste_is_not_exported_as_inf(tmp_path):
    """
    The scheduler reports +Inf expected waste before anything is measured.
    Prometheus accepts +Inf but it destroys dashboard autoscaling, so it is
    coerced to 0.
    """
    job = make_job(str(tmp_path))
    write(job, "scheduler_log.jsonl",
          [{"global_step": 1, "interval_steps": 4, "optimal_seconds": 0.0,
            "mtbf": 0.5, "recovery_cost": 0.0, "throughput": 0.0,
            "expected_waste": float("inf"), "failures_observed": 0}])
    t = exposition(str(tmp_path))
    assert sample(t, "scheduler_expected_waste_ratio", ckpt_job="j1") == 0.0
    assert "+Inf" not in t


def test_scheduler_metrics_absent_run_reads_zero(tmp_path):
    """Fixed-interval runs write no scheduler log; that is not an error."""
    make_job(str(tmp_path))
    t = exposition(str(tmp_path))
    assert sample(t, "scheduler_mtbf_seconds", ckpt_job="j1") == 0.0
    assert sample(t, "scheduler_decisions", ckpt_job="j1") == 0.0


def test_empty_root_still_produces_valid_exposition(tmp_path):
    t = exposition(str(tmp_path))
    assert sample(t, "exporter_jobs_discovered") == 0.0
    assert f"{PREFIX}_steps_completed_total" in t   # HELP/TYPE still emitted


def test_exposition_parses_as_prometheus_text_format(tmp_path):
    """Malformed exposition makes Prometheus drop the whole scrape."""
    from prometheus_client.parser import text_string_to_metric_families
    make_job(str(tmp_path), "a")
    make_job(str(tmp_path), "b", detected=3, recovered=3)
    families = list(text_string_to_metric_families(exposition(str(tmp_path))))
    assert len(families) >= 20
    for f in families:
        assert f.name.startswith(PREFIX)
        assert f.documentation


def test_snapshot_name_is_the_job_dir_basename(tmp_path):
    job = make_job(str(tmp_path), "my-run")
    assert JobSnapshot(job).name == "my-run"


def test_collector_yields_only_prefixed_metrics(tmp_path):
    make_job(str(tmp_path))
    for m in CheckpointLabCollector(str(tmp_path)).collect():
        assert m.name.startswith(PREFIX), m.name
