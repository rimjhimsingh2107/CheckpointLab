"""
Tests for Phase 4 automatic failure detection and recovery.

The claim being tested is that a job survives worker death with no human in
the loop. Three things have to hold for that:

  1. a dead worker is *noticed* — including the awkward case where the
     process is alive but wedged;
  2. the checkpoint recovery picks up from is actually loadable;
  3. per-job state is isolated, so concurrent jobs can't corrupt each
     other's view of what already failed.

Fast tier covers all three directly. The `slow` test runs the whole
unattended loop against a real injected failure.
"""
import json
import os
import signal
import subprocess
import sys
import time

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkpoint.manager import FORMAT_VERSION, list_checkpoints, save_checkpoint
from controller import recovery as recovery_mod
from controller.recovery import (
    RecoveryController,
    find_latest_valid_checkpoint,
    new_job_dir,
    validate_checkpoint,
)
from controller.worker_manager import (
    WorkerManager,
    clear_heartbeats,
    read_heartbeat,
    write_heartbeat,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tiny_state(tmp_dir: str, steps=(2, 4, 6)) -> None:
    """Write a few real checkpoints via the Phase 2 manager."""
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    for step in steps:
        loss = torch.nn.functional.mse_loss(model(torch.randn(2, 4)),
                                            torch.randn(2, 2))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        save_checkpoint(model, optimizer, epoch=0, step_in_epoch=step,
                        global_step=step, rank=0, world_size=1,
                        checkpoint_dir=tmp_dir, seed=1337)


# --------------------------------------------------------------------------
# Per-job isolation — the Phase 3 sharp edge this phase had to close
# --------------------------------------------------------------------------

def test_new_job_dir_is_unique_per_call():
    """
    Phase 3 defaulted the failure log to one shared path with a run id
    derived from rate+seed, so two jobs with the same rate and seed read each
    other's already-fired failures and suppressed them. Job identity has to
    be unique by construction, not by convention.
    """
    dirs = {new_job_dir() for _ in range(50)}
    assert len(dirs) == 50


def test_controller_scopes_all_worker_state_under_the_job_dir(tmp_path):
    job_dir = str(tmp_path / "job")
    controller = RecoveryController(job_dir=job_dir, world_size=2,
                                    train_flags=["--max-steps", "1"])

    assert controller.checkpoint_dir.startswith(job_dir)
    # --job-dir is prepended to the worker command line, which is what makes
    # the failure log, run id and heartbeat dir land inside this job.
    assert controller.manager.train_flags[:2] == ["--job-dir", job_dir]
    assert controller.manager.heartbeat_dir.startswith(job_dir)


def test_two_jobs_with_identical_failure_config_do_not_share_a_failure_log(tmp_path):
    """
    Same rate, same seed, different jobs: the exact collision Phase 3 warned
    about. Their failure logs must be different files.
    """
    flags = ["--failure-rate", "0.05", "--failure-seed", "42"]
    a = RecoveryController(job_dir=str(tmp_path / "a"), world_size=2,
                           train_flags=list(flags))
    b = RecoveryController(job_dir=str(tmp_path / "b"), world_size=2,
                           train_flags=list(flags))

    assert a.checkpoint_dir != b.checkpoint_dir
    assert a.manager.heartbeat_dir != b.manager.heartbeat_dir

    # And the derivation the workers themselves perform must agree.
    sys.path.insert(0, os.path.join(REPO_ROOT, "training"))
    from training.train import apply_job_dir, parse_args

    resolved = []
    for job in ("a", "b"):
        args = apply_job_dir(parse_args(
            ["--job-dir", str(tmp_path / job)] + flags))
        resolved.append((args.failure_log, args.run_id))

    assert resolved[0][0] != resolved[1][0], "failure logs collide"
    assert resolved[0][1] != resolved[1][1], "run ids collide"


def test_explicit_flags_win_over_job_dir_derivation(tmp_path):
    from training.train import apply_job_dir, parse_args

    explicit = str(tmp_path / "somewhere_else.jsonl")
    args = apply_job_dir(parse_args(
        ["--job-dir", str(tmp_path / "job"), "--failure-log", explicit]))
    assert args.failure_log == explicit


# --------------------------------------------------------------------------
# Checkpoint validation
# --------------------------------------------------------------------------

def test_validate_checkpoint_accepts_a_real_one(tmp_path):
    ckpt_dir = str(tmp_path / "ckpts")
    _tiny_state(ckpt_dir, steps=(2,))

    meta = validate_checkpoint(list_checkpoints(ckpt_dir)[0])
    assert meta is not None
    assert meta["global_step"] == 2


def test_validate_checkpoint_rejects_a_truncated_file(tmp_path):
    """A worker killed mid-write is precisely what Phase 3 injects."""
    ckpt_dir = str(tmp_path / "ckpts")
    _tiny_state(ckpt_dir, steps=(2,))
    path = list_checkpoints(ckpt_dir)[0]

    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) // 2)

    assert validate_checkpoint(path) is None


def test_validate_checkpoint_rejects_wrong_format_version(tmp_path):
    path = str(tmp_path / "ckpt_step00000001.pt")
    torch.save({"format_version": FORMAT_VERSION + 99, "model": {},
                "optimizer": {}, "epoch": 0, "step_in_epoch": 0,
                "global_step": 1}, path)
    assert validate_checkpoint(path) is None


def test_validate_checkpoint_rejects_missing_keys(tmp_path):
    path = str(tmp_path / "ckpt_step00000001.pt")
    torch.save({"format_version": FORMAT_VERSION, "model": {}}, path)
    assert validate_checkpoint(path) is None


def test_recovery_falls_back_past_a_corrupt_newest_checkpoint(tmp_path):
    """
    The headline behaviour: don't give up because the newest checkpoint is
    unreadable, walk back to one that loads.
    """
    ckpt_dir = str(tmp_path / "ckpts")
    _tiny_state(ckpt_dir, steps=(2, 4, 6))

    newest = list_checkpoints(ckpt_dir)[-1]
    with open(newest, "r+b") as fh:
        fh.truncate(1024)

    meta, rejected = find_latest_valid_checkpoint(ckpt_dir)
    assert meta is not None
    assert meta["global_step"] == 4, "should fall back to the previous one"
    assert rejected == [newest]


def test_find_latest_valid_checkpoint_on_empty_dir(tmp_path):
    empty = str(tmp_path / "nothing")
    os.makedirs(empty)
    meta, rejected = find_latest_valid_checkpoint(empty)
    assert meta is None and rejected == []


def test_find_latest_valid_checkpoint_when_all_are_corrupt(tmp_path):
    ckpt_dir = str(tmp_path / "ckpts")
    _tiny_state(ckpt_dir, steps=(2, 4))
    for path in list_checkpoints(ckpt_dir):
        with open(path, "r+b") as fh:
            fh.truncate(512)

    meta, rejected = find_latest_valid_checkpoint(ckpt_dir)
    assert meta is None
    assert len(rejected) == 2


# --------------------------------------------------------------------------
# Heartbeats
# --------------------------------------------------------------------------

def test_heartbeat_round_trip(tmp_path):
    hb = str(tmp_path / "hb")
    write_heartbeat(hb, 1, generation=3, global_step=17, state="training")

    beat = read_heartbeat(hb, 1)
    assert beat["rank"] == 1
    assert beat["generation"] == 3
    assert beat["global_step"] == 17
    assert beat["state"] == "training"
    assert beat["pid"] == os.getpid()
    assert beat["timestamp"] > 0


def test_heartbeat_write_leaves_no_partial_file(tmp_path):
    """
    Written tmp-then-rename: the controller reading a torn heartbeat would
    look like a missing one and trigger a spurious recovery.
    """
    hb = str(tmp_path / "hb")
    for step in range(20):
        write_heartbeat(hb, 0, generation=1, global_step=step)
        assert read_heartbeat(hb, 0) is not None
    assert not [f for f in os.listdir(hb) if f.endswith(".tmp")]


def test_read_missing_heartbeat_is_none(tmp_path):
    assert read_heartbeat(str(tmp_path / "hb"), 0) is None


def test_manager_ignores_heartbeats_from_an_older_generation(tmp_path):
    """
    A stale heartbeat from a dead generation must not make a freshly
    launched worker look alive — otherwise the controller would report a
    successful resume for a job that never restarted.
    """
    job_dir = str(tmp_path / "job")
    manager = WorkerManager(job_dir=job_dir, world_size=2, train_flags=[])
    manager.generation = 2

    write_heartbeat(manager.heartbeat_dir, 0, generation=1, global_step=5)
    assert manager.heartbeats() == {}

    write_heartbeat(manager.heartbeat_dir, 0, generation=2, global_step=9)
    assert set(manager.heartbeats()) == {0}
    assert manager.max_step_seen() == 9


def test_clear_heartbeats_removes_them(tmp_path):
    hb = str(tmp_path / "hb")
    write_heartbeat(hb, 0, generation=1)
    write_heartbeat(hb, 1, generation=1)
    clear_heartbeats(hb)
    assert read_heartbeat(hb, 0) is None
    assert read_heartbeat(hb, 1) is None


# --------------------------------------------------------------------------
# Hang detection — the case a process-exit check cannot catch
# --------------------------------------------------------------------------

def test_wedged_worker_is_detected_via_stale_heartbeat(tmp_path):
    """
    When one rank dies, the survivors don't die — they block in a collective
    waiting for a peer that isn't coming back, and gloo's default timeout is
    30 minutes. A liveness check based only on process exit would call that
    healthy. SIGSTOP reproduces it exactly: alive, scheduled, silent.
    """
    job_dir = str(tmp_path / "job")
    manager = WorkerManager(
        job_dir=job_dir, world_size=1,
        # A worker that heartbeats a few times then stops on its own signal.
        train_flags=[], heartbeat_timeout=1.5)

    script = (
        "import sys, time, os;"
        "sys.path.insert(0, %r);"
        "from controller.worker_manager import write_heartbeat;"
        "hb = %r;"
        "[ (write_heartbeat(hb, 0, generation=1, global_step=i,"
        "   state='training'), time.sleep(0.1)) for i in range(3) ];"
        "os.kill(os.getpid(), %d); time.sleep(60)"
        % (REPO_ROOT, manager.heartbeat_dir, int(signal.SIGSTOP))
    )
    proc = subprocess.Popen([sys.executable, "-c", script], cwd=REPO_ROOT,
                            start_new_session=True)
    try:
        from controller.worker_manager import Worker
        manager.generation = 1
        manager.workers = [Worker(rank=0, process=proc, log_path=os.devnull,
                                  launched_at=time.time(), port=0)]

        # Heartbeating: healthy.
        deadline = time.time() + 10
        while time.time() < deadline and not manager.heartbeats():
            time.sleep(0.05)
        assert manager.heartbeats(), "worker never heartbeat at all"
        assert manager.poll() is None, "healthy worker reported as failed"

        # Now wedged: process alive, heartbeat frozen.
        deadline = time.time() + 15
        failure = None
        while time.time() < deadline and failure is None:
            failure = manager.poll()
            time.sleep(0.1)

        assert failure is not None, "wedged worker was never detected"
        assert failure.kind == "stale_heartbeat"
        assert failure.rank == 0
        assert proc.poll() is None, "process should still be alive (SIGSTOP)"
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=10)


# --------------------------------------------------------------------------
# Slow tier — the unattended loop
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_unattended_recovery_completes_a_job_through_an_injected_failure(tmp_path):
    """
    The Phase 4 milestone: inject a failure, and the job finishes anyway with
    nobody watching.
    """
    args = recovery_mod.parse_args([
        "--failure-rate", "0.05", "--failure-seed", "42",
        "--world-size", "2", "--max-steps", "40", "--checkpoint-every", "5",
        "--job-dir", str(tmp_path / "job"), "--quiet",
    ])
    controller = RecoveryController(
        job_dir=args.job_dir, world_size=args.world_size,
        train_flags=recovery_mod.train_flags_from(args),
        max_restarts=args.max_restarts, quiet=True)
    summary = controller.run()

    assert summary["outcome"] == "completed"
    assert summary["recoveries"] >= 1, "no failure fired; test is vacuous"
    assert summary["generations"] == summary["recoveries"] + 1
    assert all(code == 0 for code in summary["exit_codes"].values())

    detail = summary["recovery_details"][0]
    assert detail["failure_kind"] == "exit"
    assert detail["exit_code"] == 137          # the injector's SIGKILL-alike
    assert detail["restored_to_step"] < detail["step_at_failure"]
    assert detail["recovery_seconds"] > 0

    # The controller's own audit trail has to show the full cycle.
    events = [json.loads(line) for line in
              open(os.path.join(args.job_dir, "recovery_log.jsonl"))]
    kinds = [e["event"] for e in events]
    for expected in ("job_started", "generation_launched", "failure_detected",
                     "generation_terminated", "checkpoint_selected",
                     "training_resumed", "job_finished"):
        assert expected in kinds, f"missing {expected} in {kinds}"
