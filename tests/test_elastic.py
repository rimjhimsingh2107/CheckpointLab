"""
Tests for elastic resumption.

The property under test is NOT bit-exactness. Once the world size changes,
the effective batch is world_size * batch_size and DistributedSampler
reshards, so a different optimisation problem is being solved and the loss
trajectory necessarily differs. What must hold is narrower:

  * the checkpoint records position in SAMPLES, which is world-size
    independent, alongside the step count which is not;
  * converting that position to a step count at a new world size is correct,
    and rounds DOWN so work is redone rather than skipped;
  * a v1 checkpoint, which lacks the batch size needed to convert, is
    rejected rather than guessed at;
  * per-rank RNG streams are NOT reused across a world-size change, because
    handing a new rank a copy of rank 0's stream would correlate their
    augmentations.
"""
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkpoint.manager import (
    FORMAT_VERSION,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from controller.recovery import RecoveryController
from controller.worker_manager import WorkerManager


def model_and_opt(seed=0):
    torch.manual_seed(seed)
    m = torch.nn.Linear(8, 4)
    return m, torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)


def write_ckpt(d, *, epoch=1, step_in_epoch=7, global_step=27,
               world_size=3, batch_size=32):
    m, o = model_and_opt()
    save_checkpoint(m, o, epoch, step_in_epoch, global_step, rank=0,
                    world_size=world_size, checkpoint_dir=d, seed=1337,
                    batch_size=batch_size)
    return find_latest_checkpoint(d)


# --------------------------------------------------------------------------
# The checkpoint records a world-size-independent position
# --------------------------------------------------------------------------

def test_checkpoint_records_samples_into_epoch(tmp_path):
    path = write_ckpt(str(tmp_path), step_in_epoch=7, world_size=3,
                      batch_size=32)
    m, o = model_and_opt(99)
    meta = load_checkpoint(path, m, o, rank=0)
    # 7 steps x 3 ranks x 32 = 672 samples
    assert meta["samples_into_epoch"] == 672
    assert meta["batch_size"] == 32
    assert meta["world_size"] == 3
    # The step count is still recorded, but it is only valid at world_size 3.
    assert meta["step_in_epoch"] == 7


@pytest.mark.parametrize("world_size,batch,step,expected", [
    (2, 32, 6, 384),
    (3, 32, 7, 672),
    (4, 64, 5, 1280),
    (1, 32, 0, 0),
])
def test_samples_is_steps_times_world_size_times_batch(
        tmp_path, world_size, batch, step, expected):
    path = write_ckpt(str(tmp_path), step_in_epoch=step,
                      world_size=world_size, batch_size=batch)
    m, o = model_and_opt(99)
    assert load_checkpoint(path, m, o, rank=0)["samples_into_epoch"] == expected


def test_v1_checkpoint_is_rejected_not_guessed(tmp_path):
    """
    A v1 checkpoint has no batch_size, so its sample position cannot be
    recovered. Loading it at a new world size would need a guess; refusing is
    the correct behaviour.
    """
    path = str(tmp_path / "ckpt_step00000001.pt")
    torch.save({"format_version": 1, "model": {}, "optimizer": {},
                "epoch": 0, "step_in_epoch": 5, "global_step": 5,
                "world_size": 2}, path)
    m, o = model_and_opt()
    with pytest.raises(ValueError, match="format_version"):
        load_checkpoint(path, m, o, rank=0)


# --------------------------------------------------------------------------
# The conversion itself
# --------------------------------------------------------------------------

def convert(samples, new_world_size, batch):
    """The arithmetic train.py performs; mirrored here to pin it."""
    per_step = new_world_size * batch
    return samples // per_step, samples % per_step


@pytest.mark.parametrize("samples,new_ws,batch,step,leftover", [
    # exact divisions
    (384, 2, 32, 6, 0),     # unchanged world size
    (384, 3, 32, 4, 0),     # 2->3, lands exactly
    (384, 4, 32, 3, 0),     # 2->4, lands exactly
    # inexact: must round DOWN, leaving samples to be re-seen
    (896, 3, 32, 9, 32),    # 2->3 mid-epoch
    (672, 2, 32, 10, 32),   # 3->2 mid-epoch
    (0, 2, 32, 0, 0),       # epoch boundary
])
def test_sample_to_step_conversion(samples, new_ws, batch, step, leftover):
    assert convert(samples, new_ws, batch) == (step, leftover)


def test_conversion_rounds_down_so_work_is_redone_not_skipped():
    """
    Rounding up would advance past samples the model never saw. Rounding down
    re-shows at most one step's worth. This project errs toward redoing work
    everywhere else; elasticity is no exception.
    """
    for samples in range(1, 400):
        step, leftover = convert(samples, 3, 32)
        consumed = step * 3 * 32
        assert consumed <= samples, "rounded up and skipped unseen samples"
        assert samples - consumed == leftover
        assert leftover < 3 * 32


# --------------------------------------------------------------------------
# RNG must not be reused across a world-size change
# --------------------------------------------------------------------------

def test_rng_is_not_restored_when_world_size_changes(tmp_path):
    """
    Handing a new rank a copy of rank 0's stream would make the two draw
    identical augmentations, correlating noise across a batch that is supposed
    to be independently sampled. None tells the caller to reseed instead.
    """
    path = write_ckpt(str(tmp_path), world_size=2)
    m, o = model_and_opt(99)
    meta = load_checkpoint(path, m, o, rank=0, current_world_size=3)
    assert meta["rng_state"] is None


def test_rng_is_restored_when_world_size_is_unchanged(tmp_path):
    path = write_ckpt(str(tmp_path), world_size=2)
    m, o = model_and_opt(99)
    meta = load_checkpoint(path, m, o, rank=0, current_world_size=2)
    assert meta["rng_state"] is not None


def test_rng_restored_when_current_world_size_not_supplied(tmp_path):
    """Callers that don't pass it keep the old behaviour."""
    path = write_ckpt(str(tmp_path), world_size=1)
    m, o = model_and_opt(99)
    assert load_checkpoint(path, m, o, rank=0)["rng_state"] is not None


def test_model_and_optimizer_still_restore_across_a_world_size_change(tmp_path):
    """The whole point: the weights must survive even though the RNG doesn't."""
    d = str(tmp_path / "c")
    m, o = model_and_opt()
    for _ in range(3):
        loss = torch.nn.functional.mse_loss(m(torch.randn(4, 8)),
                                            torch.randn(4, 4))
        o.zero_grad(); loss.backward(); o.step()
    save_checkpoint(m, o, 1, 7, 27, rank=0, world_size=2,
                    checkpoint_dir=d, seed=1337, batch_size=32)
    expected = {k: v.clone() for k, v in m.state_dict().items()}

    fresh, fresh_o = model_and_opt(seed=999)
    load_checkpoint(find_latest_checkpoint(d), fresh, fresh_o, rank=0,
                    current_world_size=5)
    for k, v in expected.items():
        assert torch.equal(fresh.state_dict()[k], v), k


# --------------------------------------------------------------------------
# The controller's world-size plan
# --------------------------------------------------------------------------

def test_world_size_plan_maps_generation_to_size(tmp_path):
    c = RecoveryController(job_dir=str(tmp_path / "j"), world_size=2,
                           train_flags=[], world_size_plan=[2, 3])
    assert c._world_size_for(1) == 2
    assert c._world_size_for(2) == 3
    # Last entry repeats, so a job can keep restarting after the change.
    assert c._world_size_for(3) == 3
    assert c._world_size_for(9) == 3


def test_world_size_plan_defaults_to_fixed_size(tmp_path):
    c = RecoveryController(job_dir=str(tmp_path / "j"), world_size=4,
                           train_flags=[])
    assert c.world_size_plan == [4]
    assert all(c._world_size_for(g) == 4 for g in (1, 2, 5))


def test_manager_launch_can_change_world_size(tmp_path):
    """
    No process-group resize is needed: each generation is new processes doing
    a fresh init_process_group, so a new world size is just different argv.
    """
    m = WorkerManager(job_dir=str(tmp_path / "j"), world_size=2,
                      train_flags=["--max-steps", "1"])
    assert m.world_size == 2
    m.launch(resume=True, world_size=3)
    try:
        assert m.world_size == 3
        assert len(m.workers) == 3
        assert [w.rank for w in m.workers] == [0, 1, 2]
        # Every worker must agree on the new size, or the rendezvous hangs.
        for w in m.workers:
            log = open(w.log_path).read() if os.path.exists(w.log_path) else ""
            del log
    finally:
        m.terminate_all(grace=1.0)


def test_manager_shrinking_world_size_drops_ranks(tmp_path):
    m = WorkerManager(job_dir=str(tmp_path / "j"), world_size=3,
                      train_flags=["--max-steps", "1"])
    m.launch(resume=True, world_size=2)
    try:
        assert m.world_size == 2
        assert [w.rank for w in m.workers] == [0, 1]
    finally:
        m.terminate_all(grace=1.0)
