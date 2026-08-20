"""
Tests for Phase 2 checkpointing.

Two tiers, because they answer different questions and cost wildly
different amounts of time:

  * Unit tests on `checkpoint/manager.py` — fast (< 1s each). Do save/load
    round-trips actually restore weights, optimizer momentum, and RNG state?
    These catch a broken manager immediately.

  * `test_resumed_run_matches_uninterrupted_run` — the real thing. Spawns
    three DDP training subprocesses (control, crash, resume) and asserts the
    resumed loss trajectory matches the uninterrupted one. Marked `slow`
    since it takes ~1 min on CPU; CI can run the fast tier on every push and
    this one on merges.

    pytest tests/                      # everything
    pytest tests/ -m "not slow"        # fast tier only
"""
import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkpoint import verify as verify_mod
from checkpoint.manager import (
    capture_rng_state,
    find_latest_checkpoint,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)


def _tiny_model_and_optimizer(seed: int = 0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(),
                               torch.nn.Linear(16, 4))
    # Momentum matters here: it gives the optimizer state something to lose.
    # A checkpoint that only saves weights passes a weights-only test and
    # then quietly changes the update direction on the first resumed step.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    return model, optimizer


def _take_steps(model, optimizer, n: int, seed: int = 123):
    torch.manual_seed(seed)
    losses = []
    for _ in range(n):
        x = torch.randn(4, 8)
        y = torch.randn(4, 4)
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


# --------------------------------------------------------------------------
# Fast tier — manager unit tests
# --------------------------------------------------------------------------

def test_rng_state_round_trip():
    """Restoring RNG state must reproduce the exact same draw sequence."""
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)

    state = capture_rng_state()
    first = (torch.randn(3).tolist(), np.random.rand(3).tolist(),
             [random.random() for _ in range(3)])

    # Burn the RNGs so a no-op "restore" cannot accidentally pass.
    torch.randn(50)
    np.random.rand(50)
    [random.random() for _ in range(50)]

    restore_rng_state(state)
    second = (torch.randn(3).tolist(), np.random.rand(3).tolist(),
              [random.random() for _ in range(3)])

    assert first == second


def test_save_load_round_trip_restores_weights_and_optimizer(tmp_path):
    ckpt_dir = str(tmp_path / "ckpts")

    model, optimizer = _tiny_model_and_optimizer()
    _take_steps(model, optimizer, 5)

    save_checkpoint(model, optimizer, epoch=1, step_in_epoch=5, global_step=5,
                    rank=0, world_size=1, checkpoint_dir=ckpt_dir, seed=1337)

    path = find_latest_checkpoint(ckpt_dir)
    assert path is not None, "no checkpoint was written"

    expected_weights = {k: v.clone() for k, v in model.state_dict().items()}
    expected_momentum = [
        buf["momentum_buffer"].clone()
        for buf in optimizer.state_dict()["state"].values()
    ]

    fresh_model, fresh_optimizer = _tiny_model_and_optimizer(seed=999)
    meta = load_checkpoint(path, fresh_model, fresh_optimizer, rank=0)

    assert meta["epoch"] == 1
    assert meta["step_in_epoch"] == 5
    assert meta["global_step"] == 5

    for key, expected in expected_weights.items():
        assert torch.equal(fresh_model.state_dict()[key], expected), key

    restored_momentum = [
        buf["momentum_buffer"]
        for buf in fresh_optimizer.state_dict()["state"].values()
    ]
    assert len(restored_momentum) == len(expected_momentum)
    for got, expected in zip(restored_momentum, expected_momentum):
        assert torch.equal(got, expected)


def test_resumed_optimizer_continues_identical_update_sequence(tmp_path):
    """
    End-to-end-in-miniature: save mid-training, restore into a fresh model,
    and confirm the following steps produce identical losses. This is the
    same property verify.py checks, minus DDP and real data.
    """
    ckpt_dir = str(tmp_path / "ckpts")

    model, optimizer = _tiny_model_and_optimizer()
    _take_steps(model, optimizer, 5, seed=42)

    save_checkpoint(model, optimizer, epoch=0, step_in_epoch=5, global_step=5,
                    rank=0, world_size=1, checkpoint_dir=ckpt_dir, seed=1337)
    control = _take_steps(model, optimizer, 5, seed=7)

    fresh_model, fresh_optimizer = _tiny_model_and_optimizer(seed=999)
    load_checkpoint(find_latest_checkpoint(ckpt_dir), fresh_model,
                    fresh_optimizer, rank=0)
    resumed = _take_steps(fresh_model, fresh_optimizer, 5, seed=7)

    assert control == pytest.approx(resumed, abs=1e-9)


def test_load_rejects_incompatible_format_version(tmp_path):
    """A future/legacy checkpoint must fail loudly, not load garbage."""
    path = tmp_path / "bad.pt"
    torch.save({"format_version": 999, "model": {}, "optimizer": {},
                "epoch": 0, "step_in_epoch": 0, "global_step": 0,
                "world_size": 1}, path)

    model, optimizer = _tiny_model_and_optimizer()
    with pytest.raises(ValueError, match="format_version"):
        load_checkpoint(str(path), model, optimizer, rank=0)


def test_find_latest_checkpoint_picks_highest_step(tmp_path):
    ckpt_dir = str(tmp_path / "ckpts")
    model, optimizer = _tiny_model_and_optimizer()

    for step in (5, 10, 15):
        save_checkpoint(model, optimizer, epoch=0, step_in_epoch=step,
                        global_step=step, rank=0, world_size=1,
                        checkpoint_dir=ckpt_dir, seed=1337)

    latest = find_latest_checkpoint(ckpt_dir)
    assert latest is not None
    assert load_checkpoint(latest, model, optimizer, rank=0)["global_step"] == 15

    # The manifest is a cache, not the source of truth — deleting it must
    # still leave the directory usable (a crash during a checkpoint write
    # can leave exactly this state).
    os.remove(os.path.join(ckpt_dir, "latest.json"))
    assert find_latest_checkpoint(ckpt_dir) == latest


def test_non_zero_rank_does_not_write(tmp_path):
    """Only rank 0 writes, so world_size ranks don't race on the same path."""
    ckpt_dir = str(tmp_path / "ckpts")
    model, optimizer = _tiny_model_and_optimizer()

    # world_size=1 keeps this out of the collective; rank=1 is the case
    # under test.
    save_checkpoint(model, optimizer, epoch=0, step_in_epoch=1, global_step=1,
                    rank=1, world_size=1, checkpoint_dir=ckpt_dir, seed=1337)

    assert find_latest_checkpoint(ckpt_dir) is None


# --------------------------------------------------------------------------
# Slow tier — the real correctness claim
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_resumed_run_matches_uninterrupted_run():
    """
    Kill a real DDP training process mid-run, resume it from its checkpoint,
    and assert the loss trajectory is indistinguishable from a run that was
    never interrupted.

    Wraps checkpoint/verify.py so CI and humans run the identical experiment.
    """
    args = verify_mod.parse_args([])
    summary = verify_mod.verify(args)   # raises VerificationError on mismatch

    assert summary["max_abs_diff"] <= summary["tolerance"]
    # Guard against a vacuous pass: the resume has to have actually replayed
    # work lost to the crash, and compared a non-trivial number of steps.
    assert summary["replayed_steps"] > 0
    assert summary["compared_steps"] >= summary["steps"] // 2
