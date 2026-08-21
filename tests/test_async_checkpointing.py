"""
Tests for asynchronous checkpointing.

The load-bearing test is `test_without_cloning_the_checkpoint_would_tear`,
which demonstrates the bug the clone exists to prevent. It is worth having as
an executable test rather than a comment because the failure mode is silent:
handing live tensor references to a background writer produces a *valid* .pt
file containing a model that never existed, and it only surfaces later as a
resumed run whose trajectory has quietly diverged.
"""
import os
import sys
import threading
import time

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkpoint.async_writer import AsyncCheckpointWriter, clone_to_cpu
from checkpoint.manager import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def model_and_opt(seed=0, width=64):
    torch.manual_seed(seed)
    m = torch.nn.Linear(width, width)
    return m, torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)


def step(m, o, width=64):
    loss = torch.nn.functional.mse_loss(m(torch.randn(4, width)),
                                       torch.randn(4, width))
    o.zero_grad(); loss.backward(); o.step()


# --------------------------------------------------------------------------
# Why the clone is mandatory
# --------------------------------------------------------------------------

def test_clone_to_cpu_deep_copies_nested_tensors():
    """
    An optimizer state_dict nests tensors under {'state': {0: {...}}}.
    Cloning only the top level leaves momentum buffers as live references --
    the subtlest form of the bug, where weights are consistent and only the
    optimizer state is torn.
    """
    t = torch.ones(3)
    payload = {"model": {"w": t}, "opt": {"state": {0: {"momentum_buffer": t}}},
               "meta": {"step": 4}, "list": [t, {"deep": t}]}
    snap = clone_to_cpu(payload)

    t.add_(99.0)   # mutate the original after cloning

    assert torch.equal(snap["model"]["w"], torch.ones(3))
    assert torch.equal(snap["opt"]["state"][0]["momentum_buffer"], torch.ones(3))
    assert torch.equal(snap["list"][0], torch.ones(3))
    assert torch.equal(snap["list"][1]["deep"], torch.ones(3))
    assert snap["meta"]["step"] == 4


def test_clone_detaches_from_autograd():
    t = torch.ones(3, requires_grad=True)
    assert clone_to_cpu({"t": t})["t"].requires_grad is False


def test_clone_preserves_container_types():
    snap = clone_to_cpu({"tup": (torch.ones(2), 3), "lst": [torch.ones(2)]})
    assert isinstance(snap["tup"], tuple)
    assert isinstance(snap["lst"], list)


def test_without_cloning_the_checkpoint_would_tear(tmp_path):
    """
    The negative control. Hand a background writer LIVE references, mutate
    the model while it serializes, and the persisted state does not match
    what was submitted. This is what save_checkpoint's clone prevents.
    """
    m, o = model_and_opt(width=256)
    step(m, o, 256)

    at_submit = {k: v.clone() for k, v in m.state_dict().items()}
    live_payload = {"model": m.state_dict()}    # references, NOT a copy

    path = str(tmp_path / "torn.pt")
    done = threading.Event()

    def write():
        torch.save(live_payload, path)
        done.set()

    th = threading.Thread(target=write)
    th.start()
    # Mutate hard while the write is in flight.
    for _ in range(300):
        step(m, o, 256)
    th.join()
    assert done.is_set()

    persisted = torch.load(path, weights_only=False)["model"]
    torn = any(not torch.equal(persisted[k], v) for k, v in at_submit.items())
    assert torn, ("expected a torn checkpoint without cloning; if this ever "
                  "stops tearing the test is no longer proving anything, not "
                  "that the hazard went away")


def test_save_checkpoint_with_writer_captures_submit_time_state(tmp_path):
    """The positive case: the same race, but through save_checkpoint."""
    d = str(tmp_path / "c")
    m, o = model_and_opt(width=256)
    step(m, o, 256)

    writer = AsyncCheckpointWriter()
    try:
        save_checkpoint(m, o, 0, 1, 1, rank=0, world_size=1, checkpoint_dir=d,
                        seed=1337, batch_size=32, writer=writer)
        at_submit = {k: v.clone() for k, v in m.state_dict().items()}
        for _ in range(300):
            step(m, o, 256)
        writer.flush()
    finally:
        writer.close()

    fresh, fresh_o = model_and_opt(seed=999, width=256)
    load_checkpoint(find_latest_checkpoint(d), fresh, fresh_o, rank=0)
    for k, v in at_submit.items():
        assert torch.equal(fresh.state_dict()[k], v), k


# --------------------------------------------------------------------------
# Writer mechanics
# --------------------------------------------------------------------------

def test_writer_runs_work_off_the_calling_thread():
    writer = AsyncCheckpointWriter()
    calling_thread = threading.get_ident()
    seen = {}

    try:
        writer.submit(lambda: seen.setdefault("tid", threading.get_ident()),
                      global_step=1)
        writer.flush()
    finally:
        writer.close()
    assert seen["tid"] != calling_thread


def test_writer_blocks_less_than_the_work_it_defers():
    """The actual claim of async: submit returns before the write finishes."""
    writer = AsyncCheckpointWriter()
    try:
        blocked = writer.submit(lambda: time.sleep(0.4), global_step=1)
        assert blocked < 0.1, f"submit blocked {blocked:.3f}s; should return fast"
        writer.flush()
        assert writer.stats()["mean_background_seconds"] >= 0.35
    finally:
        writer.close()


def test_second_submit_stalls_while_the_first_is_in_flight():
    """
    One write at a time by design. Two concurrent 90 MB writes contend for
    the same disk and neither lands sooner, while doubling peak memory. The
    stall is the honest signal that checkpoints are outpacing the disk.
    """
    writer = AsyncCheckpointWriter()
    try:
        writer.submit(lambda: time.sleep(0.4), global_step=1)
        blocked = writer.submit(lambda: None, global_step=2)
        assert blocked >= 0.3, f"second submit only blocked {blocked:.3f}s"
        assert writer.stats()["stalls"] == 1
        writer.flush()
    finally:
        writer.close()


def test_writer_error_does_not_kill_training():
    """
    A failed background write leaves the tmp file unrenamed, so recovery
    falls back to the previous checkpoint -- the same outcome as a crash
    mid-write. It must not propagate into the training loop.
    """
    writer = AsyncCheckpointWriter()

    def boom():
        raise OSError("disk full")

    try:
        writer.submit(boom, global_step=7)
        writer.flush()                     # must not raise
        stats = writer.stats()
        assert stats["writes_completed"] == 0
        assert len(stats["errors"]) == 1
        assert "disk full" in stats["errors"][0]
        assert "step 7" in stats["errors"][0]
        # And the writer stays usable afterwards.
        writer.submit(lambda: None, global_step=8)
        writer.flush()
        assert writer.stats()["writes_completed"] == 1
    finally:
        writer.close()


def test_flush_is_idempotent_and_safe_when_empty():
    writer = AsyncCheckpointWriter()
    try:
        assert writer.flush() < 0.1
        assert writer.flush() < 0.1
        assert writer.stats()["writes_completed"] == 0
    finally:
        writer.close()


def test_in_flight_reports_true_during_a_write():
    writer = AsyncCheckpointWriter()
    try:
        writer.submit(lambda: time.sleep(0.3), global_step=1)
        assert writer.in_flight
        writer.flush()
        assert not writer.in_flight
    finally:
        writer.close()


# --------------------------------------------------------------------------
# The sync path must be untouched
# --------------------------------------------------------------------------

def test_sync_path_is_unchanged_when_no_writer_given(tmp_path):
    d = str(tmp_path / "c")
    m, o = model_and_opt()
    step(m, o)
    expected = {k: v.clone() for k, v in m.state_dict().items()}

    blocked = save_checkpoint(m, o, 0, 1, 1, rank=0, world_size=1,
                             checkpoint_dir=d, seed=1337, batch_size=32)
    # Synchronous: the file is fully on disk by the time save returns.
    path = find_latest_checkpoint(d)
    assert path is not None and os.path.exists(path)
    assert blocked > 0

    fresh, fresh_o = model_and_opt(seed=999)
    load_checkpoint(path, fresh, fresh_o, rank=0)
    for k, v in expected.items():
        assert torch.equal(fresh.state_dict()[k], v)


def test_async_checkpoint_is_not_visible_until_it_lands(tmp_path):
    """
    Atomic rename means an in-flight async write is invisible to recovery --
    it never sees a partial file, it just sees the previous checkpoint. This
    is precisely why a checkpoint in flight when the process dies is lost
    cleanly rather than corruptly.
    """
    d = str(tmp_path / "c")
    m, o = model_and_opt(width=256)
    step(m, o, 256)

    gate = threading.Event()
    writer = AsyncCheckpointWriter()
    try:
        # Hold the write open, then confirm nothing is discoverable yet.
        writer.submit(lambda: gate.wait(timeout=5), global_step=1)
        assert find_latest_checkpoint(d) is None
        gate.set()
        writer.flush()
    finally:
        writer.close()


def test_non_zero_rank_still_writes_nothing_in_async_mode(tmp_path):
    d = str(tmp_path / "c")
    m, o = model_and_opt()
    writer = AsyncCheckpointWriter()
    try:
        save_checkpoint(m, o, 0, 1, 1, rank=1, world_size=1,
                        checkpoint_dir=d, seed=1337, batch_size=32,
                        writer=writer)
        writer.flush()
        assert writer.stats()["writes_completed"] == 0
    finally:
        writer.close()
    assert find_latest_checkpoint(d) is None
