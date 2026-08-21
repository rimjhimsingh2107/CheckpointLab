"""
Checkpoint save/restore for CheckpointLab (Phase 2).

The contract this module has to satisfy is stronger than "training doesn't
crash after a restart". It has to be *bit-for-bit resumable*: a run that is
killed and resumed must produce the same loss trajectory as a run that was
never interrupted. That means a checkpoint has to capture everything the
next step depends on:

  1. model weights          — model.state_dict()
  2. optimizer state        — SGD momentum buffers, etc.
  3. position in the data   — epoch + how many steps of that epoch are done
  4. RNG state              — torch / numpy / python / cuda, PER RANK

(4) is the part that is usually forgotten. Data augmentation (RandomCrop,
RandomHorizontalFlip) draws from the global torch RNG, so a resumed run
that doesn't restore RNG state sees *different augmented images* than the
uninterrupted run did, and the loss trajectories drift apart immediately
even though nothing is technically "broken".

RNG state is captured per rank, not once on rank 0. Each rank seeds itself
with `seed + rank` and sees a different shard of the data, so restoring
rank 0's RNG onto rank 1 would desynchronize rank 1 from what it would
have done uninterrupted. We all_gather every rank's RNG state and store
the whole list; each rank restores its own entry on load.

Only rank 0 writes to disk (the gather means rank 0 has everyone's state),
which avoids world_size redundant writes of an identical multi-MB file.

Writes are atomic: serialize to `<name>.tmp` then `os.replace()` onto the
final path. Without this, a failure injected *during* a checkpoint write
(exactly the case Phase 3 exists to create) leaves a truncated file that
looks like a valid checkpoint and poisons recovery.
"""
import glob
import json
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from checkpoint.async_writer import AsyncCheckpointWriter, clone_to_cpu

DEFAULT_CHECKPOINT_DIR = os.path.join("results", "checkpoints")

# Bump if the on-disk layout changes incompatibly, so old checkpoints get
# rejected loudly instead of silently loading garbage into the model.
#
# v2 added batch_size and samples_into_epoch, which is what makes elastic
# resumption possible: step_in_epoch alone is meaningless at a different world
# size, because one step consumes world_size * batch_size samples. A v1
# checkpoint cannot be converted -- it does not record the batch size -- so it
# is rejected rather than guessed at.
FORMAT_VERSION = 2

_CKPT_GLOB = "ckpt_step*.pt"
_MANIFEST = "latest.json"


# --------------------------------------------------------------------------
# RNG state
# --------------------------------------------------------------------------

def capture_rng_state() -> Dict[str, Any]:
    """Snapshot every RNG that can influence the next training step."""
    state: Dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Inverse of capture_rng_state()."""
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])

    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        # A checkpoint taken on a machine with a different GPU count can't
        # have its per-device RNG restored 1:1; skip rather than crash.
        if len(cuda_state) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_state)


def _gather_rng_states(rank: int, world_size: int) -> List[Dict[str, Any]]:
    """
    Collect every rank's RNG state so rank 0 can write all of them.

    Returns a list indexed by rank. Single-process runs skip the collective.
    """
    local = capture_rng_state()
    if world_size == 1 or not dist.is_available() or not dist.is_initialized():
        return [local]

    gathered: List[Optional[Dict[str, Any]]] = [None] * world_size
    dist.all_gather_object(gathered, local)
    return [g for g in gathered]  # type: ignore[misc]


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def checkpoint_path(checkpoint_dir: str, global_step: int) -> str:
    return os.path.join(checkpoint_dir, f"ckpt_step{global_step:08d}.pt")


def list_checkpoints(checkpoint_dir: str) -> List[str]:
    """
    Every checkpoint in the directory, oldest first.

    Recovery needs the whole list, not just the newest: if the newest one is
    truncated — a worker killed mid-write, which is the exact failure mode
    Phase 3 injects — it has to fall back to the previous one rather than
    give up.
    """
    return sorted(glob.glob(os.path.join(checkpoint_dir, _CKPT_GLOB)))


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Newest checkpoint in `checkpoint_dir`, or None.

    Prefers the manifest written by save_checkpoint(), but falls back to
    scanning the directory so a checkpoint dir stays usable even if the
    manifest write was the thing that got interrupted.
    """
    manifest = os.path.join(checkpoint_dir, _MANIFEST)
    if os.path.exists(manifest):
        try:
            with open(manifest) as fh:
                candidate = json.load(fh)["path"]
            if os.path.exists(candidate):
                return candidate
        except (json.JSONDecodeError, KeyError, OSError):
            pass  # fall through to the directory scan

    matches = sorted(glob.glob(os.path.join(checkpoint_dir, _CKPT_GLOB)))
    return matches[-1] if matches else None


def _prune(checkpoint_dir: str, keep_last: int) -> None:
    if keep_last <= 0:
        return
    matches = sorted(glob.glob(os.path.join(checkpoint_dir, _CKPT_GLOB)))
    for stale in matches[:-keep_last]:
        try:
            os.remove(stale)
        except OSError:
            pass  # a missing file is the outcome we wanted anyway


# --------------------------------------------------------------------------
# Save / load
# --------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    rank: int,
    world_size: int,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    keep_last: int = 5,
    batch_size: Optional[int] = None,
    writer: Optional[AsyncCheckpointWriter] = None,
) -> Optional[float]:
    """
    Write a resumable checkpoint. Called by every rank; only rank 0 writes.

    `step_in_epoch` is the number of steps of `epoch` that are COMPLETE, so
    it doubles as the index of the batch to resume at. `global_step` is the
    same count across the whole run.

    Returns, on rank 0, the seconds TRAINING WAS BLOCKED -- which is what the
    Phase 5 scheduler wants for `d`, since `d` is the cost charged against
    training time, not the cost in absolute terms. Synchronously that is the
    whole write. With a `writer` it is the state clone plus hand-off, and the
    serialize/fsync happens off the training thread. None on other ranks.
    """
    # Collective — must run on every rank, before the rank-0-only early exit.
    rng_states = _gather_rng_states(rank, world_size)

    if rank != 0:
        return None

    started = time.perf_counter()

    # Unwrap DDP so the checkpoint is loadable by plain (non-DDP) code too,
    # e.g. single-process eval or a resume at a different world size.
    to_save = model.module if isinstance(model, DDP) else model

    payload = {
        "format_version": FORMAT_VERSION,
        "model": to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "world_size": world_size,
        "batch_size": batch_size,
        # Position in the epoch measured in SAMPLES, not steps. This is the
        # only form that survives a world-size change: a step is
        # world_size * batch_size samples, so "step 6" means one thing at 2
        # ranks and something else entirely at 3. Samples are absolute.
        "samples_into_epoch": (step_in_epoch * world_size * batch_size
                               if batch_size else None),
        "seed": seed,
        "rng_states": rng_states,
        "saved_at": time.time(),
    }
    if extra:
        payload["extra"] = extra

    os.makedirs(checkpoint_dir, exist_ok=True)
    final_path = checkpoint_path(checkpoint_dir, global_step)
    tmp_path = final_path + ".tmp"

    def _commit(to_write: Dict[str, Any]) -> None:
        """Serialize and publish. Identical work in both modes; the only
        difference is which thread runs it."""
        torch.save(to_write, tmp_path)
        os.replace(tmp_path, final_path)  # atomic on POSIX

        manifest_tmp = os.path.join(checkpoint_dir, _MANIFEST + ".tmp")
        with open(manifest_tmp, "w") as fh:
            json.dump({"path": final_path, "global_step": global_step,
                       "epoch": epoch, "step_in_epoch": step_in_epoch}, fh)
        os.replace(manifest_tmp, os.path.join(checkpoint_dir, _MANIFEST))

        _prune(checkpoint_dir, keep_last)

    if writer is None:
        # Synchronous path, unchanged. No clone is needed because nothing
        # else can mutate the tensors while this thread is inside torch.save.
        _commit(payload)
        return time.perf_counter() - started

    # Asynchronous path. The clone is mandatory, not an optimisation:
    # payload["model"] holds references to the live parameters, and training
    # resumes mutating them the moment this returns. See async_writer.py.
    snapshot = clone_to_cpu(payload)
    writer.submit(lambda: _commit(snapshot), global_step=global_step)

    # Elapsed here = the clone, plus any wait for a previous write still in
    # flight. Both are paid out of training time, so both belong in `d`.
    return time.perf_counter() - started


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    rank: int,
    map_location: Any = "cpu",
    restore_rng: bool = True,
    current_world_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Restore a checkpoint into `model`/`optimizer` and return its metadata.

    Each rank restores its OWN RNG entry (see module docstring). Returns a
    dict with epoch / step_in_epoch / global_step so the caller knows where
    in the data stream to pick up.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)

    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint {path} has format_version={version}, "
            f"expected {FORMAT_VERSION}. Refusing to load."
        )

    saved_world_size = payload["world_size"]

    target = model.module if isinstance(model, DDP) else model
    target.load_state_dict(payload["model"])

    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])

    rng_states = payload.get("rng_states") or []

    # Under a world-size change there is no honest mapping from the saved RNG
    # streams to the new ranks. Handing rank 3 a copy of rank 0's stream --
    # the obvious fallback -- is actively harmful: the two ranks would then
    # draw IDENTICAL augmentations, correlating the noise across what is
    # supposed to be an independently-sampled batch. Returning None instead
    # tells the caller to reseed deterministically from (seed, rank), which
    # keeps the streams independent. Bit-exactness is already unavailable at a
    # new world size (the effective batch size has changed), so there is
    # nothing left to preserve by faking continuity.
    if current_world_size is not None and current_world_size != saved_world_size:
        my_rng = None
    elif rank < len(rng_states):
        my_rng = rng_states[rank]
    else:
        my_rng = None

    if restore_rng and my_rng is not None:
        restore_rng_state(my_rng)

    return {
        "epoch": payload["epoch"],
        "step_in_epoch": payload["step_in_epoch"],
        "global_step": payload["global_step"],
        "world_size": saved_world_size,
        "batch_size": payload.get("batch_size"),
        "samples_into_epoch": payload.get("samples_into_epoch"),
        "seed": payload.get("seed"),
        "extra": payload.get("extra", {}),
        "path": path,
        # Handed back so the caller can re-apply it AFTER fast-forwarding the
        # dataloader to step_in_epoch — replaying those batches consumes RNG
        # draws and would otherwise clobber the restored state.
        "rng_state": my_rng,
    }
