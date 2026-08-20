"""
Distributed process group setup/teardown for CheckpointLab.

Handles both GPU (nccl) and CPU-only (gloo) environments so the project
runs on a laptop with no GPU as well as on real multi-GPU machines.
"""
import os
import torch
import torch.distributed as dist


def get_backend() -> str:
    """Pick the right torch.distributed backend for this machine."""
    return "nccl" if torch.cuda.is_available() else "gloo"


def setup(rank: int, world_size: int, master_addr: str = "localhost", master_port: str = "29500") -> None:
    """
    Initialize the distributed process group for a single worker.

    rank: this worker's index (0..world_size-1)
    world_size: total number of workers participating
    """
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)

    backend = get_backend()
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())


def cleanup() -> None:
    """Tear down the process group cleanly."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(rank: int) -> torch.device:
    """Return the device this rank should use."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    return torch.device("cpu")
