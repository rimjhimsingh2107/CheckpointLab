"""
CheckpointLab — distributed training loop.

Phase 1 established the vanilla DDP baseline. Phase 2 adds checkpointing:
the loop now saves resumable state every N steps and, on startup, picks up
from the newest checkpoint in `--checkpoint-dir` if one exists.

The interval is deliberately a hardcoded CLI flag for now. Phase 5 replaces
it with the adaptive scheduler; keeping it dumb here means the benchmark in
Phase 6 has an honest fixed-interval control group to compare against.

Run with:
    python training/train.py                          # full 3-epoch run
    python training/train.py --checkpoint-every 50
    python training/train.py --no-resume              # ignore existing ckpts

Determinism note: `--deterministic` forces `num_workers=0`. DataLoader
worker processes derive their RNG from the parent at *iterator creation*
time and prefetch batches eagerly, so their augmentation streams can't be
restored from a checkpoint. Exact resume therefore requires in-process data
loading. Throughput runs should leave it off.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distributed import setup, cleanup, get_device
from model import build_model
from checkpoint.manager import (
    DEFAULT_CHECKPOINT_DIR,
    find_latest_checkpoint,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CheckpointLab distributed training")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--world-size", type=int, default=0,
                   help="0 = auto (GPU count, or 2 simulated CPU workers)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader workers; forced to 0 by --deterministic")
    p.add_argument("--subset-size", type=int, default=0,
                   help="Train on the first N CIFAR-10 images only (0 = all). "
                        "Used to keep correctness runs cheap on CPU.")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Stop after this many global steps (0 = run all epochs)")

    p.add_argument("--checkpoint-every", type=int, default=100,
                   help="Save a checkpoint every N global steps (0 = never)")
    p.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--no-resume", action="store_true",
                   help="Start from scratch even if a checkpoint exists")

    p.add_argument("--deterministic", action="store_true",
                   help="Force the exactly-reproducible data path")
    p.add_argument("--loss-log", default=None,
                   help="Append per-step loss as JSONL (rank 0 only)")
    p.add_argument("--launch-id", default="run",
                   help="Tag written into --loss-log to identify this launch")
    p.add_argument("--crash-at-step", type=int, default=0,
                   help="Simulate a hard worker crash (os._exit) right after "
                        "this global step. 0 = never.")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--master-port", type=int, default=0,
                   help="Rendezvous port. 0 = pick a free ephemeral port, "
                        "which is what makes back-to-back launches safe.")

    return p.parse_args(argv)


def pick_master_port(requested: int) -> str:
    """
    Choose the rendezvous port for this launch.

    A hardcoded port breaks the exact workflow this project is built around:
    kill a run, immediately relaunch it, and the dead run's listener is
    still in TIME_WAIT, so the new one dies with EADDRINUSE before training
    even starts. Binding port 0 and reading back what the OS gave us
    sidesteps that. There's a small race between closing this socket and
    the process group binding it, which is acceptable for a single-host
    testbed and would need a real rendezvous backend on a real cluster.
    """
    if requested:
        return str(requested)

    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return str(sock.getsockname()[1])


def resolve_world_size(requested: int) -> int:
    if requested > 0:
        return requested
    if torch.cuda.is_available():
        return min(2, max(1, torch.cuda.device_count()))
    return 2  # two simulated CPU workers via gloo


def seed_everything(seed: int, rank: int) -> None:
    """
    Seed every RNG this rank uses.

    Offsetting by rank is deliberate: each rank trains on a different shard,
    so giving them identical augmentation streams would correlate the noise
    across the batch. The checkpoint manager stores state per rank for the
    same reason.
    """
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def ensure_dataset_downloaded(rank: int) -> None:
    """
    Only rank 0 downloads CIFAR-10. Must be called and completed BEFORE the
    barrier below, otherwise other ranks race ahead and try to read a
    dataset that hasn't finished downloading yet.
    """
    if rank == 0:
        datasets.CIFAR10(root="./data", train=True, download=True)


def get_dataloader(rank: int, world_size: int, args: argparse.Namespace) -> DataLoader:
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    # By this point the dataset is guaranteed to already be on disk (see
    # ensure_dataset_downloaded + barrier in train_worker), so every rank
    # can safely use download=False here.
    dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)

    if args.subset_size:
        # A contiguous prefix, not a random sample — the subset has to be
        # identical across launches for a resumed run to be comparable.
        dataset = Subset(dataset, range(min(args.subset_size, len(dataset))))

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=args.seed)

    # See the module docstring: worker processes can't have their RNG
    # restored from a checkpoint, so exact resume needs in-process loading.
    num_workers = 0 if args.deterministic else args.num_workers

    return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                      num_workers=num_workers, drop_last=True)


class LossLog:
    """Line-buffered JSONL writer for per-step losses (rank 0 only)."""

    def __init__(self, path: str, launch_id: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # Append, not truncate: a resumed launch adds to the same file so
        # verify.py can reconstruct the full trajectory across restarts.
        self._fh = open(path, "a")
        self._launch_id = launch_id

    def write(self, **record) -> None:
        record["launch"] = self._launch_id
        self._fh.write(json.dumps(record) + "\n")
        # Flushed every line on purpose: --crash-at-step uses os._exit(),
        # which skips all buffer flushing. Anything still buffered when a
        # worker dies is exactly the data we need to prove the resume.
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()


def train_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    setup(rank, world_size)

    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed_everything(args.seed, rank)

    ensure_dataset_downloaded(rank)
    dist.barrier()  # all ranks wait here until rank 0's download above is fully done

    device = get_device(rank)
    loader = get_dataloader(rank, world_size, args)
    steps_per_epoch = len(loader)

    model = build_model().to(device)
    ddp_backend = "nccl" if torch.cuda.is_available() else "gloo"
    model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # ---------------- resume ----------------
    start_epoch, resume_skip, global_step = 0, 0, 0
    resume_rng = None

    if not args.no_resume:
        latest = find_latest_checkpoint(args.checkpoint_dir)
        if latest:
            meta = load_checkpoint(latest, model, optimizer, rank,
                                   map_location=device)
            start_epoch = meta["epoch"]
            resume_skip = meta["step_in_epoch"]
            global_step = meta["global_step"]
            resume_rng = meta["rng_state"]
            if rank == 0:
                print(f"[rank {rank}] RESUMED from {latest} "
                      f"(epoch={start_epoch}, step_in_epoch={resume_skip}, "
                      f"global_step={global_step})")
        elif rank == 0:
            print(f"[rank {rank}] No checkpoint in {args.checkpoint_dir}; "
                  f"starting from scratch")

    loss_log = LossLog(args.loss_log, args.launch_id) if (args.loss_log and rank == 0) else None

    if rank == 0:
        print(f"[rank {rank}] Starting training: world_size={world_size}, "
              f"device={device}, backend={ddp_backend}, epochs={args.epochs}, "
              f"steps_per_epoch={steps_per_epoch}, "
              f"checkpoint_every={args.checkpoint_every}")

    stop = False

    for epoch in range(start_epoch, args.epochs):
        loader.sampler.set_epoch(epoch)  # required so shuffling differs per epoch across ranks
        model.train()

        epoch_start = time.time()
        running_loss = 0.0
        counted = 0

        # Manual iterator so a resumed epoch can be fast-forwarded to the
        # exact batch the checkpoint stopped at. `set_epoch` above makes the
        # sampler's permutation a pure function of (seed, epoch), so batch i
        # of this epoch is the same batch i the original run saw.
        batches = iter(loader)
        step = 0

        if epoch == start_epoch and resume_skip:
            for _ in range(resume_skip):
                next(batches)
            step = resume_skip
            # Replaying those batches ran the augmentation transforms, which
            # advanced the RNG. Re-apply the checkpointed state so the NEXT
            # batch is augmented exactly as it was in the uninterrupted run.
            if resume_rng is not None:
                restore_rng_state(resume_rng)
            if rank == 0:
                print(f"[rank {rank}] fast-forwarded epoch {epoch} to step {step}")

        for images, labels in batches:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            step += 1
            global_step += 1

            local_loss = loss.detach()
            # Mean loss across ranks. Rank 0's local loss alone would only
            # catch a divergence on rank 0; the all-reduce makes the logged
            # number sensitive to every rank's data stream, so verify.py
            # detects a desynchronized peer on the very step it happens.
            if world_size > 1:
                gathered = local_loss.clone()
                dist.all_reduce(gathered, op=dist.ReduceOp.SUM)
                global_loss = (gathered / world_size).item()
            else:
                global_loss = local_loss.item()

            running_loss += global_loss
            counted += 1

            if loss_log is not None:
                loss_log.write(global_step=global_step, epoch=epoch,
                               step_in_epoch=step, loss=global_loss,
                               rank0_loss=local_loss.item())

            if rank == 0 and (step % args.log_every == 0 or step == 1):
                print(f"[epoch {epoch}] step {step}/{steps_per_epoch} "
                      f"global_step={global_step} loss={global_loss:.6f}")

            if args.checkpoint_every and global_step % args.checkpoint_every == 0:
                # A checkpoint taken at the end of an epoch records
                # (epoch+1, 0) rather than (epoch, steps_per_epoch): the
                # epoch is finished, and resuming would otherwise try to
                # fast-forward past the end of the iterator.
                ck_epoch, ck_step = ((epoch + 1, 0) if step == steps_per_epoch
                                     else (epoch, step))
                dur = save_checkpoint(
                    model, optimizer, ck_epoch, ck_step, global_step,
                    rank, world_size, checkpoint_dir=args.checkpoint_dir,
                    seed=args.seed, extra={"loss": global_loss},
                )
                if rank == 0:
                    print(f"[epoch {epoch}] checkpoint @ global_step="
                          f"{global_step} ({dur * 1000:.0f} ms)")

            if args.crash_at_step and global_step >= args.crash_at_step:
                if rank == 0:
                    print(f"[rank {rank}] SIMULATED CRASH at global_step="
                          f"{global_step}", flush=True)
                if loss_log is not None:
                    loss_log.close()
                # os._exit, not sys.exit: no atexit handlers, no buffer
                # flush, no process-group teardown. A real preemption or
                # OOM-kill doesn't get to run cleanup code either.
                os._exit(137)

            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break

        if rank == 0 and counted:
            avg_loss = running_loss / counted
            elapsed = time.time() - epoch_start
            print(f"[epoch {epoch}] DONE avg_loss={avg_loss:.4f} time={elapsed:.1f}s")

        if stop:
            if rank == 0:
                print(f"[rank {rank}] hit --max-steps={args.max_steps}, stopping")
            break

    if loss_log is not None:
        loss_log.close()
    cleanup()


def main(argv=None) -> None:
    args = parse_args(argv)
    world_size = resolve_world_size(args.world_size)

    # Set before spawn so every child inherits the same rendezvous point
    # (distributed.setup uses setdefault, so this env var wins).
    os.environ["MASTER_PORT"] = pick_master_port(args.master_port)

    print(f"Launching {world_size} simulated worker(s) "
          f"(rendezvous port {os.environ['MASTER_PORT']})...")
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    print("Training complete.")


if __name__ == "__main__":
    main()
