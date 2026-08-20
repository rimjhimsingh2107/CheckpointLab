"""
CheckpointLab — Phase 1: vanilla distributed training baseline.

This intentionally has NO checkpointing, NO failure injection, NO recovery
logic yet. The only goal here is to prove that DDP training actually works
end to end across multiple simulated workers before any fault-tolerance
machinery gets layered on top.

Run with:
    python training/train.py
"""
import os
import sys
import time

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(__file__))
from distributed import setup, cleanup, get_device
from model import build_model


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_dataloader(rank: int, world_size: int, batch_size: int = 128) -> DataLoader:
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    # Only rank 0 downloads to avoid a thundering herd of simultaneous
    # downloads; other ranks wait via a barrier after setup() is called.
    dataset = datasets.CIFAR10(root="./data", train=True, download=(rank == 0), transform=transform)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2, drop_last=True)
    return loader


def train_worker(rank: int, world_size: int, epochs: int) -> None:
    setup(rank, world_size)

    import torch.distributed as dist
    dist.barrier()  # make sure rank 0 finished downloading before others read the dataset

    device = get_device(rank)
    loader = get_dataloader(rank, world_size)

    model = build_model().to(device)
    ddp_backend = "nccl" if torch.cuda.is_available() else "gloo"
    model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    if rank == 0:
        print(f"[rank {rank}] Starting training: world_size={world_size}, "
              f"device={device}, backend={ddp_backend}, epochs={epochs}")

    for epoch in range(epochs):
        loader.sampler.set_epoch(epoch)  # required so shuffling differs per epoch across ranks
        model.train()

        epoch_start = time.time()
        running_loss = 0.0

        for step, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if rank == 0 and step % 50 == 0:
                print(f"[epoch {epoch}] step {step}/{len(loader)} "
                      f"loss={loss.item():.4f}")

        if rank == 0:
            avg_loss = running_loss / len(loader)
            elapsed = time.time() - epoch_start
            print(f"[epoch {epoch}] DONE avg_loss={avg_loss:.4f} time={elapsed:.1f}s")

    cleanup()


def main() -> None:
    world_size = min(2, max(1, torch.cuda.device_count())) if torch.cuda.is_available() else 2
    epochs = 3

    print(f"Launching {world_size} simulated worker(s)...")
    mp.spawn(train_worker, args=(world_size, epochs), nprocs=world_size, join=True)
    print("Training complete.")


if __name__ == "__main__":
    main()
