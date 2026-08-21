"""
Sync vs async checkpoint overhead.

Measures the quantity that actually matters: **seconds training is blocked
per checkpoint**. Everything else about a checkpoint write is real work, but
it is only a cost to the job if training is waiting on it.

Synchronously that is the whole write: serialize ~90 MB, write it, rename.
Asynchronously it is the state clone plus hand-off, with the serialize
happening on another thread. The interesting number is the ratio, and whether
the background write actually keeps up -- if checkpoints are requested faster
than the disk absorbs them, the caller starts stalling and async degenerates
towards sync.

This benchmarks the write path directly rather than through a training run,
so the numbers are not confounded by dataloader jitter or CPU contention from
DDP peers. `experiments/benchmark.py` covers the end-to-end effect.

Usage:
    python checkpoint/bench_async.py
    python checkpoint/bench_async.py --checkpoints 12 --work-per-step 0.05
"""
import argparse
import json
import os
import shutil
import statistics
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from checkpoint.async_writer import AsyncCheckpointWriter
from checkpoint.manager import save_checkpoint
from training.model import build_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_state():
    """A real ResNet-18 and SGD-with-momentum, so the payload is realistic."""
    torch.manual_seed(1337)
    model = build_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                          weight_decay=5e-4)
    # One step, so momentum buffers exist -- they roughly double the payload
    # and a benchmark without them understates the write by ~2x.
    out = model(torch.randn(2, 3, 32, 32))
    loss = out.square().mean()
    opt.zero_grad(); loss.backward(); opt.step()
    return model, opt


def simulate_step(model, opt, seconds: float) -> None:
    """
    Burn approximately `seconds` of real training-shaped work.

    Matters for the async case: with no work between checkpoints there is
    nothing for the background write to overlap WITH, and async would look
    artificially good (or bad, once it starts stalling).
    """
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        out = model(torch.randn(8, 3, 32, 32))
        loss = out.square().mean()
        opt.zero_grad(); loss.backward(); opt.step()


def run_mode(mode: str, n: int, work_per_step: float, work_dir: str) -> Dict:
    ckpt_dir = os.path.join(work_dir, f"ckpt_{mode}")
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    model, opt = make_state()
    writer = AsyncCheckpointWriter() if mode == "async" else None

    blocked: List[float] = []
    wall_start = time.perf_counter()

    for i in range(1, n + 1):
        simulate_step(model, opt, work_per_step)
        b = save_checkpoint(model, opt, epoch=0, step_in_epoch=i, global_step=i,
                            rank=0, world_size=1, checkpoint_dir=ckpt_dir,
                            seed=1337, batch_size=32, keep_last=3,
                            writer=writer)
        blocked.append(b)

    flush_wait = writer.flush() if writer else 0.0
    wall = time.perf_counter() - wall_start
    stats = writer.stats() if writer else {}
    if writer:
        writer.close()

    size_mb = 0.0
    if os.path.isdir(ckpt_dir):
        pts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        if pts:
            size_mb = os.path.getsize(os.path.join(ckpt_dir, pts[0])) / 1e6

    return {
        "mode": mode,
        "checkpoints": n,
        "checkpoint_mb": round(size_mb, 1),
        "blocked_total_s": round(sum(blocked), 4),
        "blocked_mean_ms": round(1000 * statistics.mean(blocked), 1),
        "blocked_median_ms": round(1000 * statistics.median(blocked), 1),
        "blocked_max_ms": round(1000 * max(blocked), 1),
        "wall_s": round(wall, 2),
        "flush_wait_s": round(flush_wait, 3),
        "background_mean_ms": round(
            1000 * stats.get("mean_background_seconds", 0.0), 1),
        "background_total_s": round(stats.get("total_background_seconds", 0.0), 3),
        "stalls": stats.get("stalls", 0),
        "errors": stats.get("errors", []),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", type=int, default=10)
    p.add_argument("--work-per-step", type=float, default=1.0,
                   help="Seconds of training-shaped work between checkpoints. "
                        "The default approximates this testbed's ~1 s/step.")
    args = p.parse_args(argv)

    work = os.path.join(REPO_ROOT, "results", "bench_async")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)

    print("=" * 78)
    print("SYNC vs ASYNC CHECKPOINT OVERHEAD")
    print(f"  ResNet-18 + SGD momentum, {args.checkpoints} checkpoints, "
          f"{args.work_per_step}s of work between each")
    print("  metric of interest: seconds TRAINING IS BLOCKED per checkpoint")
    print("=" * 78)

    rows = []
    for mode in ("sync", "async"):
        print(f"\n  running {mode} ...", flush=True)
        rows.append(run_mode(mode, args.checkpoints, args.work_per_step, work))

    sync, asyn = rows[0], rows[1]

    print(f"\n  checkpoint size: {sync['checkpoint_mb']} MB\n")
    print(f"  {'':<22} {'sync':>12} {'async':>12}")
    for label, key in (("blocked mean (ms)", "blocked_mean_ms"),
                       ("blocked median (ms)", "blocked_median_ms"),
                       ("blocked max (ms)", "blocked_max_ms"),
                       ("blocked total (s)", "blocked_total_s"),
                       ("wall clock (s)", "wall_s")):
        print(f"  {label:<22} {sync[key]:>12} {asyn[key]:>12}")
    print(f"  {'background mean (ms)':<22} {'-':>12} "
          f"{asyn['background_mean_ms']:>12}")
    print(f"  {'writer stalls':<22} {'-':>12} {asyn['stalls']:>12}")
    print(f"  {'flush wait at end (s)':<22} {'-':>12} "
          f"{asyn['flush_wait_s']:>12}")

    if asyn["errors"]:
        print(f"\n  ASYNC WRITER ERRORS: {asyn['errors']}")

    reduction = (100.0 * (sync["blocked_mean_ms"] - asyn["blocked_mean_ms"])
                 / sync["blocked_mean_ms"]) if sync["blocked_mean_ms"] else 0.0
    speedup = (sync["blocked_mean_ms"] / asyn["blocked_mean_ms"]
               if asyn["blocked_mean_ms"] else float("inf"))

    print(f"\n  -> async blocks {reduction:.1f}% less per checkpoint "
          f"({speedup:.1f}x lower), {sync['blocked_mean_ms']:.1f} ms -> "
          f"{asyn['blocked_mean_ms']:.1f} ms")
    print(f"  -> the write still costs "
          f"{asyn['background_mean_ms']:.1f} ms, just not on the training thread")
    if asyn["stalls"]:
        print(f"  -> {asyn['stalls']} stall(s): checkpoints were requested "
              f"faster than the disk absorbed them, so async partially "
              f"degenerated toward sync")
    else:
        print("  -> 0 stalls: the background writer kept up at this interval")
    print("=" * 78)

    with open(os.path.join(work, "summary.json"), "w") as fh:
        json.dump({"sync": sync, "async": asyn,
                   "blocked_reduction_pct": round(reduction, 2),
                   "blocked_speedup": round(speedup, 2)}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
