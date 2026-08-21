"""
Background checkpoint writing.

THE CORRECTNESS HAZARD, WHICH IS THE WHOLE POINT

`model.state_dict()` does not copy anything -- it returns a dict of
*references* to the live parameter tensors. `torch.save` reads those tensors
at serialize time, not at call time. So handing a payload straight to a
background thread and letting training continue produces a torn checkpoint:
the first tensors serialized hold step N weights, the ones serialized after
the next optimizer.step() hold step N+1 weights. Nothing crashes. The file
is a valid .pt containing a model that never existed, and it only shows up
later as a resumed run whose loss trajectory has quietly diverged.

So the state is CLONED on the calling thread before hand-off. That is the
irreducible cost of async checkpointing: training blocks for a memcpy of the
model and optimizer instead of for a serialize-plus-fsync. On this testbed
that is roughly 90 MB of copy against 90 MB of copy + pickle + write + fsync.

WHY A THREAD AND NOT A PROCESS

A process would sidestep the GIL, but the payload would have to be pickled
across the process boundary -- which is most of the work `torch.save` does,
paid on the training thread, defeating the purpose. Shared memory avoids
that but means managing the lifetime of a 90 MB shared buffer per in-flight
checkpoint.

A thread is the right trade here because the competing work releases the GIL:
`torch.save`'s bulk writes happen in C++ with the GIL dropped, and the
training step is almost entirely ATen kernels that also drop it. The overlap
is therefore real but not perfect, and the benchmark measures it rather than
assuming it.

WHAT ASYNC COSTS YOU

A checkpoint that is still in flight when the process dies is lost. The
atomic tmp-then-rename means it is lost *cleanly* -- recovery sees the
previous checkpoint, never a half-written one -- but it is lost. Async trades
a little durability for a lot of throughput, and under a high failure rate
that trade can go the wrong way. `--async-checkpointing` is therefore opt-in.
"""
import copy
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import torch


def clone_to_cpu(obj: Any) -> Any:
    """
    Deep-copy a state payload, detaching every tensor onto CPU.

    Recursive because an optimizer state_dict is nested: {'state': {0:
    {'momentum_buffer': tensor}}, 'param_groups': [...]}. Cloning only the
    top level would leave the momentum buffers as live references -- the
    subtlest possible version of this bug, since the model weights would be
    consistent and only the optimizer state would be torn.
    """
    if torch.is_tensor(obj):
        # detach: no autograd graph in the checkpoint.
        # clone via to(copy=True): a real copy even if already on CPU.
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: clone_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cloned = [clone_to_cpu(v) for v in obj]
        return type(obj)(cloned) if isinstance(obj, tuple) else cloned
    # Scalars, strings, numpy RNG state tuples: deepcopy is cheap and safe.
    return copy.deepcopy(obj)


class AsyncCheckpointWriter:
    """
    Serializes checkpoints on a background thread, one at a time.

    Deliberately single-threaded: two concurrent 90 MB writes contend for the
    same disk and neither finishes sooner, while doubling peak memory. If a
    new checkpoint is due while one is in flight, the caller blocks -- which
    is the honest signal that checkpoints are being requested faster than the
    disk can absorb them, and is exactly what the Phase 5 scheduler should
    react to by lengthening the interval.
    """

    def __init__(self, name: str = "ckpt-writer"):
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix=name)
        self._pending: Optional[Future] = None
        self._pending_step: Optional[int] = None
        self._lock = threading.Lock()

        # Observability. The distinction that matters is blocked-vs-background:
        # only the first is paid out of training time.
        self.writes_completed = 0
        self.background_seconds: List[float] = []
        self.blocked_seconds: List[float] = []
        self.stalls = 0            # times a write was still in flight
        self.errors: List[str] = []

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(self, write_fn, *, global_step: int) -> float:
        """
        Hand `write_fn` to the background thread; return seconds blocked.

        The caller must already have cloned anything `write_fn` closes over.
        """
        blocked_start = time.perf_counter()

        # Drain any in-flight write first. One at a time, by design.
        if self._pending is not None and not self._pending.done():
            self.stalls += 1
        self._await_pending()

        with self._lock:
            self._pending_step = global_step
            self._pending = self._executor.submit(self._run, write_fn)

        blocked = time.perf_counter() - blocked_start
        self.blocked_seconds.append(blocked)
        return blocked

    def _run(self, write_fn) -> float:
        started = time.perf_counter()
        write_fn()
        return time.perf_counter() - started

    def _await_pending(self) -> None:
        pending = self._pending
        if pending is None:
            return
        try:
            elapsed = pending.result()
            self.background_seconds.append(elapsed)
            self.writes_completed += 1
        except Exception as exc:                      # noqa: BLE001
            # A failed background write must not kill training. The tmp file
            # is simply never renamed, so recovery falls back to the previous
            # checkpoint -- the same outcome as a crash mid-write.
            self.errors.append(f"step {self._pending_step}: {exc!r}")
        finally:
            self._pending = None
            self._pending_step = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush(self) -> float:
        """
        Block until nothing is in flight. Returns seconds waited.

        Called at orderly shutdown so the last checkpoint actually lands.
        NOT called before a simulated crash: a real preemption does not get
        to flush, and pretending otherwise would make async look more durable
        than it is.
        """
        started = time.perf_counter()
        self._await_pending()
        return time.perf_counter() - started

    def close(self) -> None:
        self.flush()
        self._executor.shutdown(wait=True)

    @property
    def in_flight(self) -> bool:
        return self._pending is not None and not self._pending.done()

    def stats(self) -> Dict[str, Any]:
        def mean(xs):
            return sum(xs) / len(xs) if xs else 0.0
        return {
            "writes_completed": self.writes_completed,
            "mean_blocked_seconds": mean(self.blocked_seconds),
            "total_blocked_seconds": sum(self.blocked_seconds),
            "mean_background_seconds": mean(self.background_seconds),
            "total_background_seconds": sum(self.background_seconds),
            "stalls": self.stalls,
            "errors": list(self.errors),
        }
