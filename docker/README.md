# Docker

Containerizes the CheckpointLab worker. One image serves every role; which
rank a container is — and whether it's a worker or the controller — is
entirely the compose command's business. **No training code is
container-aware.** Phase 4 already taught `training/train.py` to run as a
single rank in one process (`--rank N --world-size N --master-port P`, much
like `torchrun`) so the controller could supervise ranks individually; a
container is just a stronger boundary around that same thing.

```bash
docker compose -f docker/docker-compose.yml --profile train   up    # 2 containers, 1 per rank
docker compose -f docker/docker-compose.yml --profile recover run --rm controller
docker compose -f docker/docker-compose.yml --profile test    run --rm tests

bash docker/verify_docker.sh        # all four checks, end to end
```

## What actually runs in containers vs. on bare metal

| | Containerized | Notes |
|---|---|---|
| DDP training, 1 container per rank | **yes** | `--profile train`, gloo over the compose bridge network |
| Fast test tier | **yes** | `--profile test`, 177 tests |
| Crash → detect → recover → resume | **yes, single container** | `--profile recover`; the controller supervises its ranks as processes *inside* one container — see below |
| Controller supervising ranks in *sibling containers* | **no** | not implemented; the honest limitation, explained below |
| Slow test tier | no | runs on bare metal / CI; ~4.6 min there, and it spawns its own subprocess topologies |
| Benchmark suite | no | 12 sequential jobs, ~35 min; nothing about it needs a container |

## Verified, on this machine

Docker Desktop 29.7.2, 8 CPUs, 4 GB, `linux/aarch64`. All four checks passed
(`bash docker/verify_docker.sh`, exit 0):

**1 — image builds, torch imports.** `torch 2.13.0+cpu`, `cuda=False`, gloo
available. 305 MB compressed / 1.46 GB on disk.

**2 — fast test tier in-container:** `177 passed, 3 deselected in 5.32s`.

**3 — two containers, one per rank, real DDP over the container network.**
12 steps, both ranks exit 0, checkpoints written to the bind-mounted volume.
Repeating the identical run reproduces the losses **bit-for-bit**:

```
in-container determinism: max abs diff over 12 steps = 0.000e+00
```

**4 — crash → detect → recover → resume, in-container.** The Phase 3 injector
killed worker 1 at step 7 (as its seed dictates), the controller detected the
exit, tore down the surviving rank, restored `ckpt_step00000006.pt` and
relaunched:

```
  !! FAILURE DETECTED  worker 1 exited 137 (last reported step 7)
     terminating generation 1 (survivors still up: [0])
     resuming from ckpt_step00000006.pt (global_step=6, epoch=0, step_in_epoch=6)
     RECOVERED in 6.95s (lost 1 step(s) of work)
  outcome=completed generations=2 exit_codes={'0': 0, '1': 0}
```

### Recovery is ~1.8× slower in a container

| | bare metal | container |
|---|---|---|
| recovery latency | 3.79 s | 6.13–6.95 s |
| of which `reach_first_step` | 3.69 s (97.5%) | 6.79 s (97.7%) |
| `terminate_workers` | 0.06 s | 0.06 s |
| `select_checkpoint` | 0.03 s | 0.10–0.16 s |

The controller's own bookkeeping is unchanged — teardown is identical to the
millisecond. The entire regression is `reach_first_step`: a fresh interpreter
importing torch, rebuilding the model and reloading a ~90 MB checkpoint,
where the checkpoint read crosses a bind mount and the interpreter starts on
an overlay filesystem. That is the number to attack if container recovery
latency ever matters, and it is the same conclusion Phase 4 reached on bare
metal — process startup dominates, not the recovery logic.

## Containerized training is NOT bit-identical to bare metal — by design

The tempting claim is "same losses in and out of Docker". It is false here,
and quietly widening the tolerance to make it true would be dishonest:

```
container (linux/aarch64 wheel) vs host (macOS/arm64 wheel)
 step       container      bare metal       diff
    1    2.3443727493    2.3443734646   7.15e-07
    4    2.3554928303    2.3573927879   1.90e-03
   12    2.2820925713    2.2756547928   6.44e-03
max abs diff = 1.365e-02
```

That is ~1000× the `1e-5` tolerance Phase 2 uses. The cause is not a bug: the
container runs a **Linux/aarch64** CPU wheel of torch and the host runs a
**macOS/arm64** build. Different builds means different kernels and different
floating-point reduction orders, so the two diverge on the very first step
(7e-07) and compound from there.

Phase 2's determinism guarantee was always a *within-environment* property —
"a resumed run reproduces an uninterrupted run **on the same build**" — and
that still holds exactly, in the container (`0.000e+00` above) and on bare
metal. It was never a cross-platform claim, and `verify_docker.sh` prints the
cross-platform delta as information, never as pass/fail.

Practical consequence: **don't compare loss trajectories across the container
boundary.** Run the control and the treatment in the same environment. The
Phase 6 benchmark does exactly this already, which is why it is unaffected.

## The real limitation: no cross-container supervision

`--profile train` runs **two containers**. `--profile recover` runs **one**.
That asymmetry is deliberate and worth stating plainly.

`controller/worker_manager.py` supervises ranks with `subprocess.Popen`,
`process.poll()` and `os.killpg`. Those operate on **local PIDs**, so a
controller in container A cannot poll or signal a rank in container B. There
is no shared PID namespace and no reason there should be.

So the containerized recovery demo proves the loop works *inside* a container
— same gloo rendezvous, same atomic-checkpoint filesystem semantics, same
signal handling, same detection logic — but not that the controller can
supervise sibling containers.

Making that work is a feature, not a compose-file detail. `worker_manager`
would need a pluggable backend: `docker create/start/kill/wait` against the
Docker socket (or the Kubernetes API) instead of `Popen`/`killpg`, plus a
liveness model that doesn't assume shared PIDs — the heartbeat files already
generalize, since they go through a shared volume, but process-exit detection
does not. In practice you would more likely delegate this entirely to
`torchrun --max-restarts` or a Kubernetes Job with `restartPolicy`, and keep
the adaptive scheduler as the part that's actually novel here.

## Implementation notes

- **`.dockerignore` is load-bearing.** The build context is the repo root, and
  the repo carries 7.8 GB of benchmark job dirs, a 782 MB host virtualenv and
  340 MB of CIFAR-10. Without it Docker tars ~8.9 GB and ships it to the
  daemon before building anything; with it the context is **0.3 MB**.
- **CPU-only wheel on purpose.** The default Linux torch wheel bundles CUDA
  and is ~2.5 GB. This project trains on gloo across CPU workers, so
  `--index-url https://download.pytorch.org/whl/cpu` is both correct and
  roughly an order of magnitude smaller. For GPU runs, swap the base image
  for `nvidia/cuda` and drop the index URL.
- **`libgomp1` is required.** torch's CPU kernels are OpenMP-parallel and
  `python:3.13-slim` doesn't ship the runtime. Without it `import torch`
  fails with a bare `libgomp.so.1: cannot open shared object file`.
- **`MASTER_ADDR` is the only container-specific setting.** Rank 0 hosts the
  gloo TCPStore; on bare metal that defaults to `localhost`, but across
  containers it has to be resolvable DNS, so the `train` profile sets it to
  the `worker0` service name and compose resolves it. `distributed.setup()`
  uses `os.environ.setdefault`, so the env var wins without a code change.
- **`data/` is mounted read-only.** CIFAR-10 is an input; mounting it rw
  invites a container to re-download over a good copy. `results/` is
  bind-mounted rw so the JSONL evidence every earlier phase depends on stays
  inspectable from the host.
- **`TRAIN_RUN`** parameterizes the train profile's output dir, which is how
  `verify_docker.sh` runs the same profile twice to different directories to
  prove in-container determinism.
