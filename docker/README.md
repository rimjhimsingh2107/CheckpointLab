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

---

# Monitoring: Prometheus + Grafana

```bash
# 1. bring up the monitoring stack (long-lived)
docker compose -f docker/docker-compose.yml --profile monitor up -d

# 2. generate some activity alongside it
docker compose -f docker/docker-compose.yml --profile recover run --rm controller \
  python controller/recovery.py \
    --job-dir=/app/results/my-run --world-size=2 --epochs=8 \
    --batch-size=32 --subset-size=1024 --max-steps=30 \
    --checkpoint-every=4 --adaptive-checkpointing \
    --failure-rate=0.15 --failure-seed=11 --max-restarts=20 --log-every=1000

# 3. look at it
open http://localhost:3000        # Grafana, anonymous admin, lands on the dashboard
open http://localhost:9090        # Prometheus
curl localhost:9101/metrics       # the exporter directly

# 4. verify the whole chain instead of trusting it
bash docker/verify_monitoring.sh
```

| service | port | image |
|---|---|---|
| `exporter` | 9101 | `checkpointlab:local` (`python -m metrics.prometheus`) |
| `prometheus` | 9090 | `prom/prometheus:v3.4.1` |
| `grafana` | 3000 | `grafana/grafana:11.6.0` |

## Why the exporter is a sidecar, not in-process instrumentation

The obvious design is to import `prometheus_client` into `training/train.py`
and increment counters as steps complete. That is wrong *for this project*:
the workers are killed on purpose, repeatedly, and a scrape target has to be
there when Prometheus arrives. An in-process exporter would vanish mid-run on
every injected failure, and the scrape history would be target-down gaps
punctuated by counters resetting to zero on each restart.

So `metrics/prometheus.py` is a long-lived process that re-reads each job's
durable JSONL on every scrape. Same insight Phase 5's scheduler rests on: the
process that made an observation is the one that died, so the log on disk is
the only source of truth that survives. It pays off twice here — the metrics
outlive the workers, and counters stay monotonic across generations because
they are recomputed from the whole log rather than accumulated in memory.
A scrape costs ~3 ms (exposed as `checkpointlab_exporter_scrape_duration_seconds`),
which is why `scrape_interval` can be 2s.

## Verified, on this machine

`bash docker/verify_monitoring.sh` — all five checks pass:

```
 1/5  EXPORTER          226 checkpointlab samples, 8 jobs discovered
 2/5  PROMETHEUS TARGET checkpointlab health=up err=''
 3/5  SERIES PRESENT    ckpt_job values: [... 8 jobs ...]
 4/5  GRAFANA           11.6.0 database=ok; datasource -> http://prometheus:9090
                        dashboard 'CheckpointLab — ...': 12 panels
 5/5  GRAFANA QUERIES   throughput/interval/recovery/active_failures: 8 series each
```

Check 5 is the one that matters. A provisioned datasource that cannot reach
Prometheus, or a dashboard whose selectors don't match the exported labels,
both present as an empty panel with **no error anywhere**. Querying through
Grafana's own datasource proxy is the only way to prove the path a panel
actually takes.

### Activity used for the screenshots

A containerized adaptive run, 30 steps, `--failure-rate=0.15 --failure-seed=11`
— 5 scheduled failures at steps 8, 10, 11, 15, 22. All 5 fired and all 5
recovered; the job completed with both ranks exiting 0:

```
   #  rank   kind   died@  restored@  lost  recovery s
   1     1   exit       8          7     1        7.66
   2     0   exit      10          9     1        6.51
   3     0   exit      11         10     1        6.50
   4     0   exit      15         14     1        7.10
   5     1   exit      22         21     1        8.63
```

### What the dashboard actually shows

Viewed in a browser at `localhost:3000`, job filtered to this run. Described
rather than embedded as a PNG — see the caveat below.

- **Stat row** — Active failures `0` (green, with visible red spikes in the
  sparkline where recoveries were in flight), Throughput `0.70`, Last recovery
  `5.82 s`, Checkpoint interval `1`, Recoveries `5`.
- **Throughput over time** — two series on purpose. `reported` is the
  exporter's rolling mean of logged step durations; `observed` is
  `rate(checkpointlab_steps_completed_total[30s])`. They diverge in a way that
  is worth seeing: the rolling mean degrades gently and then *freezes* at its
  last value when the job ends, because a dead worker stops appending to the
  log rather than writing zeros. The rate falls to a real zero. Read the gauge
  for "how fast is a step", the rate for "is work happening at all".
- **Checkpoint interval over time** — two series. Interval-in-steps (green,
  stepped) starts at 4, the launch value, then sits at 1. T\* in seconds
  (yellow) swings 2.9 → 1.2 and oscillates around 1.4 as the scheduler
  re-estimates. Seeing them disagree is the point: it is the integer
  quantization from Phase 5, on screen.
- **Recovery duration split by phase** — `reach_first_step` (yellow) fills
  5.5–6.5 s while the other three phases sit flat near zero. This is the
  Phase 6 finding rendered: ~96% of recovery is process startup, not
  controller logic. A panel showing only the total would send you optimizing
  the wrong thing.
- **Useful vs recomputed work** — useful steps climbing to 30, recomputed
  climbing to 5, one per failure.

### Caveats, stated rather than glossed

- **PNGs are rendered server-side, not screenshotted.** The `monitor` profile
  includes a `grafana-image-renderer:3.12.9` service, so panels are captured
  reproducibly with `curl`:

  ```bash
  curl "localhost:3000/render/d-solo/checkpointlab/?orgId=1&panelId=6\
&var-job=<job>&from=now-12m&to=now&width=1100&height=420&theme=light&scale=2" \
    -o docs/grafana-throughput.png
  ```

  `panelId` 6 = throughput, 7 = checkpoint interval, 8 = recovery phases.
  Committed examples are in [`docs/`](../docs).
- **Gauges freeze when a job ends, they do not fall to zero.** The exporter
  reports the last value in the log, and a finished job's log stops changing —
  so throughput reads `0.70` forever after completion rather than dropping.
  `checkpointlab_job_active` is the metric that distinguishes running from
  finished; filter on it before reading a gauge as "current".
- **The interval panel shows little variation in-container.** Not a bug:
  containerized throughput is ~0.4–0.7 steps/sec (two ranks plus the
  controller sharing one container in a 4 GB VM), so a perfectly reasonable
  T\* of 1–3 seconds converts to ~1 step. The `clamped` field in
  `scheduler_log.jsonl` reads `-`, confirming the floor is not being hit —
  the arithmetic really does land at 1. Interval variation is visible on bare
  metal, where throughput is ~1.0 steps/sec (Phase 5 shows 2 → 3 → 4).
- **Anonymous admin is on.** Local demo stack, bridge network, nothing worth
  protecting, and a login wall is the main reason nobody looks at a dashboard.
  Do not copy that setting anywhere reachable.

## A bug worth recording: `job` is a reserved label

The first version labelled every metric `job="<job dir>"`. Prometheus
attaches its own `job` label from the scrape config, and with `honor_labels`
at its default of `false` the *exporter's* version loses — it is silently
renamed to `exported_job`. The result:

- the scrape succeeded,
- the target reported `health=up` with `lastError=''`,
- Prometheus held all 27 metric names,
- and every dashboard panel would have rendered **No data**, because
  `job=~"$job"` matched `job="checkpointlab"` rather than the training job.

Nothing anywhere reported an error. The fix was renaming the dimension to
`ckpt_job` rather than setting `honor_labels: true` — the latter works but
discards the scrape-job name, and `job` is reserved by convention for exactly
that. `tests/test_prometheus_exporter.py` pins it so it cannot regress.
