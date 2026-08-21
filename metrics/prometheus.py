"""
Prometheus exporter for CheckpointLab.

WHY THIS IS A SIDECAR AND NOT INSTRUMENTATION INSIDE THE TRAINING LOOP

The obvious design is to import prometheus_client into training/train.py and
increment counters as steps complete. That is wrong for this project
specifically: the workers are *killed on purpose*, repeatedly, and a
Prometheus scrape target has to be there when Prometheus comes knocking. An
in-process exporter would vanish mid-run every time a failure fired, and the
scrape history would be a sequence of target-down gaps punctuated by counters
that reset to zero on every restart -- which is exactly the signal you cannot
reason about.

So the exporter is a long-lived process that re-reads the job's durable JSONL
logs on every scrape. That is the same insight Phase 5's scheduler rests on:
the process that made an observation is the one that died, so the log on disk
is the only source of truth that survives. Here it pays off twice -- the
metrics outlive the workers, and counters stay monotonic across generations
because they are recomputed from the whole log rather than accumulated in
memory.

IMPLEMENTED AS A CUSTOM COLLECTOR

Metrics are derived from files at scrape time, not accumulated by calls to
inc(). Using module-level Counter/Gauge objects would mean tracking deltas
and calling inc() by the difference, which invents state the log already
holds. A custom collector that computes CounterMetricFamily/GaugeMetricFamily
on demand is the idiomatic fit, and it keeps `_total` names honestly typed as
counters.

Usage:
    python metrics/prometheus.py --once                 # print exposition, exit
    python metrics/prometheus.py --port 9101            # serve /metrics
    python metrics/prometheus.py --results-root results --port 9101
"""
import argparse
import json
import os
import time
from typing import Dict, Iterable, List, Optional

from prometheus_client import REGISTRY, generate_latest, start_http_server
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

from metrics.recovery_timing import PHASES, load_recovery_timings, summarize

# A job directory is anything that has produced at least one of these.
JOB_MARKERS = ("recovery_log.jsonl", "loss_log.jsonl", "checkpoint_log.jsonl")

# Throughput is a rate, so it needs a window. Whole-run mean would smear the
# post-recovery ramp into the steady state and hide exactly the transient the
# dashboard exists to show.
THROUGHPUT_WINDOW = 20

PREFIX = "checkpointlab"


def _read_jsonl(path: str) -> List[dict]:
    """Tolerant reader -- a killed worker can leave a torn final line."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def discover_jobs(results_root: str) -> List[str]:
    """Every job directory under `results_root`, newest first."""
    if not os.path.isdir(results_root):
        return []
    found = []
    for entry in sorted(os.listdir(results_root)):
        path = os.path.join(results_root, entry)
        if os.path.isdir(path) and any(
                os.path.exists(os.path.join(path, m)) for m in JOB_MARKERS):
            found.append(path)
    return found


class JobSnapshot:
    """Everything the exporter knows about one job, read fresh."""

    def __init__(self, job_dir: str):
        self.job_dir = job_dir
        self.name = os.path.basename(os.path.normpath(job_dir))

        loss = _read_jsonl(os.path.join(job_dir, "loss_log.jsonl"))
        ckpt = _read_jsonl(os.path.join(job_dir, "checkpoint_log.jsonl"))
        sched = _read_jsonl(os.path.join(job_dir, "scheduler_log.jsonl"))
        fails = _read_jsonl(os.path.join(job_dir, "failure_log.jsonl"))
        recov = _read_jsonl(os.path.join(job_dir, "recovery_log.jsonl"))

        # ---- steps: useful vs recomputed --------------------------------
        # A global_step appearing twice was computed twice. Same definition
        # the Phase 6 benchmark uses, so the dashboard and the benchmark
        # cannot disagree about what "wasted" means.
        seen = set()
        self.steps_recomputed = 0
        for r in loss:
            s = r.get("global_step")
            if s in seen:
                self.steps_recomputed += 1
            else:
                seen.add(s)
        self.steps_completed = len(seen)
        self.last_step = max(seen) if seen else 0

        recent = [float(r["step_seconds"]) for r in loss[-THROUGHPUT_WINDOW:]
                  if r.get("step_seconds")]
        mean_step = sum(recent) / len(recent) if recent else 0.0
        self.throughput = 1.0 / mean_step if mean_step > 0 else 0.0
        self.last_step_seconds = recent[-1] if recent else 0.0

        # ---- checkpoints -----------------------------------------------
        writes = [r for r in ckpt if r.get("event") == "checkpoint"]
        replays = [r for r in ckpt if r.get("event") == "fast_forward"]
        self.checkpoints_written = len(writes)
        self.checkpoint_seconds_total = sum(
            float(r.get("duration_seconds") or 0.0) for r in writes)
        self.last_checkpoint_seconds = (
            float(writes[-1].get("duration_seconds") or 0.0) if writes else 0.0)
        self.fast_forward_seconds_total = sum(
            float(r.get("duration_seconds") or 0.0) for r in replays)

        # Interval actually in effect. Phase 6 learned the hard way that the
        # computed interval and the one in effect can differ, so this reports
        # what the loop is really doing.
        self.interval_steps = int(writes[-1].get("interval_used") or 0) if writes else 0

        # ---- scheduler (present only under --adaptive-checkpointing) ----
        self.has_scheduler = bool(sched)
        last_sched = sched[-1] if sched else {}
        self.optimal_interval_seconds = float(last_sched.get("optimal_seconds") or 0.0)
        self.mtbf_seconds = float(last_sched.get("mtbf") or 0.0)
        self.recovery_cost_estimate = float(last_sched.get("recovery_cost") or 0.0)
        waste = last_sched.get("expected_waste")
        # expected_waste is inf before anything is measured; inf serialises as
        # +Inf which Prometheus accepts but which poisons dashboard autoscaling.
        self.expected_waste = (float(waste)
                               if isinstance(waste, (int, float))
                               and float(waste) == float(waste)
                               and abs(float(waste)) != float("inf") else 0.0)
        self.decisions = len(sched)

        # ---- failures and recoveries ------------------------------------
        self.failures_injected = sum(
            1 for r in fails if r.get("event") == "injected_failure")

        events = [r.get("event") for r in recov]
        self.detected = events.count("failure_detected")
        self.resumed = events.count("training_resumed")
        self.generations = events.count("generation_launched")

        # A failure that has been detected but not yet resumed is a recovery
        # in flight. This is the one metric that is meaningfully non-zero only
        # DURING an incident, which makes it the useful alerting signal.
        self.active_failures = max(0, self.detected - self.resumed)

        timings = load_recovery_timings(os.path.join(job_dir, "recovery_log.jsonl"))
        stats = summarize(timings)
        self.recoveries = stats["recoveries"]
        self.recovery_seconds_total = stats["total_seconds"]
        self.recovery_mean_seconds = stats["mean_seconds"]
        self.last_recovery_seconds = (timings[-1].recovery_seconds
                                      if timings else 0.0)
        # Reuses the Phase 4/6 breakdown type rather than re-deriving it.
        self.last_recovery_phases: Dict[str, float] = (
            timings[-1].breakdown if timings else {p: 0.0 for p in PHASES})
        self.steps_lost = stats["steps_lost"]

        finished = [r for r in recov if r.get("event") == "job_finished"]
        self.active = 1.0 if (recov and not finished) else 0.0
        self.outcome = finished[-1].get("outcome") if finished else "running"


class CheckpointLabCollector:
    """Computes the whole metric set from disk on every scrape."""

    def __init__(self, results_root: str):
        self.results_root = results_root

    def collect(self) -> Iterable:
        started = time.perf_counter()
        jobs = [JobSnapshot(d) for d in discover_jobs(self.results_root)]

        # NOT "job". `job` is reserved by Prometheus for the scrape-job name
        # from prometheus.yml, and an exporter that also exports `job` gets
        # its version silently renamed to `exported_job` on ingest (because
        # honor_labels defaults to false). The scrape still succeeds, the
        # target still reads healthy, and every dashboard selector written
        # against `job` matches nothing -- which presents as "No data" with
        # no error anywhere to explain it. Naming the dimension `ckpt_job`
        # sidesteps the collision instead of fighting it with honor_labels,
        # which would work but would throw away the scrape-job name.
        L = ["ckpt_job"]

        def counter(name, doc, pick, labels=L):
            m = CounterMetricFamily(f"{PREFIX}_{name}", doc, labels=labels)
            for j in jobs:
                m.add_metric([j.name], pick(j))
            return m

        def gauge(name, doc, pick, labels=L):
            m = GaugeMetricFamily(f"{PREFIX}_{name}", doc, labels=labels)
            for j in jobs:
                m.add_metric([j.name], pick(j))
            return m

        # ---- counters (monotonic within a job) --------------------------
        yield counter("steps_completed", "Distinct training steps that survived into the result.", lambda j: j.steps_completed)
        yield counter("steps_recomputed", "Steps computed more than once because a crash discarded them.", lambda j: j.steps_recomputed)
        yield counter("checkpoints_written", "Checkpoints successfully written.", lambda j: j.checkpoints_written)
        yield counter("checkpoint_seconds", "Cumulative time spent writing checkpoints.", lambda j: j.checkpoint_seconds_total)
        yield counter("fast_forward_seconds", "Cumulative time replaying a resumed epoch's dataloader.", lambda j: j.fast_forward_seconds_total)
        yield counter("failures_injected", "Worker failures injected by the Phase 3 injector.", lambda j: j.failures_injected)
        yield counter("recoveries", "Completed crash-detect-restore-resume cycles.", lambda j: j.recoveries)
        yield counter("recovery_seconds", "Cumulative time from failure detected to training resumed.", lambda j: j.recovery_seconds_total)
        yield counter("generations_launched", "Worker generations launched (1 + number of restarts).", lambda j: j.generations)
        yield counter("steps_lost", "Steps of work discarded by failures.", lambda j: j.steps_lost)

        # ---- gauges -----------------------------------------------------
        yield gauge("throughput_steps_per_second", f"Training throughput over the last {THROUGHPUT_WINDOW} steps.", lambda j: j.throughput)
        yield gauge("step_duration_seconds", "Duration of the most recent training step.", lambda j: j.last_step_seconds)
        yield gauge("checkpoint_duration_seconds", "Duration of the most recent checkpoint write.", lambda j: j.last_checkpoint_seconds)
        yield gauge("checkpoint_interval_steps", "Checkpoint interval currently in effect, in steps.", lambda j: j.interval_steps)
        yield gauge("recovery_duration_seconds", "Duration of the most recent recovery.", lambda j: j.last_recovery_seconds)
        yield gauge("recovery_duration_mean_seconds", "Mean recovery duration for this job.", lambda j: j.recovery_mean_seconds)
        yield gauge("active_failures", "Failures detected but not yet recovered (a recovery in flight).", lambda j: j.active_failures)
        yield gauge("last_step", "Highest global step reached.", lambda j: j.last_step)
        yield gauge("job_active", "1 while a job is running, 0 once it has finished.", lambda j: j.active)

        # ---- scheduler internals (adaptive runs only) -------------------
        yield gauge("scheduler_optimal_interval_seconds", "T* = sqrt(2 d M), the scheduler's target interval.", lambda j: j.optimal_interval_seconds)
        yield gauge("scheduler_mtbf_seconds", "Scheduler's current mean-time-between-failures estimate.", lambda j: j.mtbf_seconds)
        yield gauge("scheduler_recovery_cost_seconds", "Scheduler's current recovery-cost estimate R.", lambda j: j.recovery_cost_estimate)
        yield gauge("scheduler_expected_waste_ratio", "W(T) -- expected wasted compute as a fraction of useful compute.", lambda j: j.expected_waste)
        yield gauge("scheduler_decisions", "Interval decisions the scheduler has recorded.", lambda j: j.decisions)

        # ---- recovery breakdown, labelled by phase ----------------------
        # Phase 6 showed the total is ~97% process startup; a dashboard that
        # only plotted the total would send you optimising the wrong thing.
        phases = GaugeMetricFamily(
            f"{PREFIX}_recovery_phase_seconds",
            "Most recent recovery's duration, split by phase.",
            labels=["ckpt_job", "phase"])
        for j in jobs:
            for phase in PHASES:
                phases.add_metric([j.name, phase],
                                  j.last_recovery_phases.get(phase, 0.0))
        yield phases

        # ---- exporter self-observability --------------------------------
        yield GaugeMetricFamily(f"{PREFIX}_exporter_jobs_discovered",
                                "Job directories the exporter is reading.",
                                value=float(len(jobs)))
        yield GaugeMetricFamily(f"{PREFIX}_exporter_scrape_duration_seconds",
                                "Time spent building this scrape.",
                                value=time.perf_counter() - started)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="results",
                   help="Directory containing job subdirectories")
    p.add_argument("--port", type=int, default=9101)
    p.add_argument("--once", action="store_true",
                   help="Print one exposition to stdout and exit (no server)")
    return p.parse_args(argv)


def build_registry(results_root: str):
    """A fresh registry, so the default process/GC collectors stay out."""
    from prometheus_client import CollectorRegistry
    reg = CollectorRegistry()
    reg.register(CheckpointLabCollector(results_root))
    return reg


def main(argv=None) -> int:
    args = parse_args(argv)
    root = os.path.abspath(args.results_root)

    if args.once:
        print(generate_latest(build_registry(root)).decode(), end="")
        return 0

    jobs = discover_jobs(root)
    print(f"CheckpointLab exporter: results_root={root}", flush=True)
    print(f"  {len(jobs)} job dir(s) at startup: "
          f"{[os.path.basename(j) for j in jobs]}", flush=True)
    print(f"  jobs are re-discovered on every scrape, so a job created after "
          f"startup appears without a restart", flush=True)

    start_http_server(args.port, registry=build_registry(root))
    print(f"  serving /metrics on :{args.port}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    raise SystemExit(main())
