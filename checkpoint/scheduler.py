"""
Adaptive checkpoint scheduler.

Replaces Phase 2's fixed interval. The interval is recomputed from three
measured quantities, so it tracks what the system is actually doing rather
than what was guessed at launch.

================================================================================
THE MATH
================================================================================

Symbols, all in seconds unless noted:

    d       checkpoint write cost (delta)
    M       mean time between failures (MTBF)
    R       mean recovery cost: detect + teardown + restart + reload
    T       the decision variable — seconds of useful compute between
            checkpoints
    lambda  failure rate, = 1/M

Consider one cycle: compute for T seconds, then spend d writing a
checkpoint.

**Checkpointing cost.** You pay d once per T seconds of useful work, so as a
fraction of useful work that's `d / T`. Checkpoint more often and this grows.

**Failure cost.** Failures arrive as a Poisson process with rate 1/M. In T
seconds of work the expected number of failures is T/M. When one hits, the
work since the last checkpoint is lost. A failure lands uniformly within the
interval, so the expected loss is the interval's midpoint, `T/2`. Then you
pay R to get going again. Expected failure cost per cycle is therefore
`(T/M) * (T/2 + R)`, or as a fraction of useful work, `(T/2 + R) / M`.

Total expected waste, as a fraction of useful compute:

    W(T) = d/T  +  T/(2M)  +  R/M
           ^^^^     ^^^^^^     ^^^
           writes   lost work  restarts

This is exactly the shape the project plan describes -- `failure_rate x
avg_time_since_last_checkpoint + checkpoint_overhead` -- with
avg_time_since_last_checkpoint made explicit as T/2.

**Minimize.** The two T-dependent terms pull against each other, so there's
an interior optimum:

    dW/dT = -d/T^2 + 1/(2M) = 0
    =>  T^2 = 2dM
    =>  T* = sqrt(2 * d * M)

That's Young's formula (Young 1974). W is convex on T > 0 (W'' = 2d/T^3 > 0),
so this is a minimum, not a saddle.

**Sanity checks on the shape.** Expensive checkpoints (large d) push the
interval up; frequent failures (small M) pull it down; and it's a square
root, so a 100x worse failure rate only shortens the interval 10x. Both
knobs have to move a long way to move the interval much, which is why
guessing a fixed interval "feels fine" right up until it isn't.

**R is not a term in the optimality condition.** This surprises people, so
it's worth being explicit: `R/M` is constant in T, so it shifts the waste
curve up without moving its minimum. Restarts being slow makes failures more
expensive but does *not* argue for checkpointing more often -- the thing you
control is how much work is at risk, and that's T/2 alone.

Note carefully what that does and does not say. R is not a term in the
closed form, but R *does* affect the interval, because it affects the
estimate of M -- see "Downtime is not time at risk" below. Those are two
different mechanisms and it's easy to conflate them: one is about the shape
of the objective, the other about measuring its inputs.

Daly (2006) refines Young to `T* = sqrt(2d(M+R)) - d`, accounting for
failures that strike during the restart itself. That's deliberately not used:
it's a second-order correction, and keeping the implemented formula the exact
analytic argmin of the W(T) above means the two can be checked against each
other. `tests/test_scheduler.py` does exactly that -- it grid-searches W
numerically and asserts the closed form agrees.

================================================================================
ESTIMATING THE INPUTS
================================================================================

**d** -- rolling mean of observed checkpoint write durations.
`save_checkpoint` already returns them.

**throughput** -- rolling mean steps/sec, used only to convert T* from
seconds into the step count the training loop actually counts in.

**M** -- this is the interesting one. A worker cannot observe its own death,
so failure history is rebuilt from the job's `failure_log.jsonl` on startup:
the process that would have held it in memory is the one that died.

Given failure timestamps, M is estimated from the recent inter-arrival gaps
*plus* the in-progress gap since the last failure. Including that last,
unfinished interval matters in both directions:

  - Without it, a job that has stabilised keeps checkpointing at the old
    crisis rate forever, because history says failures were frequent.
  - With zero failures observed there are no gaps at all, and the
    time-since-start is the only evidence available. It grows as the system
    proves itself, so the interval lengthens the longer nothing goes wrong.

Counting an unfinished interval as if it were finished underestimates M,
which overestimates the failure rate, which checkpoints slightly too often.
That bias is deliberate: erring toward durability is the cheaper mistake.

**Downtime is not time at risk.** The gap between two failure timestamps is
wall-clock, and it contains a recovery: detection, teardown, relaunch,
checkpoint reload. A worker cannot fail while it is not running, so that
downtime is not exposure. The hazard rate we want is failures per unit of
*training* time, so exactly one recovery cost R is subtracted from each gap.

This matters more than it sounds. Measured on this testbed under a high
failure rate, R was ~3.4s against a wall-clock gap of ~8.0s -- so 42% of the
apparent time-between-failures was actually the system being restarted.
Leaving it in inflates M, which lengthens the interval, which risks more work
under exactly the conditions where work is most at risk.
"""
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, Iterable, List, Optional

# Defaults chosen to be inert: with no observations the scheduler returns the
# interval it was constructed with, so a job that never measures anything
# behaves exactly like Phase 2's fixed interval.
DEFAULT_WINDOW = 20
DEFAULT_MTBF_FLOOR = 0.5


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


@dataclass
class SchedulerDecision:
    """
    One interval decision, with every input that produced it.

    Logged verbatim so a chosen interval can be audited after the fact —
    "why did it pick 3?" should be answerable from the record, not by
    re-deriving it.
    """
    global_step: int
    interval_steps: int
    interval_seconds: float
    optimal_seconds: float
    checkpoint_cost: float
    mtbf: float
    recovery_cost: float
    throughput: float
    expected_waste: float
    failures_observed: int
    clamped: str = ""
    reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class CheckpointScheduler:
    """
    Tracks checkpoint cost, failure rate and throughput; decides intervals.

    Deliberately knows nothing about the configured failure rate. It only
    ever sees measurements — write durations, failure timestamps, step
    durations — which is what makes "shorter interval under a high failure
    rate" an emergent result rather than a hardcoded branch.
    """

    def __init__(self, initial_interval_steps: int,
                 window: int = DEFAULT_WINDOW,
                 min_interval_steps: int = 1,
                 max_interval_steps: int = 10_000,
                 mtbf_floor: float = DEFAULT_MTBF_FLOOR,
                 started_at: Optional[float] = None):
        if initial_interval_steps < 1:
            raise ValueError("initial_interval_steps must be >= 1")
        if min_interval_steps < 1:
            raise ValueError("min_interval_steps must be >= 1")
        if max_interval_steps < min_interval_steps:
            raise ValueError("max_interval_steps < min_interval_steps")

        self.initial_interval_steps = initial_interval_steps
        self.min_interval_steps = min_interval_steps
        self.max_interval_steps = max_interval_steps
        self.window = window
        # Guards against a divide-by-zero and against a pathological
        # "checkpoint every step forever" if two failures land microseconds
        # apart (e.g. a whole world dying at once being logged per-rank).
        self.mtbf_floor = mtbf_floor

        self.started_at = started_at
        self._checkpoint_costs: Deque[float] = deque(maxlen=window)
        self._step_durations: Deque[float] = deque(maxlen=window)
        self._recovery_costs: Deque[float] = deque(maxlen=window)
        self._failure_times: List[float] = []

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def record_checkpoint(self, duration_seconds: float) -> None:
        if duration_seconds is not None and duration_seconds >= 0:
            self._checkpoint_costs.append(float(duration_seconds))

    def record_step(self, duration_seconds: float) -> None:
        if duration_seconds is not None and duration_seconds > 0:
            self._step_durations.append(float(duration_seconds))

    def record_recovery(self, seconds: float) -> None:
        if seconds is not None and seconds >= 0:
            self._recovery_costs.append(float(seconds))

    def observe_failures(self, timestamps: Iterable[float]) -> None:
        """Absolute epoch timestamps of observed failures, in any order."""
        self._failure_times = sorted(float(t) for t in timestamps)

    # ------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------

    def checkpoint_cost(self) -> float:
        """d — rolling mean checkpoint write seconds."""
        return _mean(self._checkpoint_costs)

    def throughput(self) -> float:
        """Steps per second — rolling mean, inverted."""
        mean_step = _mean(self._step_durations)
        return 1.0 / mean_step if mean_step > 0 else 0.0

    def recovery_cost(self) -> float:
        """R — rolling mean seconds from failure detected to training resumed."""
        return _mean(self._recovery_costs)

    @property
    def failures_observed(self) -> int:
        return len(self._failure_times)

    def mtbf(self, now: float) -> float:
        """
        M — mean time between failures, per unit of TRAINING time.

        Combines completed inter-arrival gaps with the in-progress gap since
        the most recent failure (or since the job started, if there hasn't
        been one). See the module docstring for why the unfinished interval
        is counted.

        Each gap has one recovery cost removed from it. Consecutive failures
        are separated by exactly one recovery, and a worker cannot fail while
        it is being restarted, so that downtime is not exposure to failure.
        Leaving it in inflates M and lengthens the interval under precisely
        the conditions where work is most at risk.
        """
        downtime = self.recovery_cost()

        gaps = [(b - a) - downtime
                for a, b in zip(self._failure_times, self._failure_times[1:])]

        if self._failure_times:
            # One recovery also sits inside the in-progress gap.
            censored = (now - self._failure_times[-1]) - downtime
        elif self.started_at is not None:
            # No failure yet means no recovery yet, so nothing to subtract.
            censored = now - self.started_at
        else:
            censored = 0.0

        observations = [g for g in gaps if g > 0]
        if censored > 0:
            observations.append(censored)

        return max(_mean(observations), self.mtbf_floor)

    # ------------------------------------------------------------------
    # The decision
    # ------------------------------------------------------------------

    def expected_waste(self, interval_seconds: float, now: float) -> float:
        """
        W(T) — expected wasted compute as a fraction of useful compute.

        W(T) = d/T + T/(2M) + R/M
        """
        if interval_seconds <= 0:
            return math.inf
        d = self.checkpoint_cost()
        m = self.mtbf(now)
        r = self.recovery_cost()
        return d / interval_seconds + interval_seconds / (2 * m) + r / m

    def optimal_interval_seconds(self, now: float) -> float:
        """T* = sqrt(2 d M). Zero if checkpoints appear to be free."""
        d = self.checkpoint_cost()
        if d <= 0:
            return 0.0
        return math.sqrt(2.0 * d * self.mtbf(now))

    def decide(self, now: float, global_step: int = 0) -> SchedulerDecision:
        """Recompute the interval and return it along with its inputs."""
        d = self.checkpoint_cost()
        tput = self.throughput()
        m = self.mtbf(now)
        r = self.recovery_cost()
        optimal = self.optimal_interval_seconds(now)

        clamped, reason = "", ""

        if d <= 0 or tput <= 0:
            # Nothing measured yet — behave exactly like the fixed interval.
            steps = self.initial_interval_steps
            reason = ("no checkpoint cost measured yet" if d <= 0
                      else "no throughput measured yet")
        else:
            steps = int(round(optimal * tput))
            reason = "T* = sqrt(2 d M), converted to steps by throughput"

        if steps < self.min_interval_steps:
            steps, clamped = self.min_interval_steps, "min"
        elif steps > self.max_interval_steps:
            steps, clamped = self.max_interval_steps, "max"

        interval_seconds = steps / tput if tput > 0 else 0.0

        return SchedulerDecision(
            global_step=global_step,
            interval_steps=steps,
            interval_seconds=interval_seconds,
            optimal_seconds=optimal,
            checkpoint_cost=d,
            mtbf=m,
            recovery_cost=r,
            throughput=tput,
            expected_waste=(self.expected_waste(interval_seconds, now)
                            if interval_seconds > 0 else math.inf),
            failures_observed=self.failures_observed,
            clamped=clamped,
            reason=reason,
        )

    def next_interval_steps(self, now: float) -> int:
        return self.decide(now).interval_steps


# --------------------------------------------------------------------------
# Rebuilding state from a crashed job's logs
# --------------------------------------------------------------------------

def failure_times_from_log(failure_log: str) -> List[float]:
    """
    Timestamps of every injected failure recorded for a job.

    A worker cannot witness its own death, so this is how the scheduler
    learns the failure history that a previous generation experienced.
    """
    import json
    import os

    if not os.path.exists(failure_log):
        return []

    times: List[float] = []
    with open(failure_log) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated final line from the worker that died
            if record.get("event") == "injected_failure" and "timestamp" in record:
                times.append(float(record["timestamp"]))
    return sorted(times)


def scheduler_from_job(job_dir: str, initial_interval_steps: int,
                       started_at: Optional[float] = None,
                       **kwargs) -> CheckpointScheduler:
    """
    Build a scheduler primed with everything a job's logs already know.

    Called at worker startup so a restarted generation inherits the failure
    history and recovery costs of the ones before it, instead of starting
    over as an optimist every time it crashes.
    """
    import os

    from metrics.recovery_timing import load_recovery_timings

    scheduler = CheckpointScheduler(
        initial_interval_steps=initial_interval_steps,
        started_at=started_at, **kwargs)

    failures = failure_times_from_log(
        os.path.join(job_dir, "failure_log.jsonl"))
    scheduler.observe_failures(failures)

    # The very first failure of a job has no preceding gap, so without an
    # anchor a one-failure job would have no completed interval at all.
    if failures and scheduler.started_at is None:
        scheduler.started_at = failures[0]

    for timing in load_recovery_timings(
            os.path.join(job_dir, "recovery_log.jsonl")):
        scheduler.record_recovery(timing.recovery_seconds)

    return scheduler
