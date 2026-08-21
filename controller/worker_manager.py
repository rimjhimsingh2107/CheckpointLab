"""
Worker process spawning and liveness monitoring.

Phases 1-3 had `train.py` spawn its own ranks with `mp.spawn` and join them.
That's fine for a single job, but it can't support recovery: `mp.spawn`
tears the whole process group down on the first failure, and there's nobody
left to notice or restart anything. So the controller takes over the role a
real cluster launcher plays — it spawns one OS process per rank and watches
them individually, which is what makes "worker 1 died, restart the job from
its last checkpoint" expressible at all.

Two detectors, because DDP fails in two different shapes:

  * **Process exit.** The killed worker dies immediately and `poll()` sees a
    non-zero exit code within a poll interval. This is the fast path.

  * **Stale heartbeat.** The *survivors* of a peer's death usually don't
    die — they block inside a collective waiting for a rank that's never
    coming back. gloo's default collective timeout is 30 minutes, so a job
    that lost one worker would sit there burning wall-clock for half an hour
    looking superficially healthy. Each worker touches a heartbeat file, and
    a rank whose heartbeat goes stale is treated as failed even though its
    process is alive.

The second detector is why the controller kills survivors rather than
waiting for them: after any failure the whole group has to come down before
it can be relaunched, because the checkpoint is the only consistent state
they can all agree on.
"""
import glob
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(REPO_ROOT, "training", "train.py")


# --------------------------------------------------------------------------
# Heartbeats — written by workers (training/train.py), read by the controller
# --------------------------------------------------------------------------

def heartbeat_path(heartbeat_dir: str, rank: int) -> str:
    return os.path.join(heartbeat_dir, f"rank{rank}.json")


def write_heartbeat(heartbeat_dir: str, rank: int, **fields) -> None:
    """
    Record that this rank is alive and how far it has got.

    Written tmp-then-rename so the controller can never read a half-written
    file: a torn heartbeat would look like a corrupt or missing one and
    trigger a spurious recovery, which is worse than a slightly stale
    reading.
    """
    os.makedirs(heartbeat_dir, exist_ok=True)
    payload = {"rank": rank, "pid": os.getpid(), "timestamp": time.time()}
    payload.update(fields)

    final = heartbeat_path(heartbeat_dir, rank)
    tmp = f"{final}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, final)


def read_heartbeat(heartbeat_dir: str, rank: int) -> Optional[dict]:
    try:
        with open(heartbeat_path(heartbeat_dir, rank)) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def clear_heartbeats(heartbeat_dir: str) -> None:
    for path in glob.glob(os.path.join(heartbeat_dir, "rank*.json")):
        try:
            os.remove(path)
        except OSError:
            pass


def pick_free_port() -> int:
    """
    A fresh rendezvous port for each generation of workers.

    Reusing one port across restarts is how this breaks in practice: the
    dead generation's listener is still in TIME_WAIT and the replacement
    job dies with EADDRINUSE before it can even load its checkpoint.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


# --------------------------------------------------------------------------
# Monitoring
# --------------------------------------------------------------------------

@dataclass
class WorkerFailure:
    """What the controller observed, and when it noticed."""
    rank: int
    kind: str                      # "exit" | "stale_heartbeat"
    detected_at: float
    exit_code: Optional[int] = None
    last_step: Optional[int] = None
    detail: str = ""

    def describe(self) -> str:
        if self.kind == "exit":
            return (f"worker {self.rank} exited {self.exit_code} "
                    f"(last reported step {self.last_step})")
        return (f"worker {self.rank} heartbeat stale — {self.detail} "
                f"(last reported step {self.last_step})")


@dataclass
class Worker:
    rank: int
    process: subprocess.Popen
    log_path: str
    launched_at: float
    port: int


class WorkerManager:
    """
    Owns one generation of worker processes at a time.

    A "generation" is one launch of the full world. Every restart is a new
    generation, tagged into heartbeats and log filenames so evidence from a
    dead generation can't be mistaken for the live one.
    """

    def __init__(self, job_dir: str, world_size: int, train_flags: List[str],
                 heartbeat_timeout: float = 60.0,
                 python: str = sys.executable, cwd: str = REPO_ROOT):
        self.job_dir = job_dir
        self.world_size = world_size
        self.train_flags = list(train_flags)
        self.heartbeat_timeout = heartbeat_timeout
        self.python = python
        self.cwd = cwd

        self.heartbeat_dir = os.path.join(job_dir, "heartbeats")
        self.log_dir = os.path.join(job_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.heartbeat_dir, exist_ok=True)

        self.generation = 0
        self.workers: List[Worker] = []
        self._log_handles: List = []

    # ---------------- lifecycle ----------------

    def launch(self, resume: bool, world_size: Optional[int] = None) -> None:
        """
        Start a fresh generation of worker processes.

        `world_size` overrides the manager's current world size, which is what
        makes elastic resumption possible. Nothing about the process group
        needs resizing: each generation calls dist.init_process_group from
        scratch in brand-new processes, so a different world size is just
        different argv. That falls out of the Phase 4 decision to supervise
        ranks as separate OS processes rather than mp.spawn children -- the
        expensive part of elasticity was paid for already, without knowing it.
        """
        if world_size is not None and world_size != self.world_size:
            print(f"     world size {self.world_size} -> {world_size}")
            self.world_size = world_size
        self.generation += 1
        # Heartbeats from the previous generation would otherwise be read as
        # this generation's, making a just-launched worker look stale.
        clear_heartbeats(self.heartbeat_dir)

        port = pick_free_port()
        self.workers = []
        self._close_logs()

        for rank in range(self.world_size):
            cmd = [
                self.python, "-u", TRAIN_SCRIPT,
                "--rank", str(rank),
                "--world-size", str(self.world_size),
                "--master-port", str(port),
                "--generation", str(self.generation),
            ] + self.train_flags
            if not resume:
                cmd.append("--no-resume")

            log_path = os.path.join(
                self.log_dir, f"gen{self.generation}_rank{rank}.log")
            handle = open(log_path, "w")
            self._log_handles.append(handle)

            # start_new_session: the workers get their own process group, so
            # a Ctrl+C aimed at the controller doesn't race the controller's
            # own orderly shutdown of them.
            process = subprocess.Popen(
                cmd, cwd=self.cwd, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True)

            self.workers.append(Worker(rank=rank, process=process,
                                       log_path=log_path,
                                       launched_at=time.time(), port=port))

    def _close_logs(self) -> None:
        for handle in self._log_handles:
            try:
                handle.close()
            except OSError:
                pass
        self._log_handles = []

    # ---------------- observation ----------------

    def heartbeats(self) -> Dict[int, dict]:
        """Current-generation heartbeats, keyed by rank."""
        found = {}
        for rank in range(self.world_size):
            beat = read_heartbeat(self.heartbeat_dir, rank)
            if beat and beat.get("generation") == self.generation:
                found[rank] = beat
        return found

    def max_step_seen(self) -> Optional[int]:
        steps = [b.get("global_step") for b in self.heartbeats().values()
                 if b.get("global_step") is not None]
        return max(steps) if steps else None

    def exit_codes(self) -> Dict[int, Optional[int]]:
        return {w.rank: w.process.poll() for w in self.workers}

    def all_exited_cleanly(self) -> bool:
        codes = self.exit_codes().values()
        return all(code == 0 for code in codes)

    def poll(self) -> Optional[WorkerFailure]:
        """
        Check every worker once. Returns the first failure seen, else None.

        Exit codes are checked before heartbeats: a dead process is
        unambiguous, whereas a stale heartbeat could just be a slow step.
        """
        now = time.time()

        for worker in self.workers:
            code = worker.process.poll()
            if code is not None and code != 0:
                beat = self.heartbeats().get(worker.rank)
                return WorkerFailure(
                    rank=worker.rank, kind="exit", detected_at=now,
                    exit_code=code,
                    last_step=(beat or {}).get("global_step"),
                    detail=f"exit code {code}")

        beats = self.heartbeats()
        for worker in self.workers:
            if worker.process.poll() is not None:
                continue  # already exited; handled above or exited cleanly

            beat = beats.get(worker.rank)
            # Before the first heartbeat lands, judge staleness from launch
            # time — a worker that hangs during startup still has to be
            # caught, but it gets a full timeout of grace to get going.
            reference = beat["timestamp"] if beat else worker.launched_at
            age = now - reference
            if age > self.heartbeat_timeout:
                return WorkerFailure(
                    rank=worker.rank, kind="stale_heartbeat", detected_at=now,
                    last_step=(beat or {}).get("global_step"),
                    detail=(f"no heartbeat for {age:.1f}s "
                            f"(timeout {self.heartbeat_timeout:.0f}s)"))
        return None

    def any_alive(self) -> bool:
        return any(w.process.poll() is None for w in self.workers)

    # ---------------- teardown ----------------

    def terminate_all(self, grace: float = 5.0) -> Dict[int, Optional[int]]:
        """
        Bring the whole generation down.

        SIGTERM to each surviving worker's process group first, then SIGKILL
        anything still standing. The survivors of a peer's death are usually
        blocked in a collective and won't necessarily honour SIGTERM
        promptly, so the SIGKILL escalation isn't optional.
        """
        for worker in self.workers:
            if worker.process.poll() is None:
                try:
                    os.killpg(os.getpgid(worker.process.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass

        deadline = time.time() + grace
        while time.time() < deadline and self.any_alive():
            time.sleep(0.05)

        for worker in self.workers:
            if worker.process.poll() is None:
                try:
                    os.killpg(os.getpgid(worker.process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

        for worker in self.workers:
            try:
                worker.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

        self._close_logs()
        return self.exit_codes()

    def tail_log(self, rank: int, lines: int = 12) -> List[str]:
        """Last few lines of a worker's log, for reporting why it died."""
        for worker in self.workers:
            if worker.rank == rank:
                try:
                    with open(worker.log_path) as fh:
                        return fh.read().splitlines()[-lines:]
                except OSError:
                    return []
        return []
