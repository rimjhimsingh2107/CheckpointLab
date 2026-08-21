"""
Checkpoint correctness verification (Phase 2).

Claiming "we checkpoint and resume" is cheap. The claim worth making is
that resuming is *semantically invisible* — a run that died and restarted
computes exactly what an uninterrupted run would have computed. That is the
property every downstream phase depends on: Phase 6's benchmark compares
checkpoint strategies on wall-clock time and wasted compute, and those
numbers only mean something if all strategies are training the same model.

The experiment:

  Run A (control)   train K steps, uninterrupted, log loss every step.
  Run B (treatment) train the same K steps, but hard-kill the process
                    (os._exit, no cleanup) at step C. Relaunch. It resumes
                    from the last checkpoint, replays the steps lost since
                    that checkpoint, and continues to K.

  Assert: for every step Run B's resumed launch computed, its loss matches
  Run A's loss at that same global step within `--tolerance`.

Note this checks the *replayed* steps too, not just the ones after the
crash point. Those replayed steps are the strictest part of the test — they
are where a stale optimizer buffer or an unrestored RNG shows up first.

Both runs are real subprocesses, so the kill is a real process death rather
than an in-process simulation of one.

Usage:
    python checkpoint/verify.py
    python checkpoint/verify.py --steps 32 --crash-at-step 19 --tolerance 1e-6
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(REPO_ROOT, "training", "train.py")

# Defaults are sized to run in ~1 minute on a CPU-only laptop while still
# exercising every code path that matters: a mid-epoch crash, a replay of
# lost steps, and an epoch boundary crossed after the resume.
DEFAULTS = dict(
    steps=24,
    checkpoint_every=5,
    crash_at_step=13,
    batch_size=32,
    subset_size=1024,
    world_size=2,
    epochs=3,
    seed=1337,
    tolerance=1e-5,
)

PRE_CRASH = "pre-crash"
RESUMED = "resumed"
CONTROL = "control"


class VerificationError(AssertionError):
    """Raised when a resumed run does not reproduce the control trajectory."""


def _train_cmd(args: argparse.Namespace, *, checkpoint_dir: str, loss_log: str,
               launch_id: str, no_resume: bool, crash_at_step: int = 0) -> List[str]:
    cmd = [
        sys.executable, "-u", TRAIN_SCRIPT,
        "--deterministic",
        "--epochs", str(args.epochs),
        "--world-size", str(args.world_size),
        "--batch-size", str(args.batch_size),
        "--subset-size", str(args.subset_size),
        "--seed", str(args.seed),
        "--max-steps", str(args.steps),
        "--checkpoint-every", str(args.checkpoint_every),
        "--checkpoint-dir", checkpoint_dir,
        "--loss-log", loss_log,
        "--launch-id", launch_id,
        "--log-every", "1000",   # keep the transcript readable; JSONL has it all
    ]
    if args.async_checkpointing:
        cmd.append("--async-checkpointing")
    if no_resume:
        cmd.append("--no-resume")
    if crash_at_step:
        cmd += ["--crash-at-step", str(crash_at_step)]
    return cmd


def _run(cmd: List[str], label: str, verbose: bool) -> int:
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if verbose or proc.returncode not in (0, 137, 1):
        for line in proc.stdout.splitlines():
            print(f"  | {line}")
    else:
        # Just the lines that carry evidence about the resume.
        for line in proc.stdout.splitlines():
            if any(k in line for k in ("RESUMED", "checkpoint @", "CRASH",
                                       "fast-forwarded", "No checkpoint",
                                       "max-steps", "DONE")):
                print(f"  | {line}")
    print(f"  -> {label} exited {proc.returncode}")
    if proc.returncode != 0 and "SIMULATED CRASH" not in proc.stdout:
        tail = proc.stderr.strip().splitlines()[-15:]
        for line in tail:
            print(f"  ! {line}")
    return proc.returncode


def read_loss_log(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _by_step(records: List[dict], launch: Optional[str] = None) -> Dict[int, float]:
    return {r["global_step"]: r["loss"]
            for r in records if launch is None or r["launch"] == launch}


def _diff_report(control: Dict[int, float], candidate: Dict[int, float],
                 tolerance: float, title: str) -> float:
    """Print a per-step comparison table and return the max absolute diff."""
    shared = sorted(set(control) & set(candidate))
    if not shared:
        raise VerificationError(f"{title}: no overlapping steps to compare")

    print(f"\n{title}  ({len(shared)} steps compared, tolerance={tolerance:g})")
    print(f"  {'step':>5}  {'control loss':>16}  {'candidate loss':>16}  {'abs diff':>12}")
    worst = 0.0
    for step in shared:
        diff = abs(control[step] - candidate[step])
        worst = max(worst, diff)
        flag = "" if diff <= tolerance else "   <-- MISMATCH"
        print(f"  {step:>5}  {control[step]:>16.10f}  {candidate[step]:>16.10f}  "
              f"{diff:>12.3e}{flag}")
    print(f"  max abs diff = {worst:.3e}")
    return worst


def verify(args: argparse.Namespace) -> dict:
    """
    Run the control/treatment experiment. Returns a summary dict on success,
    raises VerificationError on a mismatch.
    """
    work = os.path.join(REPO_ROOT, "results", "verify")
    if os.path.exists(work):
        shutil.rmtree(work)   # stale checkpoints would silently invalidate the run
    os.makedirs(work)

    ctrl_dir = os.path.join(work, "ckpt_control")
    ctrl_log = os.path.join(work, "loss_control.jsonl")
    test_dir = os.path.join(work, "ckpt_resumed")
    test_log = os.path.join(work, "loss_resumed.jsonl")

    print("=" * 78)
    print("CHECKPOINT CORRECTNESS VERIFICATION")
    print(f"  steps={args.steps}  checkpoint_every={args.checkpoint_every}  "
          f"crash_at_step={args.crash_at_step}")
    print(f"  world_size={args.world_size}  batch_size={args.batch_size}  "
          f"subset_size={args.subset_size}  seed={args.seed}")
    print("=" * 78)

    # ---- Run A: control, never interrupted -------------------------------
    print("\n[1/3] CONTROL: uninterrupted run")
    rc = _run(_train_cmd(args, checkpoint_dir=ctrl_dir, loss_log=ctrl_log,
                         launch_id=CONTROL, no_resume=True),
              "control", args.verbose)
    if rc != 0:
        raise VerificationError(f"control run failed with exit code {rc}")

    control = _by_step(read_loss_log(ctrl_log))
    if len(control) != args.steps:
        raise VerificationError(
            f"control logged {len(control)} steps, expected {args.steps}")

    # ---- Run B, launch 1: dies mid-run -----------------------------------
    print(f"\n[2/3] TREATMENT launch 1: crash at step {args.crash_at_step}")
    rc = _run(_train_cmd(args, checkpoint_dir=test_dir, loss_log=test_log,
                         launch_id=PRE_CRASH, no_resume=True,
                         crash_at_step=args.crash_at_step),
              "treatment launch 1", args.verbose)
    if rc == 0:
        raise VerificationError(
            "treatment launch 1 exited 0 — the simulated crash never fired, "
            "so nothing was actually recovered from")

    pre_crash = _by_step(read_loss_log(test_log), PRE_CRASH)
    if not pre_crash:
        raise VerificationError("treatment launch 1 logged no steps")

    # ---- Run B, launch 2: resumes from the last checkpoint ---------------
    print("\n[3/3] TREATMENT launch 2: resume from last checkpoint")
    rc = _run(_train_cmd(args, checkpoint_dir=test_dir, loss_log=test_log,
                         launch_id=RESUMED, no_resume=False),
              "treatment launch 2", args.verbose)
    if rc != 0:
        raise VerificationError(f"resumed run failed with exit code {rc}")

    resumed = _by_step(read_loss_log(test_log), RESUMED)
    if not resumed:
        raise VerificationError("resumed launch logged no steps")

    # ---- Assertions ------------------------------------------------------
    first_resumed = min(resumed)
    last_resumed = max(resumed)
    replayed = sorted(set(resumed) & set(pre_crash))

    if last_resumed != args.steps:
        raise VerificationError(
            f"resumed run stopped at step {last_resumed}, expected {args.steps}")
    if first_resumed > args.crash_at_step:
        raise VerificationError(
            f"resumed run started at step {first_resumed}, after the crash "
            f"point {args.crash_at_step} — it did not resume, it restarted")

    # Sanity check first: if two identical launches don't agree with each
    # other, the harness itself is nondeterministic and any result below
    # would be meaningless. Reported separately so the failure is legible.
    pre_worst = _diff_report(control, pre_crash, args.tolerance,
                             "A) pre-crash segment vs control "
                             "(harness determinism check)")
    if pre_worst > args.tolerance:
        raise VerificationError(
            f"pre-crash segment already diverges from control "
            f"(max diff {pre_worst:.3e}) — the training harness is not "
            f"deterministic, so the resume result below is not meaningful")

    worst = _diff_report(control, resumed, args.tolerance,
                         "B) RESUMED segment vs control (the actual test)")
    if worst > args.tolerance:
        raise VerificationError(
            f"resumed trajectory diverges from control: max abs diff "
            f"{worst:.3e} > tolerance {args.tolerance:g}")

    summary = dict(
        steps=args.steps,
        crash_at_step=args.crash_at_step,
        resumed_from_step=first_resumed - 1,
        replayed_steps=len(replayed),
        compared_steps=len(set(control) & set(resumed)),
        max_abs_diff=worst,
        tolerance=args.tolerance,
    )

    print("\n" + "=" * 78)
    print("PASS — resumed run reproduces the uninterrupted loss trajectory")
    print(f"  crashed after step        {args.crash_at_step}")
    print(f"  resumed from checkpoint   step {summary['resumed_from_step']}")
    print(f"  steps replayed after resume {summary['replayed_steps']} "
          f"(work lost to the crash)")
    print(f"  steps compared            {summary['compared_steps']}")
    print(f"  max abs loss difference   {worst:.3e}  (tolerance {args.tolerance:g})")
    print("=" * 78)

    with open(os.path.join(work, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    return summary


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, val in DEFAULTS.items():
        p.add_argument(f"--{key.replace('_', '-')}", type=type(val), default=val)
    p.add_argument("--async-checkpointing", action="store_true",
                   help="Run the same correctness experiment with checkpoints "
                        "written on a background thread. Must still be "
                        "bit-identical: async changes WHEN the bytes reach "
                        "disk, never what training computes.")
    p.add_argument("--verbose", action="store_true",
                   help="Show full stdout from each training subprocess")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        verify(args)
    except VerificationError as exc:
        print(f"\nFAIL — {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
