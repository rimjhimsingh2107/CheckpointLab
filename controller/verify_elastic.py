"""
Verification for elastic resumption.

WHAT "CORRECT" CAN MEAN HERE, AND WHAT IT CANNOT

Phase 2's guarantee was bit-exactness: a resumed run reproduces the
uninterrupted loss trajectory to 0.0. That guarantee is **unavailable** at a
different world size, and not because of a bug:

  * the effective batch is world_size * batch_size, so 2 ranks x 32 = 64
    becomes 3 x 32 = 96. Gradients are averaged over more samples per step.
    A different optimisation problem is being solved.
  * DistributedSampler reshards. Rank r of 3 gets a different slice of the
    epoch's permutation than rank r of 2, so even at the same sample position
    the *composition* of each batch differs.

So the honest success criterion is: **resumes at the correct data position and
continues converging from the state it had**, not "matches the old
trajectory".

"Continues converging" needs a control to be falsifiable. A resumed run whose
loss looks plausible proves nothing on its own -- a job that silently
discarded the checkpoint and restarted from random init would also produce a
plausible falling loss. So this script trains a FRESH run at the new world
size for the same number of optimiser steps, and requires the resumed run to
be decisively better. If the checkpoint were being ignored, the two would
agree; they must not.

Usage:
    python controller/verify_elastic.py
    python controller/verify_elastic.py --from-workers 2 --to-workers 4
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(REPO_ROOT, "training", "train.py")
PY = sys.executable

COMMON = ["--deterministic", "--batch-size", "32", "--subset-size", "1024",
          "--epochs", "12", "--checkpoint-every", "3", "--log-every", "1000"]


class ElasticError(AssertionError):
    pass


def run(world_size: int, job_dir: str, max_steps: int, *, resume: bool,
        label: str) -> Tuple[int, str]:
    """Launch one generation directly (no controller) and return (rc, stdout)."""
    cmd = [PY, "-u", TRAIN, "--world-size", str(world_size),
           "--job-dir", job_dir, "--max-steps", str(max_steps)] + COMMON
    if not resume:
        cmd.append("--no-resume")
    print(f"  [{label}] world_size={world_size} max_steps={max_steps} "
          f"resume={resume}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
    return proc.returncode, proc.stdout


def losses(job_dir: str) -> Dict[int, float]:
    """Last recorded loss per global step (the surviving one)."""
    path = os.path.join(job_dir, "loss_log.jsonl")
    out: Dict[int, float] = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["global_step"]] = r["loss"]
    return out


def parse_elastic_line(stdout: str) -> Optional[dict]:
    """Pull the conversion the worker reported, so it can be checked."""
    m = re.search(r"ELASTIC RESUME: world_size (\d+) -> (\d+)", stdout)
    if not m:
        return None
    d = {"from": int(m.group(1)), "to": int(m.group(2))}
    m2 = re.search(r"(\d+) samples into epoch (\d+); at (\d+)x(\d+) per step "
                   r"that is step (\d+)( \(exact\)| \(\+(\d+) samples)?",
                   stdout)
    if m2:
        d.update(samples=int(m2.group(1)), epoch=int(m2.group(2)),
                 world_size=int(m2.group(3)), batch=int(m2.group(4)),
                 step=int(m2.group(5)),
                 leftover=int(m2.group(7)) if m2.group(7) else 0)
    return d


def scenario(from_ws: int, to_ws: int, pre_steps: int, post_steps: int,
             work: str) -> dict:
    """
    Train at from_ws, then resume the SAME job dir at to_ws, and compare
    against a fresh run at to_ws.
    """
    tag = f"{from_ws}to{to_ws}"
    job = os.path.join(work, f"elastic_{tag}")
    control = os.path.join(work, f"control_{tag}")
    for d in (job, control):
        shutil.rmtree(d, ignore_errors=True)

    total = pre_steps + post_steps
    print(f"\n--- {from_ws} -> {to_ws} workers ---")

    rc, _ = run(from_ws, job, pre_steps, resume=False, label="phase 1")
    if rc != 0:
        raise ElasticError(f"{tag}: initial {from_ws}-worker run failed (rc={rc})")
    pre = losses(job)
    if max(pre, default=0) != pre_steps:
        raise ElasticError(f"{tag}: phase 1 reached step {max(pre, default=0)}, "
                           f"expected {pre_steps}")
    loss_at_crash = pre[pre_steps]

    # Same job dir, new world size -> elastic resume off the last checkpoint.
    rc, out = run(to_ws, job, total, resume=True, label="phase 2 (elastic)")
    if rc != 0:
        raise ElasticError(f"{tag}: elastic resume at {to_ws} workers failed "
                           f"(rc={rc})")

    conv = parse_elastic_line(out)
    if conv is None:
        raise ElasticError(f"{tag}: worker never reported an ELASTIC RESUME -- "
                           f"it did not notice the world size changed")
    if (conv["from"], conv["to"]) != (from_ws, to_ws):
        raise ElasticError(f"{tag}: reported {conv['from']}->{conv['to']}")

    # The conversion is arithmetic, so check it rather than trust it.
    expected_step = conv["samples"] // (to_ws * conv["batch"])
    if conv["step"] != expected_step:
        raise ElasticError(
            f"{tag}: converted {conv['samples']} samples to step "
            f"{conv['step']}, but {conv['samples']}//({to_ws}*"
            f"{conv['batch']}) = {expected_step}")

    post = losses(job)
    if max(post, default=0) != total:
        raise ElasticError(f"{tag}: elastic run reached step "
                           f"{max(post, default=0)}, expected {total}")
    loss_after = post[total]

    # Control: a fresh run at the NEW world size, same number of optimiser
    # steps as the resumed segment performed.
    rc, _ = run(to_ws, control, post_steps, resume=False, label="control (fresh)")
    if rc != 0:
        raise ElasticError(f"{tag}: control run failed (rc={rc})")
    ctrl = losses(control)
    loss_control = ctrl[max(ctrl)]

    print(f"  loss at crash (step {pre_steps}, {from_ws} workers): "
          f"{loss_at_crash:.4f}")
    print(f"  loss after elastic resume (step {total}, {to_ws} workers): "
          f"{loss_after:.4f}")
    print(f"  loss of FRESH {to_ws}-worker run after {post_steps} steps: "
          f"{loss_control:.4f}")

    if loss_after >= loss_control:
        raise ElasticError(
            f"{tag}: resumed run ({loss_after:.4f}) is no better than a fresh "
            f"run at the same world size ({loss_control:.4f}) -- the "
            f"checkpoint was not actually carried across the world-size change")

    margin = loss_control - loss_after
    print(f"  -> resumed model is {margin:.4f} better than fresh "
          f"({100 * margin / loss_control:.1f}%): state survived the change")

    # Guard against a vacuous pass. If the checkpoint happened to land on an
    # epoch boundary, samples_into_epoch is 0 and the conversion is trivially
    # 0 -> 0: the scenario would pass without ever exercising the arithmetic
    # it exists to test. Callers must choose step counts that land mid-epoch.
    if conv["samples"] == 0:
        raise ElasticError(
            f"{tag}: checkpoint landed on an epoch boundary "
            f"(samples_into_epoch=0), so the sample->step conversion was "
            f"trivial and this scenario proved nothing about it. Choose "
            f"--pre-steps so the last checkpoint falls mid-epoch.")

    return {
        "from_workers": from_ws, "to_workers": to_ws,
        "pre_steps": pre_steps, "post_steps": post_steps,
        "samples_into_epoch": conv["samples"],
        "converted_step": conv["step"], "leftover_samples": conv["leftover"],
        "effective_batch": [from_ws * conv["batch"], to_ws * conv["batch"]],
        "loss_at_crash": round(loss_at_crash, 6),
        "loss_after_resume": round(loss_after, 6),
        "loss_fresh_control": round(loss_control, 6),
        "margin_vs_fresh": round(margin, 6),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # 28 on purpose. With checkpoint-every 3 the last checkpoint is at 27,
    # which is mid-epoch at BOTH world sizes tested (16 steps/epoch at 2
    # workers, 10 at 3), so the sample->step conversion is non-trivial in
    # both directions. 30 would put the 3-worker checkpoint exactly on an
    # epoch boundary and make that scenario vacuous.
    p.add_argument("--pre-steps", type=int, default=28)
    p.add_argument("--post-steps", type=int, default=15)
    p.add_argument("--pairs", default="2:3,3:2",
                   help="Comma-separated from:to world-size pairs to test")
    args = p.parse_args(argv)

    work = os.path.join(REPO_ROOT, "results", "elastic")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)

    print("=" * 78)
    print("ELASTIC RESUMPTION")
    print("  criterion: resumes at the correct data position and continues")
    print("  converging -- NOT bit-identical, which is unavailable once the")
    print("  effective batch size changes.")
    print("=" * 78)

    results, failures = [], []
    for pair in args.pairs.split(","):
        a, b = (int(x) for x in pair.split(":"))
        try:
            results.append(scenario(a, b, args.pre_steps, args.post_steps, work))
        except ElasticError as exc:
            print(f"\n  FAIL {exc}", file=sys.stderr)
            failures.append(str(exc))

    print("\n" + "=" * 78)
    if failures:
        print(f"FAIL — {len(failures)} of {len(failures) + len(results)} "
              f"scenarios failed")
        print("=" * 78)
        return 1

    print("PASS — elastic resumption verified")
    print(f"  {'change':<10} {'samples':>8} {'-> step':>8} {'eff batch':>12} "
          f"{'loss resumed':>13} {'loss fresh':>11}")
    for r in results:
        print(f"  {r['from_workers']}->{r['to_workers']} workers "
              f"{r['samples_into_epoch']:>8} {r['converted_step']:>8} "
              f"{str(r['effective_batch']):>12} "
              f"{r['loss_after_resume']:>13.4f} {r['loss_fresh_control']:>11.4f}")
    print("=" * 78)

    with open(os.path.join(work, "summary.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
