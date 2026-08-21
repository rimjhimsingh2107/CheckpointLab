#!/usr/bin/env bash
# End-to-end verification that the containerized stack actually works.
#
# Four checks, in increasing strength:
#   1. the image builds and torch imports inside it
#   2. the fast test tier passes inside the image
#   3. two containers do real DDP training across the container network, and
#      REPEATING that run reproduces the losses bit-for-bit
#   4. the Phase 4 crash-detect-recover-resume loop works in a container
#
# ON CHECK 3, AND A CLAIM THIS SCRIPT DELIBERATELY DOES NOT MAKE
#
# The tempting check is "container losses == bare-metal losses". That is FALSE
# here and it should be: the container runs a Linux/aarch64 CPU wheel of torch
# while the host runs a macOS/arm64 build. Different builds mean different
# kernels and different floating-point reduction orders, so the two diverge
# from the very first step and compound -- measured at 1.4e-2 by step 12,
# a thousand times the 1e-5 tolerance Phase 2 uses.
#
# That is not a bug and papering over it with a loose tolerance would be
# dishonest. Phase 2's guarantee was always *within* one environment: a
# resumed run matches an uninterrupted run on the same build. So the check
# that means something is container-vs-container, which must be exactly 0.
# The cross-platform delta is printed as information, never as pass/fail.
#
# Usage: bash docker/verify_docker.sh

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DC="docker compose -f docker/docker-compose.yml"
PY="$REPO/.venv/bin/python"; [ -x "$PY" ] || PY="$(command -v python3)"
FAIL=0

bar()  { printf '%s\n' "===================================================================="; }
ok()   { echo "  [PASS] $1"; }
bad()  { echo "  [FAIL] $1"; FAIL=1; }

run_train() {  # $1 = results subdir
  TRAIN_RUN="$1" $DC --profile train up \
      --abort-on-container-exit --exit-code-from worker0 >/dev/null 2>&1
  local rc=$?
  $DC --profile train down --remove-orphans >/dev/null 2>&1
  return $rc
}

bar; echo " 1/4  BUILD"; bar
if $DC --profile train build 2>&1 | tail -3; then ok "image built"; else bad "build failed"; exit 1; fi
# Two different numbers, both real: docker's DISK USAGE column is the
# uncompressed on-disk footprint, while inspect .Size is the compressed
# content size you actually pull. Printing only the small one flatters.
docker image inspect checkpointlab:local --format \
  '         image: {{.Size}} bytes compressed (pull size)' \
  | sed 's/\([0-9]\{6,\}\)/~\1/'
docker images checkpointlab:local --format '         on disk (uncompressed): {{.Size}}'
if docker run --rm checkpointlab:local 2>&1 | tail -1 | grep -q "image OK"; then
  ok "torch imports inside the image"
else
  bad "torch does not import inside the image"; exit 1
fi

bar; echo " 2/4  FAST TEST TIER INSIDE THE IMAGE"; bar
if $DC --profile test run --rm tests 2>&1 | tail -3; then
  ok "fast tier passed in-container"
else
  bad "tests failed in-container"
fi
$DC --profile test down --remove-orphans >/dev/null 2>&1

bar; echo " 3/4  TWO CONTAINERS, ONE PER RANK"; bar
rm -rf results/docker-train-a results/docker-train-b results/baremetal-train
echo "  container run A..."; run_train docker-train-a || bad "container run A failed"
echo "  container run B (identical config)..."; run_train docker-train-b || bad "container run B failed"
echo "  bare metal, same world size, for the cross-platform delta..."
"$PY" training/train.py --world-size 2 \
    --job-dir "$REPO/results/baremetal-train" --deterministic \
    --batch-size 32 --subset-size 1024 --epochs 2 --max-steps 12 \
    --checkpoint-every 4 --log-every 1000 --no-resume >/dev/null 2>&1
echo

"$PY" - <<'INNER_PY'
import json, os, sys

def losses(name):
    p = f"results/{name}/loss_log.jsonl"
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["global_step"]] = r["loss"]
    return out

a, b, bare = losses("docker-train-a"), losses("docker-train-b"), losses("baremetal-train")
rc = 0

if not a or not b:
    print("  [FAIL] a container run produced no loss log"); sys.exit(1)
for name, d in (("A", a), ("B", b)):
    if max(d) != 12:
        print(f"  [FAIL] container run {name} reached step {max(d)}, expected 12"); rc = 1
if rc == 0:
    print("  [PASS] both container runs trained 12 steps over the container network")

shared = sorted(set(a) & set(b))
worst = max(abs(a[s] - b[s]) for s in shared)
print(f"  in-container determinism: max abs diff over {len(shared)} steps = {worst:.3e}")
if worst == 0.0:
    print("  [PASS] repeating the containerized run is BIT-IDENTICAL")
else:
    print(f"  [FAIL] containerized training is not reproducible (diff {worst:.3e})")
    rc = 1

if bare:
    s2 = sorted(set(a) & set(bare))
    w2 = max(abs(a[s] - bare[s]) for s in s2)
    print()
    print("  cross-platform delta (information, NOT pass/fail):")
    print(f"    container (linux/aarch64 wheel) vs host (macOS/arm64 wheel)")
    print(f"    {'step':>5} {'container':>15} {'bare metal':>15} {'diff':>10}")
    for s in (1, 2, 4, 8, 12):
        if s in s2:
            print(f"    {s:>5} {a[s]:>15.10f} {bare[s]:>15.10f} {abs(a[s]-bare[s]):>10.2e}")
    print(f"    max abs diff = {w2:.3e} -- expected: different torch builds")
    print(f"    reorder floating-point reductions. Determinism is a")
    print(f"    within-environment property, not a cross-platform one.")

sys.exit(rc)
INNER_PY
[ $? -eq 0 ] || bad "containerized training check failed"

bar; echo " 4/4  CRASH -> DETECT -> RECOVER -> RESUME, IN A CONTAINER"; bar
rm -rf results/docker-recover
$DC --profile recover run --rm controller 2>&1 | tail -22
$DC --profile recover down --remove-orphans >/dev/null 2>&1

"$PY" - <<'INNER_PY2'
import json, os, sys
p = "results/docker-recover/recovery_log.jsonl"
if not os.path.exists(p):
    print("  [FAIL] no recovery log"); sys.exit(1)
ev = [json.loads(l) for l in open(p) if l.strip()]
kinds = [e["event"] for e in ev]
need = ["job_started", "generation_launched", "failure_detected",
        "generation_terminated", "checkpoint_selected", "training_resumed",
        "job_finished"]
missing = [k for k in need if k not in kinds]
if missing:
    print(f"  [FAIL] recovery cycle incomplete, missing {missing}"); sys.exit(1)
print("  [PASS] full crash-detect-restore-resume cycle present in-container")
for r in [e for e in ev if e["event"] == "training_resumed"]:
    print(f"         rank {r['failed_rank']} died at step {r['step_at_failure']}, "
          f"restored to {r['restored_to_step']}, recovered in "
          f"{r['recovery_seconds']:.2f}s")
fin = [e for e in ev if e["event"] == "job_finished"][-1]
if fin.get("outcome") == "completed":
    print(f"  [PASS] outcome=completed generations={fin['generations']} "
          f"exit_codes={fin.get('exit_codes')}")
    sys.exit(0)
print(f"  [FAIL] outcome={fin.get('outcome')}")
sys.exit(1)
INNER_PY2
[ $? -eq 0 ] || bad "containerized recovery did not complete"

bar
[ "$FAIL" -eq 0 ] && echo " ALL DOCKER CHECKS PASSED" || echo " SOME DOCKER CHECKS FAILED"
bar
exit "$FAIL"
