#!/usr/bin/env bash
# 30-second demo: training -> injected worker failure -> automatic detection
# -> checkpoint restore -> resume to completion, with nobody watching.
#
# Deterministic on purpose. The failure schedule is a pure function of
# (--failure-seed, step), so this run injects exactly one failure, at step 7,
# on worker 1 -- every single time. A demo that showed a different failure on
# every take would undercut the Phase 3 reproducibility claim it exists to
# illustrate.
#
# Sized for ~30s on a CPU-only laptop: 14 steps of a 1024-image CIFAR-10
# subset across 2 DDP workers, checkpointing every 3 steps.
#
# Usage:  bash demo/demo.sh
# Record: see demo/README.md

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

# Relative so the captured transcript is portable, not full of one
# machine's home directory.
JOB_DIR="results/demo"
rm -rf "$JOB_DIR"

# Checkpoints land at steps 3 and 6; the failure hits step 7. So the resume
# has a real checkpoint to come back to AND real work to replay -- which is
# the whole point. A failure before the first checkpoint would just restart
# from zero and show nothing.
RATE=0.05
SEED=23
STEPS=14
EVERY=3

bar() { printf '%s\n' "────────────────────────────────────────────────────────────────────"; }

bar
echo " CheckpointLab — crash → detect → recover → resume, unattended"
bar
echo
echo " The failure schedule is decided up front from the seed:"
echo
"$PY" failures/injector.py --failure-rate "$RATE" --workers 2 --seed "$SEED" \
      --max-steps "$STEPS" | sed 's/^/   /'
echo
echo " Checkpoints every $EVERY steps → 3, 6.  Failure at step 7."
echo " So the job must lose step 7, restore step 6, replay, and finish at $STEPS."
echo
bar
echo " Handing the job to the controller. No further human input."
bar
echo

"$PY" -u controller/recovery.py \
    --job-dir "$JOB_DIR" \
    --world-size 2 \
    --epochs 4 \
    --batch-size 32 \
    --subset-size 1024 \
    --max-steps "$STEPS" \
    --checkpoint-every "$EVERY" \
    --failure-rate "$RATE" \
    --failure-seed "$SEED" \
    --log-every 1000
RC=$?

echo
bar
if [ "$RC" -eq 0 ]; then
  echo " Job completed through a worker failure with zero manual intervention."
else
  echo " Demo FAILED (exit $RC)"
fi
bar
exit "$RC"
