#!/usr/bin/env bash
# Verification for the Phase 7 monitoring stack.
#
# Checks the whole chain, not just that containers are running:
#   1. exporter serves a parseable /metrics with real samples
#   2. Prometheus's scrape target is up with no error
#   3. Prometheus has the series, under the label the dashboard queries
#   4. Grafana is healthy, and its datasource + dashboard are provisioned
#   5. Grafana can actually QUERY the data through its datasource proxy
#
# Check 5 is the one that matters. A provisioned datasource that cannot
# reach Prometheus, or a dashboard whose selectors don't match the exported
# labels, both present as an empty panel with no error anywhere. Querying
# through Grafana's own proxy is the only way to prove the path a panel
# takes actually works.
#
# Usage: bash docker/verify_monitoring.sh   (monitor profile must be up)

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"; [ -x "$PY" ] || PY="$(command -v python3)"
FAIL=0
EXP=http://localhost:9101
PROM=http://localhost:9090
GRAF=http://localhost:3000
DSUID=checkpointlab-prom

bar() { printf '%s\n' "===================================================================="; }
ok()  { echo "  [PASS] $1"; }
bad() { echo "  [FAIL] $1"; FAIL=1; }

bar; echo " 1/5  EXPORTER"; bar
N=$(curl -s --max-time 10 "$EXP/metrics" | grep -c '^checkpointlab' || echo 0)
[ "$N" -gt 20 ] && ok "exporter serving $N checkpointlab samples" \
                || bad "exporter served only $N samples"
JOBS=$(curl -s --max-time 10 "$EXP/metrics" | grep '^checkpointlab_exporter_jobs_discovered' | awk '{print $2}')
echo "         jobs discovered: ${JOBS:-?}"

bar; echo " 2/5  PROMETHEUS TARGET"; bar
curl -s --max-time 10 "$PROM/api/v1/targets" | "$PY" -c '
import json, sys
d = json.load(sys.stdin)
bad = 0
for t in d["data"]["activeTargets"]:
    j = t["labels"]["job"]; h = t["health"]
    e = t.get("lastError")
    print(f"         {j:<15} health={h} err={e!r}")
    if h != "up":
        bad = 1
sys.exit(bad)' && ok "all scrape targets up" || bad "a scrape target is down"

bar; echo " 3/5  PROMETHEUS HAS THE SERIES (label the dashboard uses)"; bar
# ckpt_job, NOT job: `job` is reserved for the scrape-job name and an
# exporter that exports it gets silently renamed to exported_job.
curl -sG --max-time 10 "$PROM/api/v1/label/ckpt_job/values" | "$PY" -c '
import json, sys
v = json.load(sys.stdin)["data"]
print(f"         ckpt_job values: {v}")
sys.exit(0 if v else 1)' && ok "series carry ckpt_job" \
  || bad "no ckpt_job label -- did it collide with the reserved job label?"

bar; echo " 4/5  GRAFANA PROVISIONING"; bar
curl -s --max-time 10 "$GRAF/api/health" | "$PY" -c '
import json,sys; d=json.load(sys.stdin)
v = d["version"]; db = d["database"]
print(f"         grafana {v} database={db}")
sys.exit(0 if d["database"]=="ok" else 1)' && ok "grafana healthy" || bad "grafana unhealthy"

curl -s --max-time 10 "$GRAF/api/datasources" | "$PY" -c '
import json,sys
ds=[d for d in json.load(sys.stdin) if d["uid"]=="checkpointlab-prom"]
if not ds: print("         no datasource with uid checkpointlab-prom"); sys.exit(1)
n = ds[0]["name"]; u = ds[0]["url"]
print(f"         datasource {n} -> {u}")' \
  && ok "datasource provisioned with the uid the dashboard references" \
  || bad "datasource missing or wrong uid"

curl -s --max-time 10 "$GRAF/api/dashboards/uid/checkpointlab" | "$PY" -c '
import json,sys
d=json.load(sys.stdin)
if "dashboard" not in d: print("         dashboard not found"); sys.exit(1)
p=d["dashboard"]["panels"]
t = d["dashboard"]["title"]
print(f"         dashboard {t!r}: {len(p)} panels")
sys.exit(0 if len(p)>=8 else 1)' && ok "dashboard provisioned" || bad "dashboard missing"

bar; echo " 5/5  GRAFANA CAN QUERY THE DATA (via its datasource proxy)"; bar
for expr in checkpointlab_throughput_steps_per_second \
            checkpointlab_checkpoint_interval_steps \
            checkpointlab_recovery_duration_seconds \
            checkpointlab_active_failures; do
  curl -sG --max-time 15 "$GRAF/api/datasources/proxy/uid/$DSUID/api/v1/query" \
       --data-urlencode "query=$expr" | "$PY" -c "
import json, sys
d = json.load(sys.stdin)
r = d.get('data', {}).get('result', []) if d.get('status') == 'success' else []
print(f'         {\"$expr\":<48} {len(r)} series')
sys.exit(0 if r else 1)" || bad "$expr returned nothing through Grafana"
done
[ "$FAIL" -eq 0 ] && ok "Grafana resolves every dashboard metric"

bar
[ "$FAIL" -eq 0 ] && echo " ALL MONITORING CHECKS PASSED" || echo " SOME MONITORING CHECKS FAILED"
bar
exit "$FAIL"
