"""Generate the Grafana dashboard JSON.

Written as a generator rather than hand-authored JSON so panel geometry is
consistent and series colours come from one place -- the same validated
categorical slots the Phase 6 comparison chart uses, so a reader moving from
the chart to the dashboard sees the same hue mean the same thing.
"""
import json

DS = {"type": "prometheus", "uid": "checkpointlab-prom"}

# Validated categorical slots 1-4 (see the dataviz palette reference).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED = "#e34948"

panels = []
_id = [0]


def nid():
    _id[0] += 1
    return _id[0]


def target(expr, legend, ref="A"):
    return {"datasource": DS, "editorMode": "code", "expr": expr,
            "legendFormat": legend, "range": True, "refId": ref}


def timeseries(title, targets, x, y, w, h, unit=None, colors=None,
               desc="", fill=8, stepped=False, minval=None):
    overrides = []
    for name, hexc in (colors or {}).items():
        overrides.append({
            "matcher": {"id": "byName", "options": name},
            "properties": [{"id": "color",
                            "value": {"mode": "fixed", "fixedColor": hexc}}],
        })
    custom = {
        "drawStyle": "line",
        # 2px lines and >=8px points: the mark spec the charts follow.
        "lineWidth": 2,
        "fillOpacity": fill,
        "showPoints": "auto",
        "pointSize": 8,
        "spanNulls": False,
    }
    if stepped:
        # The interval is a decision that holds until it is changed. Linear
        # interpolation would draw a smooth ramp between two discrete choices
        # and imply values the scheduler never picked.
        custom["lineInterpolation"] = "stepAfter"
    fc = {"defaults": {"custom": custom,
                       "color": {"mode": "palette-classic"},
                       "unit": unit or "short"},
          "overrides": overrides}
    if minval is not None:
        fc["defaults"]["min"] = minval
    panels.append({
        "id": nid(), "type": "timeseries", "title": title,
        "description": desc, "datasource": DS, "targets": targets,
        "fieldConfig": fc,
        "options": {"legend": {"displayMode": "list", "placement": "bottom",
                               "showLegend": True},
                    "tooltip": {"mode": "multi", "sort": "none"}},
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
    })


def stat(title, expr, x, y, w, h, unit=None, desc="", thresholds=None,
         color=BLUE, decimals=None):
    steps = thresholds or [{"color": "text", "value": None}]
    fd = {"color": {"mode": "thresholds" if thresholds else "fixed",
                    "fixedColor": color},
          "thresholds": {"mode": "absolute", "steps": steps},
          "unit": unit or "short"}
    if decimals is not None:
        fd["decimals"] = decimals
    panels.append({
        "id": nid(), "type": "stat", "title": title, "description": desc,
        "datasource": DS, "targets": [target(expr, "")],
        "fieldConfig": {"defaults": fd, "overrides": []},
        "options": {"colorMode": "value", "graphMode": "area",
                    "justifyMode": "auto", "orientation": "auto",
                    "reduceOptions": {"calcs": ["lastNotNull"],
                                      "fields": "", "values": False},
                    "textMode": "auto"},
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
    })


J = 'ckpt_job=~"$job"'

# ---- row 1: the incident at a glance -----------------------------------
stat("Active failures", f'max(checkpointlab_active_failures{{{J}}})', 0, 0, 4, 4,
     desc="Failures detected but not yet recovered. Non-zero only DURING an "
          "incident, which is what makes it the alertable signal.",
     thresholds=[{"color": AQUA, "value": None}, {"color": RED, "value": 1}])
stat("Throughput", f'max(checkpointlab_throughput_steps_per_second{{{J}}})',
     4, 0, 5, 4, unit="none", decimals=2, color=BLUE,
     desc="Steps/sec over the last 20 steps.")
stat("Last recovery", f'max(checkpointlab_recovery_duration_seconds{{{J}}})',
     9, 0, 5, 4, unit="s", decimals=2, color=ORANGE,
     desc="Most recent failure-detected to training-resumed duration.")
stat("Checkpoint interval",
     f'max(checkpointlab_checkpoint_interval_steps{{{J}}})', 14, 0, 5, 4,
     unit="none", color=AQUA,
     desc="Interval currently in effect, in steps (not the computed one).")
stat("Recoveries", f'max(checkpointlab_recoveries_total{{{J}}})',
     19, 0, 5, 4, color=YELLOW,
     desc="Completed crash-detect-restore-resume cycles.")

# ---- row 2: throughput + interval over time ----------------------------
timeseries(
    "Throughput over time",
    [target(f'checkpointlab_throughput_steps_per_second{{{J}}}',
            "{{ckpt_job}} reported (rolling mean)"),
     target(f'rate(checkpointlab_steps_completed_total{{{J}}}[30s])',
            "{{ckpt_job}} observed (rate)", "B")],
    0, 4, 12, 8, unit="none", minval=0,
    colors={}, 
    desc="TWO different things, plotted together on purpose. 'reported' is "
         "the exporter's gauge: a rolling mean of the last 20 logged step "
         "durations. It degrades gently across a recovery rather than "
         "collapsing, because a dead worker stops APPENDING to the log -- it "
         "does not write zeros -- so the mean just goes stale. 'observed' is "
         "rate() over the step counter, which does fall to zero while no "
         "steps are completing, and whose trough width is the recovery cost. "
         "Read the gauge for 'how fast is a step', the rate for 'is work "
         "happening at all'.")

timeseries(
    "Checkpoint interval over time",
    [target(f'checkpointlab_checkpoint_interval_steps{{{J}}}',
            "{{ckpt_job}} interval (steps)"),
     target(f'checkpointlab_scheduler_optimal_interval_seconds{{{J}}}',
            "{{ckpt_job}} T* (seconds)", "B")],
    12, 4, 12, 8, unit="none", stepped=True, minval=0,
    desc="Interval in effect (steps, stepped) against the scheduler's target "
         "T* = sqrt(2dM) in seconds. Phase 6 found these can disagree, which "
         "is why both are plotted.")

# ---- row 3: recovery detail --------------------------------------------
timeseries(
    "Recovery duration, split by phase",
    [target(f'checkpointlab_recovery_phase_seconds{{{J}}}', "{{phase}}")],
    0, 12, 12, 8, unit="s", fill=25,
    colors={"terminate_workers": BLUE, "select_checkpoint": ORANGE,
            "relaunch_workers": AQUA, "reach_first_step": YELLOW},
    desc="~97% is reach_first_step: a fresh interpreter importing torch, "
         "rebuilding the model, loading the checkpoint. A dashboard showing "
         "only the total would send you optimising the controller instead.")

timeseries(
    "Useful vs recomputed work",
    [target(f'checkpointlab_steps_completed_total{{{J}}}', "{{ckpt_job}} useful"),
     target(f'checkpointlab_steps_recomputed_total{{{J}}}',
            "{{ckpt_job}} recomputed", "B")],
    12, 12, 12, 8, unit="none",
    colors={}, minval=0,
    desc="Recomputed steps are work a crash discarded and the job had to redo "
         "-- the same definition the Phase 6 benchmark uses for wasted "
         "compute, so dashboard and benchmark cannot disagree.")

# ---- row 4: cost + scheduler internals ---------------------------------
timeseries(
    "Checkpoint write duration",
    [target(f'checkpointlab_checkpoint_duration_seconds{{{J}}}',
            "{{ckpt_job}} last write")],
    0, 20, 8, 7, unit="s", minval=0,
    desc="d in T* = sqrt(2dM). ~90 MB of weights plus SGD momentum.")

timeseries(
    "Scheduler estimates",
    [target(f'checkpointlab_scheduler_mtbf_seconds{{{J}}}', "{{ckpt_job}} MTBF (M)"),
     target(f'checkpointlab_scheduler_recovery_cost_seconds{{{J}}}',
            "{{ckpt_job}} recovery cost (R)", "B")],
    8, 20, 8, 7, unit="s", minval=0,
    desc="Only populated under --adaptive-checkpointing. R lags one recovery "
         "behind by construction, so it reads 0 until the second recovery.")

timeseries(
    "Cumulative cost",
    [target(f'checkpointlab_checkpoint_seconds_total{{{J}}}', "{{ckpt_job}} checkpointing"),
     target(f'checkpointlab_recovery_seconds_total{{{J}}}', "{{ckpt_job}} recovering", "B"),
     target(f'checkpointlab_fast_forward_seconds_total{{{J}}}', "{{ckpt_job}} fast-forward", "C")],
    16, 20, 8, 7, unit="s", minval=0,
    desc="Where wall clock goes that is not useful compute.")

dashboard = {
    "uid": "checkpointlab",
    "title": "CheckpointLab — training, failures, recovery",
    "description": "Live view of the CheckpointLab testbed. Metrics come from "
                   "metrics/prometheus.py, which re-reads each job's JSONL "
                   "logs on scrape because the training workers are killed on "
                   "purpose and cannot host a scrape target themselves.",
    "tags": ["checkpointlab", "phase7"],
    "timezone": "browser",
    "editable": True,
    "schemaVersion": 39,
    "version": 1,
    "refresh": "5s",
    "time": {"from": "now-15m", "to": "now"},
    "templating": {"list": [{
        "name": "job",
        "label": "Job",
        "type": "query",
        "datasource": DS,
        "query": {"query": "label_values(checkpointlab_last_step, ckpt_job)",
                  "refId": "job-values"},
        "definition": "label_values(checkpointlab_last_step, ckpt_job)",
        "includeAll": True,
        "multi": True,
        "allValue": ".*",
        "current": {"selected": True, "text": ["All"], "value": ["$__all"]},
        "refresh": 2,
        "sort": 1,
    }]},
    "panels": panels,
}

with open("docker/grafana/dashboards/checkpointlab.json", "w") as fh:
    json.dump(dashboard, fh, indent=2)
print(f"wrote dashboard: {len(panels)} panels")
for p in panels:
    g = p["gridPos"]
    print(f"  {p['type']:<11} {p['title']:<34} x={g['x']:>2} y={g['y']:>2} w={g['w']:>2} h={g['h']}")
