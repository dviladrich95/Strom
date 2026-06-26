"""Exploratory: characterise the thermostat's on/off pulse train to size a rolling
window that smooths the bang-bang HeaterOutput into a local duty-cycle curve.

Reports run-length (ON / OFF) and on->on period distributions, and plots the
period histogram + CDF with the window size that captures 95% / 99% of cycles.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "data/chunks_Nov/thermostat.csv"   # 5-min November thermostat run

df = pd.read_csv(SRC, index_col="Timestamp", parse_dates=["Timestamp"])
dt_min = (df.index[1] - df.index[0]).total_seconds() / 60.0
on = (df["HeaterOutput"] > 0.5).astype(int).to_numpy()

# ── run-length encode the binary signal ──────────────────────────────────────
change = np.diff(on)
edges = np.flatnonzero(change)                       # indices just before a transition
bounds = np.concatenate(([0], edges + 1, [len(on)]))
run_len = np.diff(bounds)                             # length of each constant run (steps)
run_state = on[bounds[:-1]]                           # 0 or 1 for each run

on_runs_min  = run_len[run_state == 1] * dt_min
off_runs_min = run_len[run_state == 0] * dt_min

rising = np.flatnonzero(change == 1) + 1              # 0->1 transitions (pulse starts)
periods_min = np.diff(rising) * dt_min               # on->on period

n_transitions = len(edges)

def pct(a, q):
    return np.percentile(a, q) if len(a) else float("nan")

print(f"resolution           : {dt_min:.0f} min/step, {len(on)} steps")
print(f"transitions (jumps)  : {n_transitions}")
print(f"ON pulses            : {len(on_runs_min)}  | OFF gaps: {len(off_runs_min)}")
print()
for name, a in [("ON-run (min)", on_runs_min), ("OFF-run (min)", off_runs_min),
                ("PERIOD on->on (min)", periods_min)]:
    if len(a):
        print(f"{name:22s} median {np.median(a):6.0f} | p90 {pct(a,90):6.0f} | "
              f"p95 {pct(a,95):6.0f} | p99 {pct(a,99):6.0f} | max {a.max():6.0f}")
print()
for q in (90, 95, 99):
    w = pct(periods_min, q)
    print(f"window to catch {q}% of cycles: {w:6.0f} min  = {w/dt_min:5.1f} steps")

# ── plot: period histogram + CDF ─────────────────────────────────────────────
p95, p99 = pct(periods_min, 95), pct(periods_min, 99)
fig, (axh, axc) = plt.subplots(1, 2, figsize=(12, 4.2))

axh.hist(periods_min, bins=40, color="#9a3b2e", alpha=0.55, edgecolor="white", linewidth=0.4)
axh.axvline(np.median(periods_min), color="#1a1a1a", ls="--", lw=1.2, label=f"median {np.median(periods_min):.0f} min")
axh.axvline(p95, color="#b5532a", ls="-", lw=1.4, label=f"p95 {p95:.0f} min")
axh.axvline(p99, color="#2f6f8f", ls="-", lw=1.6, label=f"p99 {p99:.0f} min")
axh.set_xlabel("on→on period (min)")
axh.set_ylabel("count")
axh.set_title("Thermostat pulse-period distribution")
axh.legend(fontsize=8)

s = np.sort(periods_min)
cdf = np.arange(1, len(s) + 1) / len(s)
axc.plot(s, cdf * 100, color="#9a3b2e", lw=1.8)
for q, c in [(95, "#b5532a"), (99, "#2f6f8f")]:
    w = pct(periods_min, q)
    axc.axhline(q, color=c, ls=":", lw=1.0)
    axc.axvline(w, color=c, ls="-", lw=1.4, label=f"p{q}: {w:.0f} min ({w/dt_min:.0f} steps)")
axc.set_xlabel("on→on period (min)")
axc.set_ylabel("cumulative % of cycles")
axc.set_title("CDF — window to catch X% of cycles")
axc.set_ylim(0, 100)
axc.legend(fontsize=8, loc="lower right")

plt.tight_layout()
out = "plots/pulse_distribution_thermostat_Nov.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nSaved {out}")
