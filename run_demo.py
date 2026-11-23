"""Step 5 - The demo: risk map + alert levels for a field, and one zone's timeline.
Run:  python src/run_demo.py            (after python src/train.py)
Outputs: results/risk_map.png, results/zone_timeline.png and a text alert list.
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

b = joblib.load("results/bundle.joblib")
test, thr = b["test"], b["thr"]
WATCH, WARN, ACT = thr, 0.4, 0.7          # alert levels (paper Sec. IV-E)


def level(r):
    return "ACT NOW" if r >= ACT else "WARNING" if r >= WARN else "WATCH" if r >= WATCH else "ok"


field = int(test.field.min())
f = test[test.field == field]
# pick the day where the most zones are still hidden-infected: most interesting to show
day = int(f.groupby("day").y_hidden.sum().idxmax())
snap = f[f.day == day]

# ---- 1. Risk map (5x5 zones) ----
grid = np.full((5, 5), np.nan)
for _, r in snap.iterrows():
    grid[int(r.gx), int(r.gy)] = r.risk
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=1)
for _, r in snap.iterrows():
    mark = "!" if r.y_hidden == 1 else ""
    ax.text(r.gy, r.gx, f"{r.risk:.2f}{mark}", ha="center", va="center", fontsize=8)
ax.set_title(f"Field {field}, day {day}: predicted risk\n('!' = truly infected, no visible symptoms yet)")
ax.set_xticks([]); ax.set_yticks([]); plt.colorbar(im, label="risk of symptoms within 5 days")
plt.tight_layout(); plt.savefig("results/risk_map.png", dpi=150); plt.close()

# ---- 2. Alert list, highest risk first (scouts inspect these first) ----
print(f"\nField {field}, day {day} - zones ranked by risk")
for _, r in snap.sort_values("risk", ascending=False).head(6).iterrows():
    print(f"  zone {int(r.zone):2d}  risk {r.risk:.2f}  {level(r.risk):8s}  "
          f"{'(truly in hidden phase)' if r.y_hidden else ''}")

# ---- 3. Timeline for one infected zone: the alert arrives before the symptom day ----
inf = f[f.tau >= 0].groupby("zone").size().sort_values(ascending=False).index
z = f[f.zone == inf[0]].sort_values("day")
tau = int(z.tau.iloc[0])
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(z.day, z.risk, label="predicted risk")
ax.axhline(thr, ls="--", c="gray", label=f"alert threshold ({thr:.2f})")
hid = z[z.y_hidden == 1]
ax.axvspan(hid.day.min(), tau, color="orange", alpha=0.2, label="hidden phase")
ax.axvline(tau, c="red", label="first visible symptom")
first = z[(z.risk >= thr) & (z.y_hidden == 1)]
if len(first):
    ax.scatter(first.day.min(), first.risk.iloc[0], c="green", zorder=5, s=60, label="first alert")
    print(f"\nZone {int(z.zone.iloc[0])}: first alert on day {first.day.min()}, "
          f"symptoms appear day {tau} -> lead time {tau - first.day.min()} days")
ax.set_xlabel("day"); ax.set_ylabel("risk"); ax.legend(fontsize=7)
ax.set_title("Alert fires during the hidden phase, before symptoms")
plt.tight_layout(); plt.savefig("results/zone_timeline.png", dpi=150)
print("\nSaved results/risk_map.png and results/zone_timeline.png")
