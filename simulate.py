"""Step 1 - Make a synthetic dataset (clearly labelled SIMULATED, as the paper allows).

Each field has a 5x5 grid of zones observed daily for 70 days.
Weather is shared by the field. Infection happens more easily after warm, wet days.
After infection there is a hidden phase (4-8 days) where sensors drift
BEFORE the first visible symptom (tau). We only keep rows before tau.
"""
import numpy as np
import pandas as pd

H = 5  # forecast horizon in days (paper: 3 to 7)


def simulate(n_fields=14, grid=5, days=70, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_fields):
        t = np.arange(days)
        temp = 23 + 4 * np.sin(t / 9 + rng.uniform(0, 6)) + rng.normal(0, 1.2, days)
        hum = np.clip(65 + 15 * np.sin(t / 7 + rng.uniform(0, 6)) + rng.normal(0, 6, days), 30, 100)
        wet = np.clip((hum - 60) / 4 + rng.normal(0, 1.5, days), 0, 16)  # leaf-wetness hours
        favourable = ((wet >= 7) & (temp >= 18) & (temp <= 28)).astype(float)
        for gx in range(grid):
            for gy in range(grid):
                susc = rng.uniform(0.5, 1.5)
                # drought confounder: looks like stress in sensors but is NOT disease
                d0 = rng.integers(10, 55) if rng.random() < 0.3 else -99
                inf_day, tau = None, None
                for d in range(3, days):
                    hazard = 0.002 + 0.12 * favourable[d - 3:d].mean() * susc
                    if rng.random() < hazard:
                        inf_day, tau = d, d + int(rng.integers(4, 9))
                        break
                for d in range(days):
                    if tau is not None and d >= tau:
                        break  # visible symptoms: stop, this is what we want to beat
                    stress = 0.0
                    if inf_day is not None and d >= inf_day:
                        stress = min(1.0, (d - inf_day + 1) / (tau - inf_day))
                    drought = 0.45 if d0 <= d < d0 + 12 else 0.0
                    s = max(stress, drought)
                    rows.append(dict(
                        field=f, zone=gx * grid + gy, gx=gx, gy=gy, day=d,
                        ndvi=0.80 - 0.12 * s + rng.normal(0, 0.025),
                        temp_gap=1.0 + 2.2 * s + rng.normal(0, 0.7),   # canopy - air temp
                        red_edge=720 - 5 * s + rng.normal(0, 1.5),
                        temp=temp[d] + rng.normal(0, 0.4),
                        humidity=hum[d] + rng.normal(0, 2),
                        leaf_wet_h=wet[d] + rng.normal(0, 0.5),
                        susceptibility=susc,
                        y_hidden=int(inf_day is not None and inf_day <= d < tau),
                        r_onset=int(tau is not None and tau - d <= H),
                        tau=tau if tau is not None else -1,
                    ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = simulate()
    df.to_csv("data_simulated.csv", index=False)
    print(df.shape, "hidden-infection rate:", round(df.y_hidden.mean(), 3))
