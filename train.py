"""Step 4 - Train + evaluate. Splits are by FIELD (never random rows) to avoid leakage (paper Sec. V-D).
Run:  python src/train.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from simulate import simulate
from features import build_features
from models import stage1, stage2, add_history
from evaluate import best_threshold, report

df = simulate()
df, SENSOR_F, WEATHER_F, META_F = build_features(df)
train, val, test = (df[df.field <= 8].copy(), df[df.field.isin([9, 10])].copy(), df[df.field >= 11].copy())
print(f"rows train/val/test = {len(train)}/{len(val)}/{len(test)}  (split by field)")


def rf(X, y):
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced",
                                  n_jobs=-1, random_state=0).fit(X, y)


results = {}

# Baseline 1: symptom-only CNN. By design it needs visible symptoms; we only have pre-symptom rows,
# so it can never alert early -> constant 0 score (lead time 0).
p0 = np.zeros(len(test))
results["Symptom-only (reactive)"] = dict(Recall=0.0, Precision=0.0, F1=0.0, AUROC=0.5, LeadTime_days=0.0)

# Baselines 2 and 3: single-source Random Forests
for name, cols in [("Weather-only RF", WEATHER_F + META_F), ("Sensor-only RF", SENSOR_F)]:
    m = rf(train[cols], train.r_onset)
    thr = best_threshold(val.r_onset.values, m.predict_proba(val[cols])[:, 1])
    pt = m.predict_proba(test[cols])[:, 1]
    results[name] = report(test, test.r_onset.values, pt, thr)

# Proposed fusion: Stage 1 (hidden infection) -> Stage 2 (onset within H days)
F = SENSOR_F + WEATHER_F + META_F
s1 = stage1(train[F], train.y_hidden)
# out-of-fold Stage-1 probabilities for training rows (so Stage 2 does not see over-fit scores)
oof = cross_val_predict(rf(train[F], train.y_hidden), train[F], train.y_hidden,
                        groups=train.field, cv=GroupKFold(4), method="predict_proba")[:, 1]
for d, p in [(train, oof), (val, s1.predict_proba(val[F])[:, 1]), (test, s1.predict_proba(test[F])[:, 1])]:
    d["p_hidden"] = p
train, seq_cols = add_history(train, F + ["p_hidden"])
val, _ = add_history(val, F + ["p_hidden"])
test, _ = add_history(test, F + ["p_hidden"])
s2 = stage2(train[seq_cols], train.r_onset)
thr = best_threshold(val.r_onset.values, s2.predict_proba(val[seq_cols])[:, 1])
test["risk"] = s2.predict_proba(test[seq_cols])[:, 1]
results["Proposed fusion (2-stage)"] = report(test, test.r_onset.values, test.risk.values, thr)

table = pd.DataFrame(results).T.round(3)
print("\n=== Table III (filled from this simulated run) ===")
print(table.to_string())
os.makedirs("results", exist_ok=True)
table.to_csv("results/table3.csv")

imp = pd.Series(s1.feature_importances_, index=F).sort_values(ascending=False).head(8)
print("\nTop Stage-1 features (what the model relies on):\n", imp.round(3).to_string())
joblib.dump(dict(s1=s1, s2=s2, F=F, seq_cols=seq_cols, thr=thr, test=test), "results/bundle.joblib")
print(f"\nAlert threshold chosen on validation to minimise J: {thr:.2f}")
