"""Step 3 - The two-stage model (paper Sec. IV-D).

Stage 1: Random Forest -> "is this zone ALREADY infected (but no symptoms)?"  -> y_hidden
Stage 2: sequence model -> "will symptoms appear in the next H days?"          -> r_onset
         Default: gradient boosting fed the last K days of Stage-1 probability + features
         Optional: a real LSTM (see src/lstm_stage2.py, needs PyTorch)
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

K = 5  # days of history given to Stage 2


def stage1(X, y, seed=0):
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced",
                                  n_jobs=-1, random_state=seed).fit(X, y)


def add_history(df, cols, k=K):
    """Stack the last k days of `cols` as columns (this is the 'sequence' the model sees)."""
    g = df.sort_values(["field", "zone", "day"]).groupby(["field", "zone"])
    out = df.sort_values(["field", "zone", "day"]).copy()
    names = list(cols)
    for lag in range(1, k):
        for c in cols:
            out[f"{c}_lag{lag}"] = g[c].shift(lag)
            names.append(f"{c}_lag{lag}")
    return out.bfill().fillna(0), names


def stage2(Xseq, y, seed=0):
    return GradientBoostingClassifier(random_state=seed).fit(Xseq, y)
