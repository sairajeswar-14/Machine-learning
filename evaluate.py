"""Metrics from paper Sec. V-C: recall, precision, F1, AUROC, lead time, cost J."""
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

C_MISS, C_FA = 10.0, 1.0  # a missed outbreak costs 10x a needless inspection


def best_threshold(y, p):
    """Pick the alert threshold that minimises J = c_miss*misses + c_fa*false_alarms (on validation)."""
    best_t, best_j = 0.5, 1e18
    for t in np.linspace(0.05, 0.95, 91):
        pred = p >= t
        j = C_MISS * ((y == 1) & ~pred).sum() + C_FA * ((y == 0) & pred).sum()
        if j < best_j:
            best_t, best_j = t, j
    return best_t


def lead_time(df, p, thr):
    """For each zone that gets infected: tau - first alert raised during the hidden phase.
    Zones never alerted in time count 0 days. (Alerts before infection are false alarms, not lead time.)"""
    d = df.assign(alert=p >= thr)
    leads = []
    for _, z in d[d.tau >= 0].groupby(["field", "zone"]):
        a = z[z.alert & (z.y_hidden == 1)]
        leads.append(max(0, int(z.tau.iloc[0] - a.day.min())) if len(a) else 0)
    return float(np.mean(leads)) if leads else 0.0


def report(df, y, p, thr):
    pred = (p >= thr).astype(int)
    return dict(Recall=recall_score(y, pred), Precision=precision_score(y, pred, zero_division=0),
                F1=f1_score(y, pred), AUROC=roc_auc_score(y, p), LeadTime_days=lead_time(df, p, thr))
