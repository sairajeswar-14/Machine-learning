"""OPTIONAL: real LSTM for Stage 2 (needs: pip install torch). Not required for the demo.
NOTE: this file is a starting point and has not been run by the author of this template.
Input per sample: last K days of features (+ Stage-1 probability). Output: P(onset within H days).
"""
import numpy as np
import torch
import torch.nn as nn


class RiskLSTM(nn.Module):
    def __init__(self, n_feat, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                      # x: (batch, K, n_feat)
        out, _ = self.lstm(x)
        return torch.sigmoid(self.head(out[:, -1])).squeeze(-1)


def to_sequences(df, cols, k=5):
    X, y = [], []
    for _, z in df.sort_values("day").groupby(["field", "zone"]):
        a, t = z[cols].values.astype("float32"), z.r_onset.values
        for i in range(k - 1, len(z)):
            X.append(a[i - k + 1:i + 1]); y.append(t[i])
    return torch.tensor(np.array(X)), torch.tensor(np.array(y), dtype=torch.float32)


def fit(X, y, epochs=15, lr=1e-3):
    m, opt, loss = RiskLSTM(X.shape[2]), None, nn.BCELoss()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); l = loss(m(X), y); l.backward(); opt.step()
    return m
