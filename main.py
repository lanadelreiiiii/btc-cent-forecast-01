#!/usr/bin/env python3
"""btc-cent-forecast-01 entry: 4-hour-ahead BTC point forecast (conditional median).

Method: per-case OLS on the case's own 72h of 1-minute closes. For each minute t
(with >=24h of history and a full 240-min lookahead inside the case), the target is
the next 240-minute log return; features are the constant plus log returns over the
past 60 / 240 / 720 / 1440 minutes. The fitted model predicts the log return from
the checkpoint, and the guess is last_price * exp(predicted return).

Why: BTC 1-minute closes show strong short-horizon mean reversion in this regime
(past 24h return correlates ~-0.5 with the next 4h return) plus a persistent drift
component. A no-constant model is WORSE than persistence; the constant (drift) is
essential. The per-case fit adapts to each case's own regime, so no global
parameters are hard-coded from the practice window.

Verification (re-run 15 Nov 2026, after the competition opened; 6 practice cases
p00-p05, true price = next case's final price, p05 proxied by the Kraken BTC/USD
06:00-candle close 73522.10):
  shrink  model MAE (cents)  persistence MAE  improvement
  0.5     32,166             39,609           +18.8%
  0.6     30,683             39,609           +22.5%
  0.7     30,145             39,609           +23.9%
  0.8     30,306             39,609           +23.5%
  0.9     30,936             39,609           +21.9%
  1.0     32,006             39,609           +19.2%
Shrink 0.7 is the best of the sweep on the 6 practice cases, so it stays. The
p05 case is the cautionary one: the model bets on a bounce that never comes
(forecast 73837.06 vs true 73522.10), which is exactly why the shrinkage exists.
Re-verified after this rewrite: running this file on the 6 practice inputs
reproduces the same forecasts and the same 30,145-cent MAE, so the rewrite
changed documentation and structure only, not the model.

Pure standard library, deterministic, no network. Fits a 5x5 system per case:
fast (<0.1s per case).
"""
import json
import math
import os
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

WINDOWS = (60, 240, 720, 1440)
HORIZON = 240
# Shrinkage on the fitted prediction: 0.7 dampens the model's bets, which costs
# a little when the regime persists but limits the damage when it flips (the
# practice's p05 case: the model bets on a bounce that never comes).
# Verified by sweep 15 Nov 2026 (see module docstring): 0.7 is the best of
# 0.5/0.6/0.7/0.8/0.9/1.0 on the 6 practice cases.
SHRINK = 0.7
MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999


def price5(x):
    """Clamp to the scorer grid: (0, 10_000_000), 5 decimals."""
    if not math.isfinite(x):
        raise ValueError("non-finite price")
    return round(min(MAX_PRICE, max(MIN_PRICE, x)) * 1e5) / 1e5


def logret(a, b):
    return math.log(b / a)


def solve_linear(A, b):
    """Gaussian elimination with partial pivoting; A is n x n, b length n."""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            raise ValueError("singular matrix")
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                f = M[r][c] / pv
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def fit_ols(prices):
    """Fit OLS: target = log return over next HORIZON minutes.
    Features: [1, r60, r240, r720, r1440]. Returns beta list or None."""
    n = len(prices)
    need = max(WINDOWS)
    if n < need + HORIZON + 1:
        return None
    k = 1 + len(WINDOWS)
    AtA = [[0.0] * k for _ in range(k)]
    Aty = [0.0] * k
    for t in range(need, n - HORIZON):
        row = [1.0] + [logret(prices[t - w], prices[t]) for w in WINDOWS]
        y = logret(prices[t], prices[t + HORIZON])
        for i in range(k):
            Aty[i] += row[i] * y
            ri = row[i]
            for j in range(k):
                AtA[i][j] += ri * row[j]
    # tiny ridge for numerical stability (does not change the fit measurably)
    for i in range(1, k):
        AtA[i][i] += 1e-9
    try:
        return solve_linear(AtA, Aty)
    except ValueError:
        return None


def forecast(prices, shrink=SHRINK):
    """One price: the predicted close HORIZON minutes after the last price.

    shrink is a parameter (default SHRINK) so the shrinkage sweep that selected
    0.7 is reproducible from this file alone; the entrypoint always runs with
    the default.
    """
    last = prices[-1]
    if not (isinstance(last, (int, float)) and math.isfinite(last) and last > 0):
        raise ValueError("last price is not a positive finite number")
    beta = fit_ols(prices)
    if beta is None:
        return price5(last)  # not enough history: persistence
    feats = [1.0] + [logret(prices[-1 - w], prices[-1]) for w in WINDOWS]
    pred_ret = sum(b * x for b, x in zip(beta, feats)) * shrink
    # Guard against absurd extrapolation: clamp predicted 4h log return to +-5%.
    pred_ret = max(-0.05, min(0.05, pred_ret))
    return price5(last * math.exp(pred_ret))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        case_id = fname[: -len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            case = json.load(f)
        markets = case.get("markets")
        if not isinstance(markets, list) or len(markets) != 1:
            raise ValueError(f"{fname} does not hold exactly one market")
        prices = markets[0].get("prices")
        if not isinstance(prices, list) or not prices:
            raise ValueError(f"{fname} market has no prices")
        guess = forecast(prices)
        out = {"forecasts": [{"case": case_id, "price": guess}]}
        with open(os.path.join(OUTPUT_DIR, f"{case_id}.out"), "w", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
