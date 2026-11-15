# BTC Cent Forecast — make.fun btc-cent-forecast-01

Forecast Bitcoin's US-dollar price 4 hours ahead, 12 times over 48 hours.
Score: mean absolute error in cents (lower is better). Must beat the house
baseline (persistence: guess = last close) to take a prize.

## Method

Per-case OLS on the case's own 72 hours of 1-minute closes. For each minute
with >= 24 h of history and a full 240-minute lookahead inside the case, the
target is the next 240-minute log return; features are the constant plus log
returns over the past 60 / 240 / 720 / 1440 minutes. The fitted model predicts
the log return from the checkpoint; the guess is `last_price * exp(pred)`,
clamped to a +-5% move and to the scorer's price grid.

Why this works: BTC 1-minute closes show strong short-horizon mean reversion
(past 24 h return correlates ~-0.5 with the next 4 h return in the practice
window) plus a persistent drift component. A no-constant model is *worse* than
persistence — the drift term is essential. Walk-forward validation on the practice checkpoints (fit only on data before
each checkpoint): mean MAE ~29.6k cents vs ~47.1k for persistence (**+37%**
unshrunk, 5 known cases). With shrinkage 0.7 (applied to the fitted prediction
to limit damage when the regime flips) the 6-case practice score is +23.9% vs
persistence. Fitting per case adapts to each case's own regime instead of
hard-coding practice-window parameters.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Pure Python 3 standard library, deterministic, no network, < 0.1 s per case.

## License

MIT
