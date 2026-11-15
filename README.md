# BTC Cent Forecast

[![make.fun · btc-cent-forecast-01](https://make.fun/badge/btc-cent-forecast-01.svg)](https://make.fun/bounty/btc-cent-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01)
[![rank](https://make.fun/badge/btc-cent-forecast-01/lanadelreiiiii/btc-cent-forecast-01/rank.svg)](https://make.fun/bounty/btc-cent-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01)
[![reward](https://make.fun/badge/btc-cent-forecast-01/lanadelreiiiii/btc-cent-forecast-01/reward.svg)](https://make.fun/bounty/btc-cent-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01)

Entry for make.fun `btc-cent-forecast-01`: forecast Bitcoin's USD price
4 hours ahead, 12 times over 48 hours. Scored by mean absolute error in
cents — lower is better — against a persistence baseline (guess the last
close). Prize goes to entries that beat the baseline.

## How it forecasts

Each case ships 72 hours of 1-minute closes. The model fits an OLS
regression on the case's own history: for every minute with at least 24
hours behind it and a full 240-minute lookahead inside the case, the target
is the log return over the next 240 minutes, and the features are a
constant plus the log returns over the last 60, 240, 720 and 1440 minutes.
The fitted model predicts the log return from the checkpoint, and the
guess is `last_price * exp(predicted return)`, with the predicted 4-hour
move clamped to ±5% and the price snapped to the scorer's grid.

Two things make it work. First, BTC shows strong short-horizon mean
reversion in this window: the past 24-hour return correlates about −0.5
with the next 4-hour return, so the model leans against the recent move.
Second, the constant term matters — a no-constant model scores *worse*
than persistence, because there is a real drift component. Fitting
per case lets the model adapt to each case's own regime instead of
hard-coding parameters from the practice window.

The fitted prediction is shrunk by 0.7. The shrinkage sweep on the 6
practice cases (below) peaks at 0.7: it gives up a little when the regime
persists, but it caps the damage when the regime flips — the p05 practice
case is the cautionary example, where the unshrunk model bets on a bounce
that never comes.

## Practice verification

Re-run 15 Nov 2026 on the 6 practice cases (p00–p05). Truth is the next
case's final price; p05 uses the Kraken BTC/USD 06:00-candle close
73,522.10 as a proxy. Reproduce it with `python3 main.py --verify <input dir>`.

| shrink | model MAE (cents) | persistence MAE | improvement |
|--------|-------------------|-----------------|-------------|
| 0.5    | 32,166            | 39,609          | +18.8%      |
| 0.6    | 30,683            | 39,609          | +22.5%      |
| 0.7    | 30,145            | 39,609          | +23.9%      |
| 0.8    | 30,306            | 39,609          | +23.5%      |
| 0.9    | 30,936            | 39,609          | +21.9%      |
| 1.0    | 32,006            | 39,609          | +19.2%      |

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Pure Python 3 standard library, deterministic, no network, under 0.1 s
per case. `python3 main.py --verify <input dir>` replays the practice
cases and prints the shrinkage sweep table above.

## License

MIT
