# GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/buahrasa01-hub/makefun-gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/buahrasa01-hub/makefun-gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

## Run

Requires Python 3.13 and the standard library only. The program reads all .in files from MAKEFUN_INPUT_DIR (default /input) in filename order and writes <id>.out files to MAKEFUN_OUTPUT_DIR (default /output).

    mkdir -p out
    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

Each output contains five nondecreasing earn-rate quantiles for every trophy. The program is offline and deterministic. If a case cannot be modeled, it emits the persistence forecast with the baseline random-walk spread.

## Model

Rates are clamped to 0.01–99.99% before conversion to logit. For each trophy, the model keeps observations from the last 24 hours, collapses repeated identical rate/tracked-count states to their latest timestamp, and fits weighted least-squares lines against x = ln(1 + hours since 2026-11-19 00:00 UTC). Recency weights rise linearly from 0.45 to 1.0. The logit-rate fit supplies 65% of the median signal.

The other 35% comes from a separate projection of estimated earned counts and tracked players. Estimated earned count is rate × players_tracked / 100; weighted log-linear fits forecast earned count and tracked count independently, then their ratio is converted back to a rate. This uses players_tracked directly so growth in the denominator can dilute the forecast. With too few distinct observations, the median falls back to the last observed rate.

The logit spread combines the random-walk floor 0.5 × sqrt(horizon_days), predictive residual uncertainty from the rate and count fits, a small tracked-growth term, and a horizon-scaled refresh term. When the whole page has been unchanged for at least two hours, a shared refresh-risk component widens the spread further. Quantiles use normal scores [-1.645, -0.674, 0, 0.674, 1.645] and are clamped to 0–100%.

At every quantile, platinum is capped at the lowest required trophy rate. Trophy IDs 3 through 7 are projected by equal-weight pool-adjacent-violators so the chapter/finale rates cannot increase along the chain. Output is checked for IDs, row count, finite ordered quantiles, and size before writing.

<sub>Entered on [make.fun](https://make.fun/?ref=pnwd2w29&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
