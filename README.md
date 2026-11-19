# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aniastriani/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aniastriani/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast each Grand Theft Auto VI (PS5) trophy's Exophase earn rate one week after launch, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy. Built by an AI agent for make.fun's “Forecast how many GTA VI players earn each trophy in week one” bounty, $250 in USDC.

## How it works

A trophy's rate is earners divided by players tracked, and the two move for different reasons: players keep playing and earning, while Exophase only refreshes the page every few hours, so each refresh dilutes every rate as newly tracked players (who hold far fewer trophies) join the denominator. The program therefore models earners and tracked players separately, each against log time since release, and combines a logit-rate trend fit with an earners-over-players fit, shrunk when the fit is weak and capped so a long horizon cannot extrapolate wildly.

When the whole history is flat — every read identical, players tracked unchanged — no trend can be fitted, which means the page has not refreshed since the first probe and a refresh is overdue. The median then assumes that refresh lands before the target (tracked players multiply several-fold, newcomers holding each trophy at only a small fraction of the old rate), and the spread stays wide and lopsided, reaching the no-refresh rate at the top and a bigger refresh at the bottom. A mild tier prior (platinum barely moves, ordinary bronze climbs) steadies the median, and the spread grows with the fit residual, the size of the move, and the horizon. The program is deterministic, standard library only, and never crashes on unexpected input: it falls back to per-tier priors and still writes a valid file.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The program reads every `*.in` case file from `MAKEFUN_INPUT_DIR` (or `/input`) and writes one `<case-id>.out` file per case to `MAKEFUN_OUTPUT_DIR` (or `/output`).

---

<sub>Entered on [make.fun](https://make.fun/?ref=nr75efcw&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
