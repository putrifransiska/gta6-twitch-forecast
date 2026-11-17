# GTA VI launch-day Twitch viewers — analog-ensemble quantile forecast

[![make.fun bounty](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/putrifransiska/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/putrifransiska/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

An entry for make.fun's [Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
bounty, built by an AI agent. Given hour 1 of release day (00:00–01:00 UTC), `main.py` forecasts the 5/25/50/75/95%
quantiles of the Twitch category's concurrent viewers for every hour 2–24.

## How it works

The forecast is built from **comparables**: the release-day curves of earlier big launches.

1. **Curves.** Each comparable becomes a log-ratio curve `r_c(h) = ln(v_h / v_1)` for hours 2–24 (UTC). Gaps are
   linearly interpolated, and the target's own pre-release category history is skipped.
2. **Who to trust.** GTA VI is a console-only, rolling-midnight-local launch. Console-only comparables (Ghost of
   Tsushima, Ghost of Yōtei, The Last of Us Part II, Stellar Blade, Zelda TotK, …) share one daily rhythm: a dip while
   Europe sleeps (hours 2–6), then a climb through the American afternoon and evening. Launches that include PC often
   unlock once, globally, mid-day, and explode by ×100 or more. They get a weight of 0.02.
3. **Run-up adjustment.** `x = ln(mean of the 6 hours before hour 1 / hour 1)` measures how far viewing fell into hour 1.
   A weighted ridge regression of `r_c(h)` on `x`, run separately for each hour, shifts every analog curve to the
   target's `x`.
4. **Quantiles.** Hour h is a weighted mixture of normals centred on the shifted curves, with kernel width
   `0.12 + 0.08·sqrt(h−1)`, so the band widens as the horizon grows. The mixture's quantiles are found by bisection and
   multiplied by hour 1's print.
5. **If the unlock turns out global.** Pre-hour-1 data cannot tell a global unlock apart from a rolling one; only a tiny
   hour 1 did (every launch that rose 10–1000× had an hour-1 print under 5,000). PC/global analogs therefore get full
   weight when the case's notes describe one simultaneous global unlock, and gain weight smoothly when hour 1 is far
   below 2,500 viewers. An analog's level is capped at 8 million viewers, because a ×1000 surge only happens from a
   tiny base.

Standard library only, no network, no randomness (byte-identical reruns), about 0.1 s per case.

## Validation

Leave-one-out backtest: each comparable plays the target in turn, forecast from the others. Score is the contest's
log-scale pinball loss (lower is better).

| set | persistence baseline | this model |
|---|---|---|
| console-only launches (n = 8, the target's kind) | 0.323 | **0.204** |
| launches that surged 10–1000× (n = 8) | 3.00 | 1.90 |

The console-only subset is the proxy for the practice cases (Wolverine, RDR2, God of War Ragnarök, Spider-Man 2).
There the house baseline scores 0.387. The parameters were tuned on that same subset. The best point scored 0.193.
The chosen settings use a slightly wider spread and the global-unlock safeguards, at a cost of about 0.01 there, for a
better worst case.

Also tested and left off: aligning each comparable to the target's clock by its DST difference (GTA VI unlocks in the
US at 05:00 UTC on 19 November, and a summer launch at 04:00 UTC). The console-only score went from 0.204 to 0.208,
with 4 cases better and 4 worse. The one winter console launch, FF7 Rebirth, lost the most. The gain on a wider set
was mixed, so the switch (`dst_align`) stays off.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` in the input directory (default `/input`) and writes `<id>.out` to the output directory
(default `/output`).

## License

MIT, see [LICENSE](LICENSE).
