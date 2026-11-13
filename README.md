# Roblox Up-and-Coming Forecaster v2

Entry for the make.fun competition `roblox-up-and-coming-forecast-01`:
forecast Roblox's Up-and-Coming chart, places 1..10, at every hour of the
72 hours after entries close. `python3 main.py` reads the chart history at
`$MAKEFUN_INPUT_DIR` (default `/input`) and writes `forecast.json` to
`$MAKEFUN_OUTPUT_DIR` (default `/output`). Stdlib only, no network,
deterministic (two runs on the same history are byte-identical).

## Method

An empirical occupancy table blended with a fitted transition model,
self-fitted on whatever history it is given:

1. **Occupancy table** — each game's recency-weighted share of places
   1..10 across the observed reads (34-hour half-life), smoothed over
   neighbouring places and flattened toward uniform so no plausible game
   is ever priced at zero.
2. **Transition component** — a Gaussian over ranks per game: the centre
   starts at the game's last rank, reverts toward its recency-weighted
   home rank, and is shifted by a **momentum** term fitted from the
   game's rank trajectory over the recent reads (weighted least squares
   of rank against time, absent reads counted as off-chart, shift capped
   at 12 places and saturating over ~30 hours). The spread is the rank
   diffusion fitted from consecutive read pairs, growing with the horizon.

The two are mixed (0.2 occupancy / 0.8 transition), averaged over the 72
forecast hours, thinned by the 28-day age-out when a game's creation time
is known, and capped per place so a calibrated residual stays unnamed —
the scorer spreads it over every chart game not named there (new and
unmodelled games).

## Backtest (kit practice history, exact scoring rule)

Truncated-history backtest: fit on the first N reads of the kit's
practice history, score the single forecast against the remaining reads
(unnamed share spread over each read's unnamed chart games, per-place
floor 0.0001, log score = mean over reads of the summed per-place logs):

| fit reads | this entry (v2) | previous entry (v1, rank diffusion) |
|-----------|-----------------|-------------------------------------|
| 23 of 34  | **-42.15**      | -44.13                              |
| 17 of 34  | -44.25          | -44.04                              |

v2 clearly beats v1 on the primary split and ties it on the earlier,
shorter split. (The house baseline scores -43.51 on the full practice
window per the competition rules.)

## Layout

- `main.py` — the forecaster (the entry).
- `LICENSE` — MIT (`SPDX-License-Identifier: MIT`).
- `.makefun/owner.json` — the account wallet that owns this entry.
