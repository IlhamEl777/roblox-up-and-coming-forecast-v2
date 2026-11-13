#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster v2 (make.fun program/v1).

Reads the chart history in /input (history.ndjson + task.json) and writes one
forecast to /output/forecast.json: per-place probabilities for places 1..10
plus a top-10 guess, scored by log score against every hourly chart of the
72 hours after the history ends.

Method: an empirical occupancy table blended with a fitted transition
model, self-fitted on whatever history it is given (the occupancy-table
style the top of the practice leaderboard uses). Two components are mixed
per game and place. (1) An occupancy table: the game's recency-weighted
share of each place 1..10 across the observed reads, smoothed over
neighbouring places and flattened toward uniform so no plausible game is
ever priced at zero. (2) A transition component: a Gaussian over ranks
whose centre starts at the game's last rank, reverts toward its
recency-weighted home rank, and is shifted by a momentum term fitted from
the game's rank trajectory over the recent reads (weighted least squares
of rank against time, absent reads counted as off-chart); its spread is
the rank diffusion fitted from consecutive read pairs, growing with the
horizon. The mixture is averaged over the 72 forecast hours, thinned by
the 28-day age-out when the game's creation time is known, and capped per
place so a calibrated residual stays unnamed: the scorer spreads it over
every game on the read's chart that was not named there (new and
unmodelled games).

Deterministic: pure arithmetic, no randomness, no network, stdlib only.
"""

import json
import math
import os
import sys
from datetime import datetime

# Tunable constants (fitted on truncated-history backtests of the kit's
# practice history; see README).
HL_HOURS = 34.0      # recency half-life of a read, in hours
MOM_SPAN = 10        # reads used for the rank-trajectory fit
MOM_CAP = 12.0       # max places of momentum shift over the full window
MOM_TAU = 30.0       # hours; momentum shift saturates like 1-exp(-h/tau)
OCC_FLAT = 0.025     # per-candidate flattening of the occupancy table
OCC_SMOOTH = 0.35    # weight of neighbouring places in the occupancy table
MIX_OCC = 0.2        # mixture weight of the occupancy component
SD0 = 1.5            # rank spread at horizon 0 for the transition component
SIGMA_SCALE = 0.85   # scale on the fitted rank-diffusion sigma
LAM = 0.02           # mean reversion of the transition centre, per hour
Q0 = 0.05            # unnamed residual at place 1
Q1 = 0.008           # extra unnamed residual per place step
MIN_P = 0.002        # rows below this are left to the unnamed pool
OFFCHART = 115.0     # rank imputed for a read where the game is absent
MAX_CAND_RANK = 45   # candidates: weighted presence up to this rank


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_input(indir):
    task = {}
    tpath = os.path.join(indir, "task.json")
    if os.path.exists(tpath):
        with open(tpath) as f:
            task = json.load(f)
    reads = []
    created = {}
    with open(os.path.join(indir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            chart = []
            for g in r.get("up_and_coming") or []:
                chart.append({
                    "id": int(g["universe_id"]),
                    "rank": int(g["rank"]),
                    "playing": g.get("playing") or 0,
                })
            chart.sort(key=lambda g: g["rank"])
            reads.append({"t": parse_time(r["at"]), "chart": chart})
            for key, d in (r.get("details") or {}).items():
                if d and d.get("created"):
                    try:
                        created[int(key)] = parse_time(d["created"])
                    except (ValueError, TypeError):
                        pass
    reads.sort(key=lambda r: r["t"])
    return task, reads, created


def build_tables(reads):
    """Recency-weighted occupancy per game + marginal occupancy per place."""
    t_last = reads[-1]["t"]
    weights = [0.5 ** ((t_last - r["t"]) / 3600.0 / HL_HOURS) for r in reads]
    wsum = sum(weights)
    occ = {}       # u -> [w at places 1..10]
    rank_w = {}    # u -> (weighted rank sum, weight sum) over reads present
    last_rank = {}
    for i, r in enumerate(reads):
        w = weights[i]
        for g in r["chart"]:
            u = g["id"]
            if g["rank"] <= 10:
                occ.setdefault(u, [0.0] * 11)[g["rank"]] += w
            s, sw = rank_w.get(u, (0.0, 0.0))
            rank_w[u] = (s + w * g["rank"], sw + w)
            last_rank[u] = g["rank"]
    marginal = [0.0] * 11  # share of total weight each game holds per place
    for u, row in occ.items():
        for k in range(1, 11):
            marginal[k] += row[k]
    for k in range(1, 11):
        marginal[k] /= wsum
    return occ, rank_w, last_rank, marginal, weights


def momentum(reads, weights):
    """Weighted least squares slope of rank vs time (places per hour) over
    the recent reads; absent reads count as OFFCHART. Negative = climbing."""
    span = reads[-MOM_SPAN:]
    ids = set()
    for r in span:
        for g in r["chart"]:
            ids.add(g["id"])
    slopes = {}
    for u in ids:
        xs, ys, ws = [], [], []
        for r in span:
            rk = OFFCHART
            for g in r["chart"]:
                if g["id"] == u:
                    rk = float(g["rank"])
                    break
            xs.append((r["t"] - span[0]["t"]) / 3600.0)
            ys.append(rk)
            ws.append(1.0)
        sw = sum(ws)
        mx = sum(w * x for w, x in zip(ws, xs)) / sw
        my = sum(w * y for w, y in zip(ws, ys)) / sw
        sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
        if sxx <= 0:
            slopes[u] = 0.0
            continue
        sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
        slopes[u] = sxy / sxx
    return slopes


def candidates(reads, occ, rank_w):
    lastmap = {g["id"]: g["rank"] for g in reads[-1]["chart"]}
    cands = set()
    for g in reads[-1]["chart"]:
        if g["rank"] <= MAX_CAND_RANK:
            cands.add(g["id"])
    for r in reads[-4:]:
        for g in r["chart"]:
            if g["rank"] <= 25:
                cands.add(g["id"])
    for u, row in occ.items():
        if sum(row[1:]) > 0:
            cands.add(u)
    return cands, lastmap


def fit_sigma(reads):
    """Rank diffusion per sqrt-hour, fitted from consecutive read pairs."""
    diffs = []
    for a, b in zip(reads, reads[1:]):
        dt = (b["t"] - a["t"]) / 3600.0
        if dt <= 0 or dt > 48:
            continue
        bmap = {g["id"]: g["rank"] for g in b["chart"]}
        for g in a["chart"]:
            r2 = bmap.get(g["id"])
            if r2 is not None and g["rank"] <= 60:
                diffs.append((r2 - g["rank"]) / math.sqrt(dt))
    if len(diffs) < 20:
        return 4.5
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    return min(9.0, max(2.0, math.sqrt(var)))


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def forecast(reads, hours, created=None):
    occ, rank_w, last_rank, marginal, weights = build_tables(reads)
    slopes = momentum(reads, weights)
    sigma = fit_sigma(reads) * SIGMA_SCALE
    cands, lastmap = candidates(reads, occ, rank_w)
    wsum_all = sum(weights)
    t0 = reads[-1]["t"]
    places = 10
    # Occupancy component: smoothed, flattened share of place k per game.
    occ_comp = {}
    for u in cands:
        row = occ.get(u, [0.0] * 11)
        occ_comp[u] = [0.0] * (places + 1)
        for k in range(1, 11):
            s = row[k] / wsum_all
            if k > 1:
                s += OCC_SMOOTH * row[k - 1] / wsum_all
            if k < 10:
                s += OCC_SMOOTH * row[k + 1] / wsum_all
            occ_comp[u][k] = s + OCC_FLAT
    for k in range(1, 11):
        tot = sum(occ_comp[u][k] for u in cands)
        if tot > 0:
            for u in cands:
                occ_comp[u][k] /= tot
    # Transition component: Gaussian around a centre that starts at the
    # last rank, reverts toward the recency-weighted home rank, and is
    # shifted by momentum; spread grows with the fitted diffusion.
    acc = {u: [0.0] * (places + 1) for u in cands}
    for u in cands:
        s, sw = rank_w.get(u, (0.0, 0.0))
        home = s / sw if sw else float(lastmap.get(u, 60))
        r0 = float(lastmap.get(u, min(60.0, home + 10.0)))
        slope = slopes.get(u, 0.0)
        born = (created or {}).get(u)
        max_h = hours
        if born:
            max_h = max(0, min(hours, int((born + 28 * 86400 - t0) // 3600)))
        for h in range(1, hours + 1):
            if h > max_h:
                continue
            decay = math.exp(-LAM * h)
            shift = max(-MOM_CAP, min(MOM_CAP,
                        slope * MOM_TAU * (1.0 - math.exp(-h / MOM_TAU))))
            mu = home + (r0 - home) * decay + shift
            sd = math.sqrt(SD0 * SD0 + sigma * sigma * h)
            prev = phi((0.5 - mu) / sd)
            for k in range(1, 11):
                cur = phi((k + 0.5 - mu) / sd)
                g = cur - prev
                prev = cur
                acc[u][k] += MIX_OCC * occ_comp[u][k] + (1.0 - MIX_OCC) * g
    for u in acc:
        for k in range(1, places + 1):
            acc[u][k] /= hours
    rows = []
    for k in range(1, places + 1):
        total = sum(acc[u][k] for u in acc)
        cap = 1.0 - (Q0 + Q1 * (k - 1))
        scale = cap / total if total > cap else 1.0
        for u in acc:
            p = acc[u][k] * scale
            if p >= MIN_P:
                rows.append({"universe_id": u, "place": k, "p": round(p, 6)})
    rows.sort(key=lambda r: (r["place"], -r["p"]))
    top_score = sorted(
        acc.items(),
        key=lambda kv: (
            sum(kv[1][k] * k for k in range(1, 11)) / max(sum(kv[1][1:]), 1e-9),
            -sum(kv[1][1:]),
        ),
    )
    top10 = [u for u, _ in top_score[:10]]
    return {"probabilities": rows, "top10": top10}


def main():
    indir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    outdir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, reads, created = load_input(indir)
    if not reads:
        result = {"probabilities": [], "top10": []}
    else:
        hours = 72
        win = task.get("window") or {}
        if win.get("slots"):
            hours = int(win["slots"])
        result = forecast(reads, hours, created)
    os.makedirs(outdir, exist_ok=True)
    outname = task.get("output") or "forecast.json"
    with open(os.path.join(outdir, outname), "w") as f:
        json.dump(result, f, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
