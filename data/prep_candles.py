"""Build a price-only event set: 30 daily candles in, next day's return out.

This is a cleaner trading test than the news one. A price pattern is fully known
at the close of day 0, and the trade is day 0 close to day 1 close, so there is
no question about whether the information was available in time.

Two things make it a better contamination test as well:

  normalisation  candles are rescaled so the first close is 100 and volume is a
                 multiple of its own median. Price level is one of the strongest
                 cues for identifying a stock and a date, and it is gone.
  two eras       2022-23 is certainly inside the model's training data. 2025-26
                 is recent enough to be at or past its cutoff. If an edge exists
                 only in the old era, it is recall.

A scrambled copy of every window is also emitted: the same daily returns in a
shuffled order. That keeps the volatility and the return distribution while
destroying every temporal pattern, so it separates reading the shape from
reacting to how choppy it looks.
"""

import json
import sys

import numpy as np
import pandas as pd

LOOKBACK = 30
PER_ERA = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
SEED = 20260916

ERAS = {
    "seen": ("2022-01-01", "2023-12-31"),
    "recent": ("2025-07-01", "2026-09-10"),
}

raw = json.load(open("ohlcv.json"))


def frame(t: str) -> pd.DataFrame:
    v = raw[t]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(v["ts"], unit="s", utc=True)
            .tz_convert("America/New_York").normalize().tz_localize(None),
            "open": v["open"], "high": v["high"], "low": v["low"],
            "close": v["close"], "volume": v["volume"],
        }
    ).dropna()
    return df[df["volume"] > 0].reset_index(drop=True)


bench = frame("QQQ").set_index("date")["close"]
bench_ret = bench.pct_change().shift(-1)  # next day's benchmark return

rng = np.random.default_rng(SEED)
events = []
frames: dict[str, pd.DataFrame] = {}

for ticker in sorted(raw):
    if ticker == "QQQ":
        continue
    df = frame(ticker)
    if len(df) < LOOKBACK + 260:
        continue
    df["ret"] = df["close"].pct_change()
    df["fwd"] = df["close"].pct_change().shift(-1)
    df["vol20"] = df["ret"].rolling(20).std()
    df["ret5"] = df["close"].pct_change(5)
    # Momentum the way it is normally defined: the last year, skipping the most
    # recent month, because that month is reversal rather than momentum.
    df["mom"] = df["close"].shift(21) / df["close"].shift(250) - 1
    df["bfwd"] = df["date"].map(bench_ret)
    df["excess"] = df["fwd"] - df["bfwd"]
    frames[ticker] = df

    for era, (lo, hi) in ERAS.items():
        mask = (df["date"] >= lo) & (df["date"] <= hi)
        idx = df.index[mask]
        idx = idx[(idx >= LOOKBACK) & (idx < len(df) - 1)]
        for i in idx:
            row = df.loc[i]
            if not np.isfinite([row["excess"], row["vol20"], row["ret5"], row["mom"], row["ret"]]).all():
                continue
            events.append((ticker, era, int(i)))

print(f"candidate stock-days: {len(events)}")

by_era = {}
for t, era, i in events:
    by_era.setdefault(era, []).append((t, i))

chosen = []
for era, rows in by_era.items():
    take = min(PER_ERA, len(rows))
    pick = rng.choice(len(rows), size=take, replace=False)
    chosen += [(rows[p][0], era, rows[p][1]) for p in pick]
    print(f"  {era}: {len(rows)} available, taking {take}")


def window(d: pd.DataFrame, i: int) -> dict:
    """Thirty candles ending at day i, rescaled so the first close is 100."""
    w = d.iloc[i - LOOKBACK + 1 : i + 1]
    base = w["close"].iloc[0]
    med = w["volume"].median()
    return {
        "o": (w["open"] / base * 100).round(2).tolist(),
        "h": (w["high"] / base * 100).round(2).tolist(),
        "l": (w["low"] / base * 100).round(2).tolist(),
        "c": (w["close"] / base * 100).round(2).tolist(),
        "v": (w["volume"] / med).round(2).tolist(),
    }


def scramble(win: dict, seed: int) -> dict:
    """Same daily returns in a shuffled order, rebuilt into a price path.

    Volatility and the spread of returns survive; trend, reversal and every
    other shape do not.
    """
    r = np.random.default_rng(seed)
    c = np.array(win["c"])
    rets = c[1:] / c[:-1] - 1
    # Each day's intraday range as a fraction of its own close, carried along so
    # the candles still look like candles.
    o = np.array(win["o"]) / c
    h = np.array(win["h"]) / c
    l = np.array(win["l"]) / c
    order = r.permutation(len(rets))
    rets = rets[order]
    path = [100.0]
    for x in rets:
        path.append(path[-1] * (1 + x))
    path = np.array(path)
    shift = np.concatenate([[0], order + 1])
    return {
        "o": (path * o[shift]).round(2).tolist(),
        "h": (path * h[shift]).round(2).tolist(),
        "l": (path * l[shift]).round(2).tolist(),
        "c": path.round(2).tolist(),
        "v": np.array(win["v"])[shift].round(2).tolist(),
    }


out = []
for n, (ticker, era, i) in enumerate(chosen):
    d = frames[ticker]
    row = d.loc[i]
    win = window(d, i)
    out.append({
        "stock": ticker,
        "era": era,
        "date": row["date"].strftime("%Y-%m-%d"),
        "candles": win,
        "scrambled": scramble(win, SEED + n),
        # Forward excess return over QQQ, day 0 close to day 1 close.
        "fwd": float(row["close"] and (d.loc[i + 1, "close"] / row["close"] - 1)),
        "excess": float(row["excess"]),
        # Mechanical baselines, computed in code, to be beaten.
        "rev1": float(-row["ret"]),
        "rev5": float(-row["ret5"]),
        "mom": float(row["mom"]),
        "vol20": float(row["vol20"]),
    })

out.sort(key=lambda e: (e["era"], e["date"], e["stock"]))
json.dump(out, open("candles.json", "w"))

df = pd.DataFrame([{k: e[k] for k in ("era", "excess", "rev1", "rev5", "mom")} for e in out])
print(f"\nwrote {len(out)} events to candles.json")
for era, g in df.groupby("era"):
    print(f"  {era:<7} n={len(g)}  next-day excess mean {g['excess'].mean()*100:+.3f}%  sd {g['excess'].std()*100:.2f}%")
