"""Build the event set, second pass.

The first pass got the event timing wrong, so this file documents the timing
carefully. In the corpus `trading_date` is always `date` plus one calendar day,
and `exact_trading_date` is the first trading day at or after THAT. So ts_0 is
the close of the day AFTER publication, not the publication day.

That matters because the news reaction lands before ts_0. Verified on Meta's
earnings releases: the 2022-02-02 after-close miss shows -26.4% from ts_-1 to
ts_0 and -0.3% from ts_0 to ts_1.

So there are two different questions, and this file computes both:

  reaction   ts_-1 -> ts_0. Where the news actually gets priced. NOT tradeable
             for news released after the close, because entry would be the
             evening before. Kept as a diagnostic: it is the window where a
             model that remembers what happened would show it most clearly.
  drift      ts_0 -> ts_1 and ts_0 -> ts_5. Entry is a full close after the
             reaction, so this is genuinely tradeable. This is the trading
             question.

Placebos, which must come out near zero if the plumbing is sound:

  pre        ts_-3 -> ts_-1, entirely before publication.
  far        ts_10 -> ts_11, long after the news is stale.

Benchmark is QQQ, not SPY: the universe is the Nasdaq-100, and the Nasdaq to
S&P spread over 2022-23 was large enough to leave a factor in the residual.
Date-demeaned returns are also computed, which removes any common factor
whatever it is.

No filtering on realised returns. The earlier version dropped rows whose move
exceeded a threshold, which is selection on the outcome, and the price series
turned out to have no bad ticks to justify it.
"""

import json
import re
import sys

import pandas as pd

N_EVENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
SEED = 20260916

# --- benchmark --------------------------------------------------------------

raw = json.load(open("qqq.json"))["chart"]["result"][0]
bench = pd.DataFrame(
    {
        "date": pd.to_datetime(raw["timestamp"], unit="s", utc=True)
        .tz_convert("America/New_York").normalize().tz_localize(None),
        "close": raw["indicators"]["quote"][0]["close"],
    }
).dropna().reset_index(drop=True)
bench["i"] = range(len(bench))
bclose = bench["close"].to_numpy()
bindex = dict(zip(bench["date"], bench["i"], strict=True))
print(f"QQQ calendar: {len(bench)} days, {bench['date'].min().date()} to {bench['date'].max().date()}")

# --- events -----------------------------------------------------------------

TS = ["ts_-5", "ts_-3", "ts_-2", "ts_-1", "ts_0", "ts_1", "ts_5", "ts_10", "ts_11"]
df = pd.read_parquet("news_test.parquet", columns=["date", "stock", "title", "exact_trading_date", *TS])
print(f"raw rows: {len(df)}")

# GOOG and GOOGL are the same company; keeping both double counts it.
df = df[df["stock"] != "GOOG"]

df = df.dropna(subset=["title", "exact_trading_date", *TS])
for c in TS:
    df = df[df[c] > 0]
df["event_date"] = pd.to_datetime(df["exact_trading_date"]).dt.normalize()
df["pub_date"] = pd.to_datetime(df["date"]).dt.normalize()
df["title"] = df["title"].str.strip()
df = df[df["title"].str.len() >= 25]
print(f"after basic filters: {len(df)}")

df = df.sample(frac=1.0, random_state=SEED).drop_duplicates(subset=["stock", "event_date"], keep="first")
print(f"one per stock-day: {len(df)}")

df = df[df["event_date"].isin(bindex)]
df["bi"] = df["event_date"].map(bindex)
df = df[(df["bi"] - 5 >= 0) & (df["bi"] + 11 < len(bench))]


def excess(num_col, den_col, off_num, off_den):
    """Stock return over a window, less the benchmark over the same offsets."""
    stock = df[num_col] / df[den_col] - 1
    market = bclose[df["bi"] + off_num] / bclose[df["bi"] + off_den] - 1
    return stock - market


df["reaction"] = excess("ts_0", "ts_-1", 0, -1)
df["drift1"] = excess("ts_1", "ts_0", 1, 0)
df["drift5"] = excess("ts_5", "ts_0", 5, 0)
# ts_-1 sits ON the publication day, so ts_-2 -> ts_-1 is the publication day
# move: where intraday news gets priced. The first attempt used ts_-3 -> ts_-1 as
# a placebo, which spans that move and so "predicted" the past at IC +0.21.
df["pubday"] = excess("ts_-1", "ts_-2", -1, -2)
# A real placebo has to end before the publication day entirely.
df["pre"] = excess("ts_-3", "ts_-5", -3, -5)
df["far"] = excess("ts_11", "ts_10", 11, 10)

print(f"usable events: {len(df)}, {df['stock'].nunique()} tickers, "
      f"{df['event_date'].min().date()} to {df['event_date'].max().date()}")
for c in ["pubday", "reaction", "drift1", "drift5", "pre", "far"]:
    print(f"  {c:<9} mean {df[c].mean()*100:+.3f}%  sd {df[c].std()*100:.2f}%")

# --- sample -----------------------------------------------------------------

sample = df.sample(n=min(N_EVENTS, len(df)), random_state=SEED).sort_values("event_date").copy()

# Cross-sectional demean within a day. Whatever moved every Nasdaq name that day
# comes out, including any factor the benchmark missed.
for c in ["pubday", "reaction", "drift1"]:
    sample[c + "_d"] = sample[c] - sample.groupby("event_date")[c].transform("mean")

# Many headlines simply state the move that already happened ("slumps as...").
# Reading one of those is description, not prediction, so flag them.
MOVE = (r"\b(jump|jumps|jumped|soar|soars|soared|surge|surges|surged|rally|rallies|"
        r"rallied|climb|climbs|climbed|gain|gains|gained|rise|rises|rose|pop|pops|"
        r"spike|spikes|spiked|slump|slumps|slumped|fall|falls|fell|drop|drops|dropped|"
        r"plunge|plunges|plunged|sink|sinks|sank|tumble|tumbles|tumbled|slide|slides|"
        r"slid|decline|declines|declined|dip|dips|dipped|crash|crashes|crashed|"
        r"sheds|retreat|retreats|higher|lower|outperform|underperform)\b")
sample["move_word"] = sample["title"].str.contains(MOVE, case=False, regex=True)
print(f"headlines stating a move: {sample['move_word'].mean()*100:.1f}%")

# --- anonymisation ----------------------------------------------------------

names = json.load(open("tickers.json"))
aliases = json.load(open("aliases2.json"))

SUFFIXES = (r"(?:,?\s+(?:Inc|Incorporated|Corp|Corporation|Company|Co|Ltd|plc|PLC|Holdings|"
            r"Group|Technologies|Systems|International|N\.V\.|NV|S\.A\.|SA|AG)\.?)+$")

# Words that are also ordinary English. Replacing these wrecks the sentence and
# tells the reader nothing, so a single-token variant matching one is skipped.
COMMON = {"first", "united", "american", "general", "global", "national", "advanced",
          "block", "match", "gap", "target", "arista", "public", "old", "new", "on",
          "cost", "fast", "team", "mar", "ea", "mu", "so", "it", "at", "all", "one",
          "now", "next", "open", "live", "prime", "charter", "discovery", "booking"}


def name_variants(ticker: str) -> list[str]:
    """Ways the subject company's name is written, longest first."""
    out = set(aliases.get(ticker, {}).get("names", []))
    full = names.get(ticker, "")
    if full:
        out.add(full)
        base = re.sub(SUFFIXES, "", full).strip().rstrip(",")
        if base:
            out.add(base)
            head = base.split()[0]
            if len(head) >= 4 and head.lower() not in COMMON:
                out.add(head)
    return sorted({v.strip() for v in out if len(v.strip()) >= 3}, key=len, reverse=True)


def sub_token(text: str, needle: str, replacement: str, ignore_case: bool) -> str:
    """Replace a name, tolerating a trailing possessive and surrounding punctuation."""
    flags = re.IGNORECASE if ignore_case else 0
    pattern = rf"(?<!\w){re.escape(needle)}(?:['’]s)?(?!\w)"
    return re.sub(pattern, replacement, text, flags=flags)


def anonymise(title: str, ticker: str) -> str:
    out = title
    # The ticker itself, matched case sensitively so the ticker ON does not eat
    # every occurrence of the word "on".
    out = sub_token(out, ticker, "the company", ignore_case=False)
    for v in name_variants(ticker):
        single = " " not in v
        if single and v.lower() in COMMON:
            continue
        out = sub_token(out, v, "the company", ignore_case=True)
    for p in aliases.get(ticker, {}).get("products", []):
        out = sub_token(out, p, "its product", ignore_case=True)
    out = re.sub(r"\b(the company)(\s*,?\s*the company)+\b", r"\1", out, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", out).strip()


sample["anon_title"] = [anonymise(t, s) for t, s in zip(sample["title"], sample["stock"], strict=True)]
sample["anon_changed"] = sample["anon_title"] != sample["title"]

# A third arm: the anonymised headline handed a different company's identity.
# If the name carries real information, this should not match the named arm.
rng = pd.Series(sample["stock"].unique())
shuffled = sample["stock"].sample(frac=1.0, random_state=SEED + 1).to_numpy()
same = shuffled == sample["stock"].to_numpy()
shuffled[same] = pd.Series(shuffled[same]).apply(
    lambda s: rng[rng != s].sample(1, random_state=SEED + 2).iloc[0]
).to_numpy()
sample["wrong_stock"] = shuffled

print(f"\nanonymisation changed {sample['anon_changed'].mean()*100:.1f}% of headlines")
for _, r in sample[sample["anon_changed"]].head(5).iterrows():
    print(f"  [{r['stock']}] {r['title'][:76]}")
    print(f"        -> {r['anon_title'][:76]}")

cols = ["stock", "wrong_stock", "event_date", "pub_date", "title", "anon_title", "anon_changed",
        "move_word", "pubday", "reaction", "drift1", "drift5", "pre", "far",
        "pubday_d", "reaction_d", "drift1_d"]
out = sample[cols].copy()
out["event_date"] = out["event_date"].dt.strftime("%Y-%m-%d")
out["pub_date"] = out["pub_date"].dt.strftime("%Y-%m-%d")
out.to_json("events3.json", orient="records")
print(f"\nwrote {len(out)} events to events3.json")
