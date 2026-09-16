# jev-alpha-bench

Does [Jev](https://typesafe.ai), TypeSafe's System One model, predict stock
returns from financial news? Built on [jev-go](https://github.com/Gaurav-Gosain/jev-go).

**Short answer: it reads the news well, and there is no money in it.**

On the day a headline is published, Jev's bullish-minus-bearish tilt has a rank
IC of **+0.24** against that day's move. By the time you can actually trade on
it, one full close later, the IC is **-0.008** and a long-short book loses
**18 bps** a trade after costs.

That gap is the whole result. The news is priced while it is being published.

Run on 2026-09-16 against `jev-1.13.0`. 5,000 events, 15,000 requests, three
arms. Raw per-event output is in [`results/`](results/).

## What was measured

Nasdaq-100 headlines from 2022-01 to 2023-10, one per stock per day, from
`CyberNative`-adjacent corpus `oliverwang15/us_stock_news_with_price`. Each
headline gets one request carrying three questions: a `Choice` over
bullish/bearish/neutral, a `Score` for how big a move it justifies, and a `Noul`
for whether it is already priced in. The pre-registered signal is the simplest
one available, `P(bullish) - P(bearish)`.

Returns are excess over QQQ across matching trading-day offsets. Confidence
intervals come from a bootstrap that resamples **dates**, not events, because
events on the same day share the market's move.

### The timing, which is the whole experiment

The corpus sets `trading_date = date + 1 calendar day`, so `ts_0` is the close
of the day **after** publication, not the publication day. That single fact
decides what is tradeable:

| window | what it is | tradeable |
| --- | --- | --- |
| `pubday` = ts_-2 → ts_-1 | the publication day. Intraday news prices here. | no, the headline does not exist yet at entry |
| `reaction` = ts_-1 → ts_0 | the next close. After-close news prices here. | no, entry is the evening before |
| `drift1` = ts_0 → ts_1 | a full close after the reaction | **yes** |

Verified on Meta's earnings: the 2022-02-02 after-close miss moves **-26.4%**
from ts_-1 to ts_0 and **-0.3%** from ts_0 to ts_1.

## Results

### It reads the news

| window | named | blind | wrong identity |
| --- | --- | --- | --- |
| publication day | **+0.2374** | +0.2075 | +0.1720 |
| reaction | +0.0615 | +0.0614 | +0.0535 |
| **drift1 (tradeable)** | **-0.0083** | -0.0126 | -0.0049 |

All three non-tradeable windows are significant. Every tradeable one is flat.

### There is no money in it

```
NAMED / tilt / drift1   n=5000
  rank IC        -0.0083  [-0.0377, +0.0214]  not distinguishable from zero
  null IC        -0.0037  [-0.0312, +0.0234]  (shuffled signal)
  long-short     +1.6 bps gross, -18.4 bps net of cost
  hit rate       49.5%
```

The gross spread is +1.6 bps. Costs are 20 bps for the pair. The strategy is
significantly *unprofitable*, which is the only significant result on the
tradeable side. Same at five days, same after demeaning within the date.

## Three controls, because a backtest is easy to fool

**Contamination.** Jev was trained on text covering 2022-23, so on a named
headline it could recall what the stock did rather than read anything. Three
arms test this: the headline as published with the ticker, an anonymised
headline with no identity, and an anonymised headline handed a **different
company's** name.

On headlines that actually named their subject, reaction window:

```
named  IC +0.0998    blind  IC +0.0990    wrong  IC +0.0972
```

Identity contributes nothing. Telling the model it is looking at the wrong
company barely moves the number. Whatever it is doing is reading, not recall.

**Description, not prediction.** A headline saying "slumps as..." can be scored
correctly with no judgment at all. Dropping the 25% of headlines containing a
move word:

| window | all | no move word |
| --- | --- | --- |
| publication day | +0.2374 | +0.1306 |
| reaction | +0.0615 | +0.0539 |

About half the publication-day number is headlines narrating the move. The rest
survives, and the reaction window barely moves.

**Placebos.** Windows the headline cannot possibly inform:

```
placebo/pre   ts_-5 -> ts_-3, ends before publication   IC +0.029  flat
placebo/far   ts_10 -> ts_11, long after the news       IC +0.006  flat
null          signal shuffled against outcomes          IC -0.004  flat
```

The first version of this placebo ran ts_-3 → ts_-1 and scored **+0.21**. That
window spans the publication day, so it was not a placebo at all. Finding that
is what located the timing issue above.

## Reproducing

```bash
export TYPESAFE_API_KEY=...
uv run --with pandas --with pyarrow python data/prep3.py 5000   # build the event set
go run ./cmd/jev-alpha -run -events data/events3.json           # 15,000 requests, ~5.5 min
go run ./cmd/jev-alpha -events data/events3.json                # re-score without the API
go test ./...
```

`-rejoin` swaps in a corrected return window on saved runs without re-calling
the API. Redefining a window does not change what the model answered, so a fix
costs nothing to apply.

```
cmd/jev-alpha        runner and report
internal/study       arms, battery, scoring, bootstrap, portfolio
data/prep3.py        event construction, the file to argue with
results/             per-event output for all three arms
```

`internal/study/stats_test.go` plants a known IC of 0.10 and checks the
machinery recovers it, checks a shuffled signal comes back flat, and checks that
clustering by date widens intervals versus treating events as independent. A
backtester that cannot pass those is not worth pointing at a model.

## What would change the answer

**This does not test intraday trading.** The publication-day IC of +0.24 is
real; it is just unreachable at daily close-to-close granularity. With
timestamped headlines and intraday prices, the question of whether any of that
is capturable in the minutes after publication is open, and is the obvious next
experiment. The corpus has no timestamps, so it cannot be asked here.

**It does not test Jev as a research tool.** Reading a thousand headlines and
returning a calibrated direction in 340ms is useful for triage, routing, or
flagging, whatever the tradeability.

## Caveats

Single model version, single run. The universe is 99 Nasdaq-100 tickers as
constituted at corpus build time, so names dropped or acquired during 2022-23
are missing; for a cross-sectional long-short this mostly cancels, but the
"bad news then delisted" tail is absent. Anonymisation is partial: context can
still identify a company, and 6-7% of anonymised headlines retain an
identifying product or person. Costs are assumed at 10 bps round trip per leg,
reasonable for these names and possibly generous. `drift5` windows overlap
across adjacent events for the same stock, so its intervals are optimistic.
At n=5000 the smallest IC separable from zero is about 0.040, so this rules out
a tradeable effect of that size, not every effect.

Design reviewed adversarially before the numbers were read; that review found
the timing issue, the placebo specification, and an anonymiser bug where the
ticker `ON` replaced every occurrence of the word "on".

## License

MIT
