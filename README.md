# jev-alpha-bench

Does [Jev](https://typesafe.ai), TypeSafe's System One model, predict stock
returns from financial news? Built on [jev-go](https://github.com/Gaurav-Gosain/jev-go).

**Short answer: it reads the news well, it cannot read a chart, and there is no
money in either.**

Two studies. The first gives it news headlines. The second gives it price
candles and no news at all.

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

## Study 2: candles only, no news

`cmd/jev-candles`. Thirty daily candles in, next day's move out. Prices are
rescaled so the first close is 100 and volume is a multiple of its own median,
which removes the level: the strongest cue for identifying which stock and which
year this is. 5,000 stock-days, three arms, 15,000 requests.

This is a cleaner trading test than the news study. A chart is fully known at
the close of day 0 and the trade is day 0 close to day 1 close, so there is no
question about whether the information was available in time.

### It says "I don't know", and it is right to

```
labels      up 5.3%, down 12.3%, flat 82.4%
```

It declines to call direction on more than four charts in five. That is the
correct answer, and a model that guessed a direction every time would be worse
calibrated, not better.

### It is genuinely reading the chart

The battery also asks whether the recent move looks overextended. That answer
correlates **+0.335** with the size of the last move, and only **+0.155** on the
scrambled arm where the daily returns have been shuffled into a random order. So
the model is looking at the data and understands the question. It is not
ignoring the input and defaulting to flat.

### The direction call is still worthless

| arm | seen (2022-23) | recent (2025-26) | pooled | net of cost |
| --- | --- | --- | --- | --- |
| blind | -0.0205 | -0.0250 | -0.0226 | **-33.9 bps** |
| named (ticker + date) | -0.0102 | -0.0243 | -0.0173 | -31.7 bps |
| **scrambled** | +0.0224 | -0.0313 | -0.0063 | -20.1 bps |

Every cell is indistinguishable from zero, and the long-short loses money in all
three. The line that matters is the last one: **shuffling the candles into a
random order changes nothing.** If the shape carried the signal, destroying the
shape would destroy the signal. It does not, so there was no signal in the shape.

No era effect either, which is the same conclusion the news study reached by a
different route: 2022-23 and 2025-26 score the same, so nothing is being
recalled.

### What was even findable

| baseline | seen | recent |
| --- | --- | --- |
| 1-day reversal | +0.0246 | +0.0255 |
| 5-day reversal | +0.0277 | +0.0082 |
| 12-1 momentum | +0.0104 | +0.0321 |

The real, documented price anomalies sit at about **+0.02** on this sample, below
the 0.056 an era of 2,500 can separate from zero. So this study rules out Jev
having a *large* chart-reading edge. It cannot resolve a small one, and neither
can it resolve the known anomalies. Any claim that a model "beats technical
analysis" at this sample size is claiming more than the data supports.

## Reproducing

```bash
export TYPESAFE_API_KEY=...

# study 1, news
uv run --with pandas --with pyarrow python data/prep3.py 5000
go run ./cmd/jev-alpha -run -events data/events3.json    # 15,000 requests, ~5.5 min
go run ./cmd/jev-alpha -events data/events3.json         # re-score without the API

# study 2, candles
uv run --with pandas --with numpy python data/prep_candles.py 2500
go run ./cmd/jev-candles -run                            # 15,000 requests, ~5.5 min
go run ./cmd/jev-candles                                 # re-score without the API

go test ./...
```

`-rejoin` swaps in a corrected return window on saved runs without re-calling
the API. Redefining a window does not change what the model answered, so a fix
costs nothing to apply.

```
cmd/jev-alpha        news study runner and report
cmd/jev-candles      candle study runner and report
internal/study       arms, battery, scoring, bootstrap, portfolio
internal/candles     chart rendering, arms, battery
data/prep3.py        news event construction, the file to argue with
data/prep_candles.py chart event construction
results/             news study, per-event output for all three arms
results-candles/     candle study, same
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
