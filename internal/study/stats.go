// Package study runs the news signal experiment and scores it.
package study

import (
	"math"
	"math/rand/v2"
	"sort"
)

// ranks returns average ranks, so ties do not invent ordering that is not there.
func ranks(xs []float64) []float64 {
	type pair struct {
		v float64
		i int
	}
	order := make([]pair, len(xs))
	for i, v := range xs {
		order[i] = pair{v, i}
	}
	sort.Slice(order, func(a, b int) bool { return order[a].v < order[b].v })

	out := make([]float64, len(xs))
	for start := 0; start < len(order); {
		stop := start
		for stop+1 < len(order) && order[stop+1].v == order[start].v {
			stop++
		}
		shared := float64(start+stop)/2 + 1
		for k := start; k <= stop; k++ {
			out[order[k].i] = shared
		}
		start = stop + 1
	}
	return out
}

// pearson is the correlation of two equal length series.
func pearson(a, b []float64) float64 {
	n := float64(len(a))
	if n < 2 {
		return math.NaN()
	}
	var ma, mb float64
	for i := range a {
		ma += a[i]
		mb += b[i]
	}
	ma, mb = ma/n, mb/n

	var num, da, db float64
	for i := range a {
		x, y := a[i]-ma, b[i]-mb
		num += x * y
		da += x * x
		db += y * y
	}
	if da == 0 || db == 0 {
		return math.NaN()
	}
	return num / math.Sqrt(da*db)
}

// SpearmanIC is the rank correlation between a signal and a forward return.
//
// Rank rather than level, because the return distribution has fat tails and a
// couple of large moves would otherwise decide the number.
func SpearmanIC(signal, forward []float64) float64 {
	if len(signal) != len(forward) || len(signal) < 3 {
		return math.NaN()
	}
	return pearson(ranks(signal), ranks(forward))
}

// Interval is a bootstrap confidence interval around a statistic.
type Interval struct {
	Point float64
	Lo    float64
	Hi    float64
	// PValue is the two sided share of bootstrap draws on the other side of zero.
	PValue float64
}

// Significant reports whether the interval excludes zero.
func (i Interval) Significant() bool { return (i.Lo > 0) == (i.Hi > 0) }

// BootstrapByDate resamples whole dates with replacement and recomputes stat.
//
// Events on the same day share the market's move, so treating each event as an
// independent draw would understate the error badly. Resampling the date keeps
// every event from that day together and prices that correlation in.
func BootstrapByDate(
	dates []string,
	stat func(idx []int) float64,
	draws int,
	seed uint64,
) Interval {
	byDate := map[string][]int{}
	var order []string
	for i, d := range dates {
		if _, seen := byDate[d]; !seen {
			order = append(order, d)
		}
		byDate[d] = append(byDate[d], i)
	}

	all := make([]int, len(dates))
	for i := range all {
		all[i] = i
	}
	point := stat(all)

	rng := rand.New(rand.NewPCG(seed, 0xA1FA))
	samples := make([]float64, 0, draws)
	for range draws {
		var idx []int
		for range order {
			idx = append(idx, byDate[order[rng.IntN(len(order))]]...)
		}
		if v := stat(idx); !math.IsNaN(v) {
			samples = append(samples, v)
		}
	}
	if len(samples) < 10 {
		return Interval{Point: point, Lo: math.NaN(), Hi: math.NaN(), PValue: math.NaN()}
	}
	sort.Float64s(samples)

	var opposite int
	for _, v := range samples {
		if (point >= 0 && v <= 0) || (point < 0 && v >= 0) {
			opposite++
		}
	}
	return Interval{
		Point:  point,
		Lo:     quantile(samples, 0.025),
		Hi:     quantile(samples, 0.975),
		PValue: math.Min(1, 2*float64(opposite)/float64(len(samples))),
	}
}

func quantile(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return math.NaN()
	}
	pos := q * float64(len(sorted)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo == hi {
		return sorted[lo]
	}
	return sorted[lo] + (pos-float64(lo))*(sorted[hi]-sorted[lo])
}

// Portfolio is the result of trading a signal.
type Portfolio struct {
	// Long and Short are the mean excess returns of each leg.
	Long, Short float64
	// Spread is Long minus Short, gross of cost.
	Spread float64
	// Net is Spread after round trip costs on both legs.
	Net float64
	// NLong and NShort are the position counts.
	NLong, NShort int
	// HitRate is the share of positions that moved the way the signal said.
	HitRate float64
}

// TercileSpread goes long the top third of the signal and short the bottom third.
//
// Terciles rather than deciles: at this sample size a decile holds too few
// events for its mean to mean anything.
func TercileSpread(signal, forward []float64, costBps float64) Portfolio {
	n := len(signal)
	if n < 6 {
		return Portfolio{}
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return signal[idx[a]] < signal[idx[b]] })

	cut := n / 3
	var short, long []int
	short = idx[:cut]
	long = idx[n-cut:]

	mean := func(group []int) float64 {
		var s float64
		for _, i := range group {
			s += forward[i]
		}
		return s / float64(len(group))
	}

	var hits int
	for _, i := range long {
		if forward[i] > 0 {
			hits++
		}
	}
	for _, i := range short {
		if forward[i] < 0 {
			hits++
		}
	}

	l, s := mean(long), mean(short)
	spread := l - s
	// Both legs pay a round trip, so the pair pays twice.
	cost := 2 * costBps / 10000
	return Portfolio{
		Long: l, Short: s,
		Spread: spread, Net: spread - cost,
		NLong: len(long), NShort: len(short),
		HitRate: float64(hits) / float64(len(long)+len(short)),
	}
}

// Shuffle returns a permutation of a signal, for the null control.
//
// Breaking the pairing between signal and outcome while keeping both
// distributions intact is what shows whether the pipeline manufactures an edge
// on its own. A null run should centre on zero; if it does not, the machinery
// is wrong before the model is even involved.
func Shuffle(signal []float64, seed uint64) []float64 {
	out := append([]float64(nil), signal...)
	rng := rand.New(rand.NewPCG(seed, 0x5EED))
	rng.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out
}

// Mean of a slice.
func Mean(xs []float64) float64 {
	if len(xs) == 0 {
		return math.NaN()
	}
	var s float64
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

// StdDev is the sample standard deviation.
func StdDev(xs []float64) float64 {
	if len(xs) < 2 {
		return math.NaN()
	}
	m := Mean(xs)
	var s float64
	for _, x := range xs {
		s += (x - m) * (x - m)
	}
	return math.Sqrt(s / float64(len(xs)-1))
}

// DetectableIC is the smallest rank IC a sample of n independent observations
// could separate from zero at 95% confidence with 80% power.
//
// Reported alongside every result so a null reads as "no effect this large"
// rather than the stronger claim that there is no effect at all.
func DetectableIC(n int) float64 {
	if n < 4 {
		return math.NaN()
	}
	return 2.8 / math.Sqrt(float64(n-3))
}
