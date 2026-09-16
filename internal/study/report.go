package study

import (
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
)

// Horizon is a forward return window.
type Horizon struct {
	Name string
	Pick func(Scored) float64
}

// The windows, all excess over QQQ.
var (
	// HPubDay is the publication day itself, where intraday news gets priced.
	HPubDay  = Horizon{"pubday", func(s Scored) float64 { return s.PubDay }}
	HPubDayD = Horizon{"pubday/demeaned", func(s Scored) float64 { return s.PubDayD }}
	// HReaction is where the news gets priced, and is not tradeable for news
	// released after the close. It is the diagnostic and the memorisation probe.
	HReaction = Horizon{"reaction", func(s Scored) float64 { return s.Reaction }}
	// HDrift1 is the tradeable window and the primary trading metric.
	HDrift1 = Horizon{"drift1", func(s Scored) float64 { return s.Drift1 }}
	HDrift5 = Horizon{"drift5", func(s Scored) float64 { return s.Drift5 }}
	// Demeaned variants, with the day's common move taken out.
	HReactionD = Horizon{"reaction/demeaned", func(s Scored) float64 { return s.ReactionD }}
	HDrift1D   = Horizon{"drift1/demeaned", func(s Scored) float64 { return s.Drift1D }}
	// Placebos. Both must land near zero or the plumbing is wrong.
	HPre = Horizon{"placebo/pre", func(s Scored) float64 { return s.Pre }}
	HFar = Horizon{"placebo/far", func(s Scored) float64 { return s.Far }}
)

// Horizons is every window, in report order.
var Horizons = []Horizon{HPubDay, HReaction, HDrift1, HDrift5, HPubDayD, HReactionD, HDrift1D, HPre, HFar}

// Subset keeps the scored events matching a predicate.
func Subset(scored []Scored, keep func(Scored) bool) []Scored {
	var out []Scored
	for _, s := range scored {
		if keep(s) {
			out = append(out, s)
		}
	}
	return out
}

// OnlyAnonymised keeps events where anonymisation actually removed an identity.
// On the rest the two arms see nearly the same text, so they cannot speak to
// whether the identity mattered.
func OnlyAnonymised(s Scored) bool { return s.Changed }

// NoMoveWord keeps headlines that do not state the price move themselves. A
// headline reading "slumps as ..." can be scored correctly without any
// judgment, so the signal has to survive dropping them.
func NoMoveWord(s Scored) bool { return !s.MoveWord }

// SignalDef is one way of turning answers into a number to trade.
type SignalDef struct {
	Name string
	Of   func(Scored) float64
}

// Signals are the pre-registered primary and the one secondary variant.
var Signals = []SignalDef{
	{"tilt", Scored.Tilt},
	{"conviction", Scored.Conviction},
}

const bootstrapDraws = 4000

// Scorecard is everything measured for one arm, signal and horizon.
type Scorecard struct {
	Arm      Arm
	Signal   string
	Horizon  string
	N        int
	IC       Interval
	NullIC   Interval
	Port     Portfolio
	PortNet  Interval
	Floor    float64
	MeanFwd  float64
	SDFwd    float64
	SpreadSD float64
}

// Score computes the scorecard for one combination.
func Score(scored []Scored, sig SignalDef, h Horizon, costBps float64, seed uint64) Scorecard {
	signal := make([]float64, len(scored))
	forward := make([]float64, len(scored))
	dates := make([]string, len(scored))
	for i, s := range scored {
		signal[i] = sig.Of(s)
		forward[i] = h.Pick(s)
		dates[i] = s.EventDate
	}

	pick := func(idx []int, src []float64) []float64 {
		out := make([]float64, len(idx))
		for i, j := range idx {
			out[i] = src[j]
		}
		return out
	}

	ic := BootstrapByDate(dates, func(idx []int) float64 {
		return SpearmanIC(pick(idx, signal), pick(idx, forward))
	}, bootstrapDraws, seed)

	// The null breaks the pairing between signal and outcome while keeping both
	// distributions. Anything other than zero here is a bug in the machinery.
	shuffled := Shuffle(signal, seed)
	null := BootstrapByDate(dates, func(idx []int) float64 {
		return SpearmanIC(pick(idx, shuffled), pick(idx, forward))
	}, bootstrapDraws, seed+1)

	port := TercileSpread(signal, forward, costBps)
	netCI := BootstrapByDate(dates, func(idx []int) float64 {
		return TercileSpread(pick(idx, signal), pick(idx, forward), costBps).Net
	}, bootstrapDraws, seed+2)

	return Scorecard{
		Arm: scored[0].Arm, Signal: sig.Name, Horizon: h.Name, N: len(scored),
		IC: ic, NullIC: null, Port: port, PortNet: netCI,
		Floor:   DetectableIC(len(scored)),
		MeanFwd: Mean(forward), SDFwd: StdDev(forward),
	}
}

func bps(v float64) string { return fmt.Sprintf("%+.1f bps", v*10000) }

// WriteScorecard prints one combination.
func WriteScorecard(w io.Writer, s Scorecard) {
	fmt.Fprintf(w, "\n  %s / %s / %s   n=%d\n",
		strings.ToUpper(string(s.Arm)), s.Signal, s.Horizon, s.N)
	fmt.Fprintf(w, "    rank IC        %+.4f  [%+.4f, %+.4f]  p=%.3f %s\n",
		s.IC.Point, s.IC.Lo, s.IC.Hi, s.IC.PValue, verdict(s.IC))
	fmt.Fprintf(w, "    null IC        %+.4f  [%+.4f, %+.4f]  (shuffled signal)\n",
		s.NullIC.Point, s.NullIC.Lo, s.NullIC.Hi)
	fmt.Fprintf(w, "    detectable     %.4f   smallest IC this n could separate from zero\n", s.Floor)
	fmt.Fprintf(w, "    long-short     %s gross, %s net of cost\n",
		bps(s.Port.Spread), bps(s.Port.Net))
	fmt.Fprintf(w, "                   [%s, %s] net, p=%.3f %s\n",
		bps(s.PortNet.Lo), bps(s.PortNet.Hi), s.PortNet.PValue, verdict(s.PortNet))
	fmt.Fprintf(w, "    legs           long %s (n=%d), short %s (n=%d), hit rate %.1f%%\n",
		bps(s.Port.Long), s.Port.NLong, bps(s.Port.Short), s.Port.NShort, 100*s.Port.HitRate)
}

func verdict(i Interval) string {
	if math.IsNaN(i.Lo) {
		return ""
	}
	if i.Significant() {
		return "SIGNIFICANT"
	}
	return "not distinguishable from zero"
}

// WriteComparison puts the two arms side by side on the primary metric, which
// is the actual question: does the edge survive losing the company's identity?
func WriteComparison(w io.Writer, named, blind Scorecard) {
	fmt.Fprintf(w, "\n%s\n", strings.Repeat("=", 72))
	fmt.Fprintf(w, "NAMED vs BLIND   %s / %s\n", named.Signal, named.Horizon)
	fmt.Fprintf(w, "%s\n", strings.Repeat("=", 72))
	fmt.Fprintf(w, "  %-14s %14s %14s\n", "", "named", "blind")
	fmt.Fprintf(w, "  %-14s %+14.4f %+14.4f\n", "rank IC", named.IC.Point, blind.IC.Point)
	fmt.Fprintf(w, "  %-14s %14s %14s\n", "95% low",
		fmt.Sprintf("%+.4f", named.IC.Lo), fmt.Sprintf("%+.4f", blind.IC.Lo))
	fmt.Fprintf(w, "  %-14s %14s %14s\n", "95% high",
		fmt.Sprintf("%+.4f", named.IC.Hi), fmt.Sprintf("%+.4f", blind.IC.Hi))
	fmt.Fprintf(w, "  %-14s %14s %14s\n", "long-short net", bps(named.Port.Net), bps(blind.Port.Net))
	fmt.Fprintf(w, "\n  difference in IC: %+.4f\n", named.IC.Point-blind.IC.Point)
}

// Distribution summarises what the model actually answered, which is worth
// seeing before any return is involved. A signal that never varies cannot
// predict anything, and a signal that is 90 percent one label is close to that.
func Distribution(w io.Writer, scored []Scored) {
	var bull, bear, neutral int
	tilts := make([]float64, len(scored))
	mags := make([]float64, len(scored))
	priced := make([]float64, len(scored))
	for i, s := range scored {
		switch {
		case s.PBull > s.PBear && s.PBull > s.PNeutral:
			bull++
		case s.PBear > s.PBull && s.PBear > s.PNeutral:
			bear++
		default:
			neutral++
		}
		tilts[i], mags[i], priced[i] = s.Tilt(), s.Magnitude, s.PricedIn
	}
	n := float64(len(scored))
	fmt.Fprintf(w, "    labels         bullish %.1f%%, bearish %.1f%%, neutral %.1f%%\n",
		100*float64(bull)/n, 100*float64(bear)/n, 100*float64(neutral)/n)
	fmt.Fprintf(w, "    tilt           mean %+.3f, sd %.3f\n", Mean(tilts), StdDev(tilts))
	fmt.Fprintf(w, "    magnitude      mean %.2f / 3, sd %.2f\n", Mean(mags), StdDev(mags))
	fmt.Fprintf(w, "    priced in      mean %.2f, sd %.2f\n", Mean(priced), StdDev(priced))
}

// AgreementRate is the share of events where the two arms picked the same label.
// It says how much of the judgment actually depended on knowing the company.
func AgreementRate(named, blind []Scored) float64 {
	key := func(s Scored) string { return s.Stock + "|" + s.EventDate }
	byKey := map[string]Scored{}
	for _, s := range blind {
		byKey[key(s)] = s
	}
	var matched, agreed int
	for _, a := range named {
		b, ok := byKey[key(a)]
		if !ok {
			continue
		}
		matched++
		if label(a) == label(b) {
			agreed++
		}
	}
	if matched == 0 {
		return math.NaN()
	}
	return float64(agreed) / float64(matched)
}

func label(s Scored) string {
	switch {
	case s.PBull > s.PBear && s.PBull > s.PNeutral:
		return "bullish"
	case s.PBear > s.PBull && s.PBear > s.PNeutral:
		return "bearish"
	default:
		return "neutral"
	}
}

// TiltCorrelation is the rank correlation between the two arms' signals, a
// finer measure of how much the identity changed the answer than the label
// agreement rate.
func TiltCorrelation(named, blind []Scored) float64 {
	key := func(s Scored) string { return s.Stock + "|" + s.EventDate }
	byKey := map[string]Scored{}
	for _, s := range blind {
		byKey[key(s)] = s
	}
	var a, b []float64
	for _, n := range named {
		if m, ok := byKey[key(n)]; ok {
			a = append(a, n.Tilt())
			b = append(b, m.Tilt())
		}
	}
	return SpearmanIC(a, b)
}

// ByYear splits a run by calendar year, so a result can be checked for coming
// from one regime rather than holding across the sample.
func ByYear(scored []Scored) map[string][]Scored {
	out := map[string][]Scored{}
	for _, s := range scored {
		if len(s.EventDate) >= 4 {
			out[s.EventDate[:4]] = append(out[s.EventDate[:4]], s)
		}
	}
	return out
}

// SortedYears returns the years present, in order.
func SortedYears(m map[string][]Scored) []string {
	years := make([]string, 0, len(m))
	for y := range m {
		years = append(years, y)
	}
	sort.Strings(years)
	return years
}
