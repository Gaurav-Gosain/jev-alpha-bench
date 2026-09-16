// Command jev-candles asks whether Jev can read a price chart with no news.
//
// Thirty daily candles in, next day's move out. The trade is day 0 close to
// day 1 close, so the information is unambiguously available at entry, which
// makes this a cleaner trading test than the news study.
//
// Everything is scored against the mechanical baselines on the same sample.
// Beating zero is not interesting if a one line reversal rule beats you.
package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"syscall"
	"time"

	"github.com/Gaurav-Gosain/jev-alpha-bench/internal/candles"
	"github.com/Gaurav-Gosain/jev-alpha-bench/internal/study"
	jev "github.com/Gaurav-Gosain/jev-go"
)

const (
	seed  = 20260916
	draws = 4000
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "jev-candles: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	path := flag.String("events", "data/candles.json", "prepared chart set")
	namesPath := flag.String("names", "data/tickers.json", "ticker to company name map")
	out := flag.String("out", "results-candles", "where to write scored runs")
	doRun := flag.Bool("run", false, "call the API and score every arm")
	limit := flag.Int("limit", 0, "cap the charts scored (0 means all)")
	concurrency := flag.Int("concurrency", 16, "requests in flight")
	cost := flag.Float64("cost", 10, "round trip cost per leg, in basis points")
	model := flag.String("model", jev.DefaultModel, "model to evaluate")
	flag.Parse()

	events, err := candles.LoadEvents(*path)
	if err != nil {
		return fmt.Errorf("loading charts: %w", err)
	}
	if *limit > 0 && *limit < len(events) {
		events = events[:*limit]
	}
	names, err := study.LoadNames(*namesPath)
	if err != nil {
		return err
	}

	if *doRun {
		client, err := jev.New(jev.WithModel(*model), jev.WithTimeout(120*time.Second))
		if err != nil {
			return err
		}
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer stop()
		if err := os.MkdirAll(*out, 0o750); err != nil {
			return err
		}

		fmt.Printf("scoring %d charts across %d arms\n", len(events), len(candles.Arms))
		for _, arm := range candles.Arms {
			scored, stats, err := candles.RunArm(ctx, client, events, arm, names, *concurrency,
				func(done, total int) {
					if done == total || done%100 == 0 {
						fmt.Printf("\r  %-10s %d/%d", arm, done, total)
					}
				})
			if err != nil {
				return fmt.Errorf("arm %s: %w", arm, err)
			}
			fmt.Println()
			run := candles.Run{
				Arm: arm, Model: *model, RunAt: time.Now().UTC(), Scored: scored,
				Requests: stats.Succeeded, Failed: stats.Failed,
				WallSec: stats.Wall.Seconds(), P50MS: stats.Percentile(0.5).Milliseconds(),
				Tokens: stats.InputTokens,
			}
			if err := run.Save(filepath.Join(*out, string(arm)+".json")); err != nil {
				return err
			}
			fmt.Printf("  %d scored, %d failed, %.1fs, p50 %dms, %d input tokens\n",
				len(scored), stats.Failed, stats.Wall.Seconds(), run.P50MS, stats.InputTokens)
		}
	}

	return report(events, *out, *cost)
}

// series pulls a signal and the outcome out of a slice of events.
func ic(signal, forward []float64, dates []string) study.Interval {
	return study.BootstrapByDate(dates, func(idx []int) float64 {
		a := make([]float64, len(idx))
		b := make([]float64, len(idx))
		for k, j := range idx {
			a[k], b[k] = signal[j], forward[j]
		}
		return study.SpearmanIC(a, b)
	}, draws, seed)
}

func verdict(i study.Interval) string {
	if math.IsNaN(i.Lo) {
		return ""
	}
	if i.Significant() {
		return "SIGNIFICANT"
	}
	return "flat"
}

func report(events []candles.Event, dir string, costBps float64) error {

	subset := func(es []candles.Event, era string) []candles.Event {
		if era == "all" {
			return es
		}
		var out []candles.Event
		for _, e := range es {
			if e.Era == era {
				out = append(out, e)
			}
		}
		return out
	}

	fmt.Printf("\n%s\nWHAT IS EVEN FINDABLE HERE\nMechanical baselines on the same sample. These are the known\n"+
		"price anomalies, and they are the bar, not zero.\n%s\n", dash(), dash())
	fmt.Printf("  %-22s %22s %22s\n", "", "seen (2022-23)", "recent (2025-26)")
	for _, b := range []struct {
		name string
		of   func(candles.Event) float64
	}{
		{"1-day reversal", func(e candles.Event) float64 { return e.Rev1 }},
		{"5-day reversal", func(e candles.Event) float64 { return e.Rev5 }},
		{"12-1 momentum", func(e candles.Event) float64 { return e.Mom }},
		{"20-day volatility", func(e candles.Event) float64 { return e.Vol20 }},
	} {
		line := fmt.Sprintf("  %-22s", b.name)
		for _, era := range []string{"seen", "recent"} {
			es := subset(events, era)
			sig, fwd, dates := columns(es, b.of)
			v := ic(sig, fwd, dates)
			line += fmt.Sprintf(" %+7.4f [%+.3f,%+.3f]", v.Point, v.Lo, v.Hi)
		}
		fmt.Println(line)
	}
	n := len(subset(events, "seen"))
	fmt.Printf("\n  smallest IC separable from zero: %.4f per era (n=%d), %.4f pooled (n=%d)\n",
		study.DetectableIC(n), n, study.DetectableIC(len(events)), len(events))

	runs := map[candles.Arm]candles.Run{}
	for _, arm := range candles.Arms {
		r, err := candles.LoadRun(filepath.Join(dir, string(arm)+".json"))
		if err != nil {
			return fmt.Errorf("loading %s: %w (run with -run first)", arm, err)
		}
		runs[arm] = r
	}

	fmt.Printf("\n%s\nWHAT THE MODEL ANSWERED\n%s\n", dash(), dash())
	for _, arm := range candles.Arms {
		s := runs[arm].Scored
		var up, down, flat int
		tilt := make([]float64, len(s))
		stretched := make([]float64, len(s))
		absMove := make([]float64, len(s))
		for i, x := range s {
			switch {
			case x.PUp > x.PDown && x.PUp > x.PFlat:
				up++
			case x.PDown > x.PUp && x.PDown > x.PFlat:
				down++
			default:
				flat++
			}
			tilt[i], stretched[i] = x.Tilt(), x.Stretched
			absMove[i] = math.Abs(x.Rev1)
		}
		t := float64(len(s))
		fmt.Printf("\n  %-10s n=%d  p50 %dms  %d input tokens\n",
			arm, len(s), runs[arm].P50MS, runs[arm].Tokens)
		fmt.Printf("    labels      up %.1f%%, down %.1f%%, flat %.1f%%\n",
			100*float64(up)/t, 100*float64(down)/t, 100*float64(flat)/t)
		fmt.Printf("    tilt        mean %+.3f, sd %.3f\n", study.Mean(tilt), study.StdDev(tilt))
		fmt.Printf("    stretched   mean %.2f, sd %.2f\n", study.Mean(stretched), study.StdDev(stretched))
		// Comprehension check: does it call a chart stretched after a big move?
		fmt.Printf("    stretched vs size of the last move: rank corr %+.3f\n",
			study.SpearmanIC(stretched, absMove))
	}

	fmt.Printf("\n%s\nCAN IT READ THE CHART?\nrank IC of its up-minus-down tilt against the next day's\n"+
		"excess return. Entry at the day 0 close, so this is tradeable.\n%s\n", dash(), dash())
	fmt.Printf("  %-12s %24s %24s\n", "", "seen (2022-23)", "recent (2025-26)")
	for _, arm := range candles.Arms {
		line := fmt.Sprintf("  %-12s", arm)
		for _, era := range []string{"seen", "recent"} {
			s := subsetScored(runs[arm].Scored, era)
			sig, fwd, dates := columnsScored(s, candles.Scored.Tilt)
			v := ic(sig, fwd, dates)
			line += fmt.Sprintf(" %+7.4f [%+.3f,%+.3f] %-5s", v.Point, v.Lo, v.Hi,
				shortFlag(v))
		}
		fmt.Println(line)
	}

	fmt.Printf("\n%s\nPOOLED, AND WHAT IT WOULD HAVE PAID\n%s\n", dash(), dash())
	for _, arm := range candles.Arms {
		s := runs[arm].Scored
		sig, fwd, dates := columnsScored(s, candles.Scored.Tilt)
		v := ic(sig, fwd, dates)
		p := study.TercileSpread(sig, fwd, costBps)
		net := study.BootstrapByDate(dates, func(idx []int) float64 {
			a := make([]float64, len(idx))
			b := make([]float64, len(idx))
			for k, j := range idx {
				a[k], b[k] = sig[j], fwd[j]
			}
			return study.TercileSpread(a, b, costBps).Net
		}, draws, seed+2)
		fmt.Printf("\n  %s  n=%d\n", arm, len(s))
		fmt.Printf("    rank IC     %+.4f [%+.4f, %+.4f]  %s\n", v.Point, v.Lo, v.Hi, verdict(v))
		fmt.Printf("    long-short  %+.1f bps gross, %+.1f bps net  [%+.1f, %+.1f]  %s\n",
			p.Spread*10000, p.Net*10000, net.Lo*10000, net.Hi*10000, verdict(net))
		fmt.Printf("    hit rate    %.1f%%\n", 100*p.HitRate)
	}

	fmt.Printf("\n%s\ncost assumption: %.0f bps round trip per leg, %.0f bps for the pair\n",
		dash(), costBps, 2*costBps)
	return nil
}

func columns(es []candles.Event, of func(candles.Event) float64) ([]float64, []float64, []string) {
	sig := make([]float64, len(es))
	fwd := make([]float64, len(es))
	dates := make([]string, len(es))
	for i, e := range es {
		sig[i], fwd[i], dates[i] = of(e), e.Excess, e.Date
	}
	return sig, fwd, dates
}

func columnsScored(s []candles.Scored, of func(candles.Scored) float64) ([]float64, []float64, []string) {
	sig := make([]float64, len(s))
	fwd := make([]float64, len(s))
	dates := make([]string, len(s))
	for i, x := range s {
		sig[i], fwd[i], dates[i] = of(x), x.Excess, x.Date
	}
	return sig, fwd, dates
}

func subsetScored(s []candles.Scored, era string) []candles.Scored {
	var out []candles.Scored
	for _, x := range s {
		if x.Era == era {
			out = append(out, x)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Date < out[j].Date })
	return out
}

func shortFlag(i study.Interval) string {
	if i.Significant() {
		return "SIG"
	}
	return "flat"
}

func dash() string { return "========================================================================" }
