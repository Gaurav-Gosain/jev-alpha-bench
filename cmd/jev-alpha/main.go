// Command jev-alpha tests whether Jev's judgment of a news headline predicts
// the stock's return after the news is already public.
//
// Two arms over the same events. The named arm gets the ticker and company
// name; the blind arm gets an anonymised headline and nothing else. The gap
// between them is the point: the model was trained on text covering this
// period, so an edge that exists only when the company is named is recall, not
// analysis.
//
//	jev-alpha -run          score both arms and report
//	jev-alpha -report       re-score saved runs without calling the API
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/Gaurav-Gosain/jev-alpha-bench/internal/study"
	jev "github.com/Gaurav-Gosain/jev-go"
)

const seed = 20260916

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "jev-alpha: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	events := flag.String("events", "data/events.json", "prepared event set")
	names := flag.String("names", "data/tickers.json", "ticker to company name map")
	out := flag.String("out", "results", "where to write scored runs")
	doRun := flag.Bool("run", false, "call the API and score every arm")
	rejoin := flag.String("rejoin", "", "replace the return windows on saved runs from this event file, without calling the API")
	limit := flag.Int("limit", 0, "cap the events scored (0 means all)")
	concurrency := flag.Int("concurrency", 10, "requests in flight")
	cost := flag.Float64("cost", 10, "round trip cost per leg, in basis points")
	model := flag.String("model", jev.DefaultModel, "model to evaluate")
	flag.Parse()

	all, err := study.LoadEvents(*events)
	if err != nil {
		return fmt.Errorf("loading events: %w", err)
	}
	if *limit > 0 && *limit < len(all) {
		all = all[:*limit]
	}
	tickers, err := study.LoadNames(*names)
	if err != nil {
		return fmt.Errorf("loading names: %w", err)
	}

	if *doRun {
		if err := scoreArms(all, tickers, *out, *model, *concurrency); err != nil {
			return err
		}
	}

	if *rejoin != "" {
		fresh, err := study.LoadEvents(*rejoin)
		if err != nil {
			return fmt.Errorf("loading %s: %w", *rejoin, err)
		}
		for _, arm := range study.Arms {
			path := filepath.Join(*out, string(arm)+".json")
			run, err := study.LoadRun(path)
			if err != nil {
				return err
			}
			before := len(run.Scored)
			joined, matched, err := study.Rejoin(run, fresh)
			if err != nil {
				return err
			}
			if err := joined.Save(path); err != nil {
				return err
			}
			fmt.Printf("  rejoined %-6s %d of %d events matched\n", arm, matched, before)
		}
	}

	return report(*out, *cost)
}

func scoreArms(events []study.Event, names map[string]string, out, model string, concurrency int) error {
	client, err := jev.New(jev.WithModel(model), jev.WithTimeout(120*time.Second))
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := os.MkdirAll(out, 0o750); err != nil {
		return err
	}

	fmt.Printf("scoring %d events across %d arms\n", len(events), len(study.Arms))
	for _, arm := range study.Arms {
		scored, stats, err := study.RunArm(ctx, client, events, arm, names, concurrency,
			func(done, total int) {
				if done == total || done%50 == 0 {
					fmt.Printf("\r  %-6s %d/%d", arm, done, total)
				}
			})
		if err != nil {
			return fmt.Errorf("arm %s: %w", arm, err)
		}
		fmt.Println()

		modelVersion := ""
		if len(scored) > 0 {
			modelVersion = model
		}
		run := study.Run{
			Arm: arm, Model: modelVersion, RunAt: time.Now().UTC(), Scored: scored,
			Requests: stats.Succeeded, Failed: stats.Failed,
			WallSec: stats.Wall.Seconds(), P50MS: stats.Percentile(0.5).Milliseconds(),
			Tokens: stats.InputTokens,
		}
		path := filepath.Join(out, string(arm)+".json")
		if err := run.Save(path); err != nil {
			return err
		}
		fmt.Printf("  wrote %s: %d scored, %d failed, %.1fs, p50 %dms\n",
			path, len(scored), stats.Failed, stats.Wall.Seconds(), run.P50MS)
	}
	return nil
}

func report(dir string, costBps float64) error {
	runs := map[study.Arm]study.Run{}
	for _, arm := range study.Arms {
		r, err := study.LoadRun(filepath.Join(dir, string(arm)+".json"))
		if err != nil {
			return fmt.Errorf("loading %s: %w (run with -run first)", arm, err)
		}
		runs[arm] = r
	}
	named, blind, wrong := runs[study.Named], runs[study.Blind], runs[study.Wrong]

	fmt.Printf("\n%s\nWHAT THE MODEL ANSWERED\n%s\n", dash(), dash())
	for _, arm := range study.Arms {
		fmt.Printf("\n  %s  (n=%d, p50 %dms)\n", arm, len(runs[arm].Scored), runs[arm].P50MS)
		study.Distribution(os.Stdout, runs[arm].Scored)
	}
	fmt.Printf("\n  named and blind agree on the label for %.1f%% of events, "+
		"signal rank correlation %.3f\n",
		100*study.AgreementRate(named.Scored, blind.Scored),
		study.TiltCorrelation(named.Scored, blind.Scored))
	fmt.Printf("  named and wrong agree on the label for %.1f%% of events, "+
		"signal rank correlation %.3f\n",
		100*study.AgreementRate(named.Scored, wrong.Scored),
		study.TiltCorrelation(named.Scored, wrong.Scored))

	// Placebos first. If these are not flat, nothing after them is worth reading.
	fmt.Printf("\n%s\nPLACEBOS  windows the headline cannot possibly inform.\n"+
		"These must be flat, or the plumbing is wrong.\n%s\n", dash(), dash())
	for _, h := range []study.Horizon{study.HPre, study.HFar} {
		for _, arm := range study.Arms {
			c := study.Score(runs[arm].Scored, study.Signals[0], h, costBps, seed)
			fmt.Printf("  %-6s %-14s IC %+.4f [%+.4f, %+.4f]  %s\n",
				arm, h.Name, c.IC.Point, c.IC.Lo, c.IC.Hi, shortVerdict(c))
		}
	}

	// Where the news actually gets priced. Neither window is tradeable.
	fmt.Printf("\n%s\nWHERE THE NEWS GETS PRICED  neither window is tradeable.\n"+
		"pubday is ts_-2 to ts_-1, which prices intraday news.\n"+
		"reaction is ts_-1 to ts_0, which prices after-close news.\n"+
		"Together these ask whether it reads the news at all.\n%s\n", dash(), dash())
	for _, h := range []study.Horizon{study.HPubDay, study.HReaction} {
		for _, arm := range study.Arms {
			c := study.Score(runs[arm].Scored, study.Signals[0], h, costBps, seed)
			fmt.Printf("  %-6s %-10s IC %+.4f [%+.4f, %+.4f]  net %s  %s\n",
				arm, h.Name, c.IC.Point, c.IC.Lo, c.IC.Hi, bpsOf(c.Port.Net), shortVerdict(c))
		}
	}

	fmt.Printf("\n%s\nDOES IT NEED TO KNOW THE COMPANY?\nreaction window, only headlines that named their subject.\n"+
		"If naming the company is doing the work, named beats blind.\n%s\n", dash(), dash())
	for _, arm := range study.Arms {
		sub := study.Subset(runs[arm].Scored, study.OnlyAnonymised)
		c := study.Score(sub, study.Signals[0], study.HReaction, costBps, seed)
		fmt.Printf("  %-6s IC %+.4f [%+.4f, %+.4f]  n=%d  %s\n",
			arm, c.IC.Point, c.IC.Lo, c.IC.Hi, c.N, shortVerdict(c))
	}

	fmt.Printf("\n%s\nIS IT JUST READING THE MOVE OFF THE HEADLINE?\n"+
		"A headline saying \"slumps as ...\" can be scored with no judgment.\n"+
		"Dropping those is the test of whether anything is left.\n%s\n", dash(), dash())
	for _, h := range []study.Horizon{study.HPubDay, study.HReaction} {
		for _, arm := range study.Arms {
			full := study.Score(runs[arm].Scored, study.Signals[0], h, costBps, seed)
			sub := study.Subset(runs[arm].Scored, study.NoMoveWord)
			quiet := study.Score(sub, study.Signals[0], h, costBps, seed)
			fmt.Printf("  %-6s %-9s all %+.4f  ->  no move word %+.4f [%+.4f, %+.4f]  n=%d  %s\n",
				arm, h.Name, full.IC.Point, quiet.IC.Point, quiet.IC.Lo, quiet.IC.Hi,
				quiet.N, shortVerdict(quiet))
		}
	}

	// The actual trading question.
	fmt.Printf("\n%s\nTRADEABLE  entry at the ts_0 close, a full close after the\n"+
		"reaction. This is the money question.\n%s\n", dash(), dash())
	cards := map[study.Arm]study.Scorecard{}
	for _, arm := range study.Arms {
		c := study.Score(runs[arm].Scored, study.Signals[0], study.HDrift1, costBps, seed)
		cards[arm] = c
		study.WriteScorecard(os.Stdout, c)
	}
	study.WriteComparison(os.Stdout, cards[study.Named], cards[study.Blind])

	fmt.Printf("\n%s\nEVERYTHING ELSE, EXPLORATORY\nEach extra cell is another chance to find a pattern by\nlooking enough times. Read them that way.\n%s\n", dash(), dash())
	for _, sig := range study.Signals {
		for _, h := range []study.Horizon{study.HDrift5, study.HPubDayD, study.HReactionD, study.HDrift1D} {
			for _, arm := range study.Arms {
				c := study.Score(runs[arm].Scored, sig, h, costBps, seed)
				fmt.Printf("  %-6s %-12s %-18s IC %+.4f [%+.4f, %+.4f]  net %s  %s\n",
					arm, sig.Name, h.Name, c.IC.Point, c.IC.Lo, c.IC.Hi,
					bpsOf(c.Port.Net), shortVerdict(c))
			}
		}
	}

	fmt.Printf("\n%s\nBY YEAR  reaction window, primary signal\n%s\n", dash(), dash())
	for _, arm := range study.Arms {
		years := study.ByYear(runs[arm].Scored)
		for _, y := range study.SortedYears(years) {
			if len(years[y]) < 100 {
				continue
			}
			c := study.Score(years[y], study.Signals[0], study.HReaction, costBps, seed)
			fmt.Printf("  %-6s %s  IC %+.4f [%+.4f, %+.4f]  n=%d\n",
				arm, y, c.IC.Point, c.IC.Lo, c.IC.Hi, c.N)
		}
	}

	fmt.Printf("\n%s\ncost assumption: %.0f bps round trip per leg, %.0f bps for the pair\n",
		dash(), costBps, 2*costBps)
	return nil
}

func shortVerdict(c study.Scorecard) string {
	if c.IC.Significant() {
		return "SIGNIFICANT"
	}
	return "flat"
}

func bpsOf(v float64) string { return fmt.Sprintf("%+.1f bps", v*10000) }

func dash() string { return "========================================================================" }
