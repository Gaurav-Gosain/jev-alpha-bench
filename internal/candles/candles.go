// Package candles asks whether Jev can read a price chart.
//
// No news, no company, no dates in the blind arm: thirty daily candles rescaled
// so the first close is 100, and the question of what happens tomorrow. The
// trade is day 0 close to day 1 close, so unlike a news headline the
// information is unambiguously available at entry.
package candles

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	jev "github.com/Gaurav-Gosain/jev-go"
)

// Window is thirty days of OHLCV, rescaled.
type Window struct {
	Open   []float64 `json:"o"`
	High   []float64 `json:"h"`
	Low    []float64 `json:"l"`
	Close  []float64 `json:"c"`
	Volume []float64 `json:"v"`
}

// Table renders the window the way a chart would be read, one row per day.
//
// Day 0 is the most recent close, which is where the decision is made. Prices
// are already rescaled so the first close is 100 and volume is a multiple of
// its own median, which removes the price level: the single strongest cue for
// identifying which stock and which year this is.
func (w Window) Table() string {
	var b strings.Builder
	b.WriteString("day   open   high    low  close    vol\n")
	n := len(w.Close)
	for i := range n {
		fmt.Fprintf(&b, "%3d %6.2f %6.2f %6.2f %6.2f %6.2f\n",
			i-(n-1), w.Open[i], w.High[i], w.Low[i], w.Close[i], w.Volume[i])
	}
	return b.String()
}

// Event is one stock-day: the chart, the answer, and the mechanical baselines
// that any useful signal has to beat.
type Event struct {
	Stock string `json:"stock"`
	// Era is "seen" for 2022-23, which is certainly in the model's training
	// data, and "recent" for 2025-26, which is at or past its cutoff.
	Era   string `json:"era"`
	Date  string `json:"date"`
	Chart Window `json:"candles"`
	// Scrambled holds the same daily returns in a shuffled order. Volatility
	// survives, every temporal pattern does not.
	Scrambled Window `json:"scrambled"`

	// Fwd is the next day's raw return, Excess is that less QQQ.
	Fwd    float64 `json:"fwd"`
	Excess float64 `json:"excess"`

	// The baselines, all computed in code from the same window.
	Rev1  float64 `json:"rev1"`
	Rev5  float64 `json:"rev5"`
	Mom   float64 `json:"mom"`
	Vol20 float64 `json:"vol20"`
}

// LoadEvents reads the prepared chart set.
func LoadEvents(path string) ([]Event, error) {
	raw, err := os.ReadFile(path) // #nosec G304 -- operator supplied path
	if err != nil {
		return nil, err
	}
	var events []Event
	return events, json.Unmarshal(raw, &events)
}

// Arm is one way of showing the chart.
type Arm string

// The arms.
const (
	// Blind is the rescaled chart with no identity and no dates. This is the
	// real test: nothing here but the shape.
	Blind Arm = "blind"
	// Named adds the ticker, the company and the actual date, which is every
	// handle a model would need to look the answer up instead of reading it.
	Named Arm = "named"
	// Scrambled is the same daily returns in a shuffled order. If this scores
	// like Blind, the model is reacting to how choppy the window looks rather
	// than to its shape.
	Scrambled Arm = "scrambled"
)

// Arms is all three, in report order.
var Arms = []Arm{Blind, Named, Scrambled}

// Question ids.
const (
	QDirection = "direction"
	QMagnitude = "magnitude"
	QStretched = "stretched"
)

// Battery is the judgment asked of every chart.
//
// The third question names mean reversion directly. Short term reversal is the
// one price pattern with real evidence behind it, so asking about it outright
// gives the model its best shot rather than hoping it volunteers the idea.
func Battery() jev.Questions {
	return jev.Questions{
		QDirection: jev.OneOf(
			"Based only on this price history, which way is this instrument more "+
				"likely to move on the next trading day, relative to the overall market?",
			map[string]string{
				"up":   "It should outperform the market tomorrow.",
				"down": "It should underperform the market tomorrow.",
				"flat": "No edge either way; tomorrow is a coin flip.",
			},
		),
		QMagnitude: jev.Levels(
			"How large a move does this chart suggest for the next trading day, "+
				"compared with this instrument's own recent daily range?",
			"Quiet: a smaller move than its recent typical day.",
			"Normal: about its recent typical day.",
			"Active: a larger move than usual.",
			"Violent: a far larger move than usual.",
		),
		QStretched: jev.YesNo(
			"Has the recent move gone far enough that a snap back in the other "+
				"direction is more likely than a continuation?",
			"The move looks overextended and due to reverse.",
			"The move looks sustainable, or there is no clear move to reverse.",
		),
	}
}

const blindNote = "Prices are rescaled so the first close is 100. Volume is a multiple " +
	"of this window's median volume. Day 0 is the most recent close."

// state builds what the model sees for one event under one arm.
func state(e Event, arm Arm, names map[string]string) any {
	switch arm {
	case Named:
		company := names[e.Stock]
		if company == "" {
			company = e.Stock
		}
		return map[string]string{
			"instrument": fmt.Sprintf("%s (%s)", company, e.Stock),
			"as_of":      e.Date,
			"note":       blindNote,
			"candles":    e.Chart.Table(),
		}
	case Scrambled:
		return map[string]string{
			"instrument": "a large US listed technology stock",
			"note":       blindNote,
			"candles":    e.Scrambled.Table(),
		}
	default:
		return map[string]string{
			"instrument": "a large US listed technology stock",
			"note":       blindNote,
			"candles":    e.Chart.Table(),
		}
	}
}

// Scored is one chart after the model has judged it.
type Scored struct {
	Event
	Arm Arm `json:"arm"`

	PUp   float64 `json:"p_up"`
	PDown float64 `json:"p_down"`
	PFlat float64 `json:"p_flat"`

	DirConf   float64 `json:"dir_confidence"`
	Magnitude float64 `json:"magnitude"`
	Stretched float64 `json:"stretched"`

	LatencyMS   int64 `json:"latency_ms"`
	InputTokens int   `json:"input_tokens"`
}

// Tilt is the pre-registered signal: how far the model leans up rather than down.
func (s Scored) Tilt() float64 { return s.PUp - s.PDown }

// Run holds one arm's output.
type Run struct {
	Arm      Arm       `json:"arm"`
	Model    string    `json:"model"`
	RunAt    time.Time `json:"run_at"`
	Scored   []Scored  `json:"scored"`
	Requests int       `json:"requests"`
	Failed   int       `json:"failed"`
	WallSec  float64   `json:"wall_seconds"`
	P50MS    int64     `json:"p50_ms"`
	Tokens   int       `json:"input_tokens"`
}

// Save writes a run.
func (r Run) Save(path string) error {
	raw, err := json.MarshalIndent(r, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, raw, 0o600)
}

// LoadRun reads a saved run.
func LoadRun(path string) (Run, error) {
	var run Run
	raw, err := os.ReadFile(path) // #nosec G304 -- operator supplied path
	if err != nil {
		return run, err
	}
	return run, json.Unmarshal(raw, &run)
}

// RunArm scores every chart under one arm.
func RunArm(
	ctx context.Context,
	client *jev.Client,
	events []Event,
	arm Arm,
	names map[string]string,
	concurrency int,
	progress func(done, total int),
) ([]Scored, jev.BatchStats, error) {
	battery := Battery()

	results, stats := jev.Batch(ctx, client, events,
		func(e Event) (any, jev.Questions) { return state(e, arm, names), battery },
		jev.BatchOptions{Concurrency: concurrency, OnProgress: progress},
	)

	out := make([]Scored, 0, len(events))
	for _, r := range results {
		if !r.OK() {
			continue
		}
		dir, err := r.Response.Answers.Choice(QDirection)
		if err != nil {
			return nil, stats, err
		}
		mag, err := r.Response.Answers.Score(QMagnitude)
		if err != nil {
			return nil, stats, err
		}
		stretched, err := r.Response.Answers.Noul(QStretched)
		if err != nil {
			return nil, stats, err
		}
		out = append(out, Scored{
			Event: r.Item, Arm: arm,
			PUp: dir.Probabilities["up"], PDown: dir.Probabilities["down"],
			PFlat: dir.Probabilities["flat"], DirConf: dir.Confidence,
			Magnitude: mag.Value, Stretched: stretched,
			LatencyMS:   r.Response.Latency.Milliseconds(),
			InputTokens: r.Response.Usage.InputTokens,
		})
	}
	return out, stats, nil
}
