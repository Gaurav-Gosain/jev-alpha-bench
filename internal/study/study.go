package study

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	jev "github.com/Gaurav-Gosain/jev-go"
)

// Event is one headline with the return windows it can be scored against.
//
// Every return is already excess over QQQ across matching trading day offsets.
// The corpus places ts_0 on the day AFTER publication, which is why Reaction
// and Drift are separate fields rather than one number.
type Event struct {
	Stock string `json:"stock"`
	// WrongStock is another company from the universe, for the arm that hands
	// the model an identity that does not belong to the headline.
	WrongStock string `json:"wrong_stock"`
	EventDate  string `json:"event_date"`
	PubDate    string `json:"pub_date"`
	Title      string `json:"title"`
	AnonTitle  string `json:"anon_title"`
	// Changed records whether anonymisation actually removed anything. Half the
	// headlines never name their subject, and on those the two arms see nearly
	// the same text, so the contrast only means something on this subset.
	Changed bool `json:"anon_changed"`
	// MoveWord marks headlines that simply state the move that already happened
	// ("slumps as ..."). Reading one of those is description, not prediction.
	MoveWord bool `json:"move_word"`

	// PubDay is ts_-2 to ts_-1, the publication day itself. Intraday news gets
	// priced here. Not tradeable: entry would be the morning of, before the
	// headline exists.
	PubDay float64 `json:"pubday"`
	// Reaction is ts_-1 to ts_0: where the news gets priced. NOT tradeable for
	// news released after the close. It is the diagnostic, and the window where
	// a model answering from memory would show it most clearly.
	Reaction float64 `json:"reaction"`
	// Drift1 and Drift5 start from the ts_0 close, a full close after the
	// reaction, so they are genuinely tradeable. This is the trading question.
	Drift1 float64 `json:"drift1"`
	Drift5 float64 `json:"drift5"`
	// Pre and Far are placebos. Pre is ts_-5 to ts_-3, ending two trading days
	// before publication, and Far sits well after the news is stale. Both must
	// score near zero. An earlier version used ts_-3 to ts_-1, which spans the
	// publication day, and duly "predicted" the past.
	Pre float64 `json:"pre"`
	Far float64 `json:"far"`
	// ReactionD and Drift1D are cross-sectionally demeaned within the event
	// date, which removes whatever moved every Nasdaq name that day.
	PubDayD   float64 `json:"pubday_d"`
	ReactionD float64 `json:"reaction_d"`
	Drift1D   float64 `json:"drift1_d"`
}

// LoadEvents reads the prepared event set.
func LoadEvents(path string) ([]Event, error) {
	raw, err := os.ReadFile(path) // #nosec G304 -- operator supplied path
	if err != nil {
		return nil, err
	}
	var events []Event
	if err := json.Unmarshal(raw, &events); err != nil {
		return nil, fmt.Errorf("decoding %s: %w", path, err)
	}
	return events, nil
}

// Arm is one way of presenting an event to the model.
type Arm string

// The two arms. They differ only in what the model is told about which company
// the headline concerns.
const (
	// Named is the headline as published, with the ticker and company name.
	// This is what a naive backtest would do, and it is the arm where the model
	// could answer from memory rather than from the text.
	Named Arm = "named"
	// Blind is the anonymised headline with no ticker and no company name. The
	// semantics survive; the lookup key does not.
	Blind Arm = "blind"
	// Wrong is the anonymised headline handed a different company's identity.
	// If the name carries information rather than just enabling recall, this
	// should land somewhere other than Named.
	Wrong Arm = "wrong"
)

// Arms is all three, in report order.
var Arms = []Arm{Named, Blind, Wrong}

// Question ids.
const (
	QDirection = "direction"
	QMagnitude = "magnitude"
	QPricedIn  = "priced_in"
)

// Battery is the judgment asked of every headline.
//
// No date is supplied in either arm. A date is not needed to judge whether news
// is good or bad, and withholding it removes one more handle for recall. That
// makes the named arm a weaker memorisation test than it could be, which is the
// safe direction: it cannot inflate the headline result.
func Battery() jev.Questions {
	return jev.Questions{
		QDirection: jev.OneOf(
			"Over the next few trading days, which way should this news move the "+
				"company's share price relative to the overall market?",
			map[string]string{
				"bullish": "The stock should outperform the market.",
				"bearish": "The stock should underperform the market.",
				"neutral": "No clear directional effect on the share price either way.",
			},
		),
		QMagnitude: jev.Levels(
			"How large a share price move does this news justify, relative to a "+
				"normal day for this kind of company?",
			"None: routine coverage with no price implication.",
			"Small: a minor move, well within a normal day's range.",
			"Moderate: a clear move, larger than a normal day.",
			"Large: a major repricing of the company.",
		),
		QPricedIn: jev.YesNo(
			"Is this news already reflected in the share price, because it is "+
				"routine coverage, previously reported, or widely expected?",
			"It is routine, already known, or fully anticipated by the market.",
			"It is genuinely new information the market has yet to absorb.",
		),
	}
}

// state builds what the model sees for one event under one arm.
func state(e Event, arm Arm, names map[string]string) any {
	identify := func(ticker, headline string) any {
		company := names[ticker]
		if company == "" {
			company = ticker
		}
		return map[string]string{
			"company":  fmt.Sprintf("%s (%s)", company, ticker),
			"ticker":   ticker,
			"headline": headline,
		}
	}
	switch arm {
	case Blind:
		return map[string]string{"headline": e.AnonTitle}
	case Wrong:
		return identify(e.WrongStock, e.AnonTitle)
	default:
		return identify(e.Stock, e.Title)
	}
}

// Scored is one event after the model has judged it.
type Scored struct {
	Event
	Arm Arm `json:"arm"`

	// PBull, PBear and PNeutral are the direction distribution.
	PBull    float64 `json:"p_bull"`
	PBear    float64 `json:"p_bear"`
	PNeutral float64 `json:"p_neutral"`
	// DirConf is the Choice's own confidence.
	DirConf float64 `json:"dir_confidence"`
	// Magnitude is the Score, 0 to 3.
	Magnitude float64 `json:"magnitude"`
	MagConf   float64 `json:"mag_confidence"`
	// PricedIn is the Noul, 0 to 1.
	PricedIn float64 `json:"priced_in"`

	LatencyMS   int64 `json:"latency_ms"`
	InputTokens int   `json:"input_tokens"`
}

// Tilt is the pre-registered primary signal: how far the direction distribution
// leans bullish rather than bearish.
//
// Deliberately the simplest thing that could work. Every extra way of combining
// the three answers is another chance to find an edge by trying enough of them.
func (s Scored) Tilt() float64 { return s.PBull - s.PBear }

// Conviction is the secondary signal: the same tilt, scaled down when the move
// should be small or when the news is already priced in.
func (s Scored) Conviction() float64 {
	return s.Tilt() * (s.Magnitude / 3) * (1 - s.PricedIn)
}

// RunArm scores every event under one arm.
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
		priced, err := r.Response.Answers.Noul(QPricedIn)
		if err != nil {
			return nil, stats, err
		}
		out = append(out, Scored{
			Event:       r.Item,
			Arm:         arm,
			PBull:       dir.Probabilities["bullish"],
			PBear:       dir.Probabilities["bearish"],
			PNeutral:    dir.Probabilities["neutral"],
			DirConf:     dir.Confidence,
			Magnitude:   mag.Value,
			MagConf:     mag.Confidence,
			PricedIn:    priced,
			LatencyMS:   r.Response.Latency.Milliseconds(),
			InputTokens: r.Response.Usage.InputTokens,
		})
	}
	return out, stats, nil
}

// Run holds everything one arm produced, for saving and re-scoring.
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

// Save writes a run to disk.
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

// LoadNames reads the ticker to company name map.
func LoadNames(path string) (map[string]string, error) {
	raw, err := os.ReadFile(path) // #nosec G304 -- operator supplied path
	if err != nil {
		return nil, err
	}
	var names map[string]string
	return names, json.Unmarshal(raw, &names)
}

// Rejoin replaces the return windows on a saved run with a freshly prepared
// event set, matching on stock and event date.
//
// The model's answers do not change when a return window is redefined, so a
// corrected window costs nothing to apply. Re-running the API would only add
// sampling noise on top of a fix.
func Rejoin(run Run, events []Event) (Run, int, error) {
	byKey := make(map[string]Event, len(events))
	for _, e := range events {
		byKey[e.Stock+"|"+e.EventDate] = e
	}
	matched := 0
	out := make([]Scored, 0, len(run.Scored))
	for _, s := range run.Scored {
		e, ok := byKey[s.Stock+"|"+s.EventDate]
		if !ok {
			continue
		}
		matched++
		s.Event = e
		out = append(out, s)
	}
	run.Scored = out
	return run, matched, nil
}
