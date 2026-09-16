package study_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/Gaurav-Gosain/jev-alpha-bench/internal/study"
)

// A backtest that cannot recover a signal it planted, or that finds one in pure
// noise, is worthless before the model is involved. These pin both ends.

func TestICIsOneOnAPerfectSignal(t *testing.T) {
	t.Parallel()
	signal := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	forward := []float64{10, 20, 30, 40, 50, 60, 70, 80}
	if got := study.SpearmanIC(signal, forward); math.Abs(got-1) > 1e-9 {
		t.Errorf("IC = %v, want 1", got)
	}
}

func TestICIsMinusOneWhenInverted(t *testing.T) {
	t.Parallel()
	signal := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	forward := []float64{80, 70, 60, 50, 40, 30, 20, 10}
	if got := study.SpearmanIC(signal, forward); math.Abs(got+1) > 1e-9 {
		t.Errorf("IC = %v, want -1", got)
	}
}

func TestICHandlesTiesWithoutInventingOrder(t *testing.T) {
	t.Parallel()
	// A signal with no variation cannot rank anything, so the correlation is
	// undefined rather than zero or one.
	flat := []float64{1, 1, 1, 1, 1, 1}
	forward := []float64{5, 3, 9, 1, 7, 2}
	if got := study.SpearmanIC(flat, forward); !math.IsNaN(got) {
		t.Errorf("IC on a flat signal = %v, want NaN", got)
	}
}

// TestICRecoversAPlantedEffect builds a signal with a known, modest correlation
// to the outcome and checks the estimate lands near it. This is the size of
// effect the real experiment is looking for, so if the machinery cannot see it
// here it cannot see it there.
func TestICRecoversAPlantedEffect(t *testing.T) {
	t.Parallel()
	rng := rand.New(rand.NewPCG(1, 2))
	const n = 20000
	signal := make([]float64, n)
	forward := make([]float64, n)
	for i := range n {
		s := rng.NormFloat64()
		signal[i] = s
		// 10% signal, 90% noise.
		forward[i] = 0.10*s + math.Sqrt(1-0.10*0.10)*rng.NormFloat64()
	}
	got := study.SpearmanIC(signal, forward)
	if math.Abs(got-0.10) > 0.02 {
		t.Errorf("IC = %.4f, want about 0.10", got)
	}
}

// TestNullSignalIsIndistinguishableFromZero is the control the real run prints.
// Shuffling the signal must destroy the relationship.
func TestNullSignalIsIndistinguishableFromZero(t *testing.T) {
	t.Parallel()
	rng := rand.New(rand.NewPCG(3, 4))
	const n = 2000
	signal := make([]float64, n)
	forward := make([]float64, n)
	dates := make([]string, n)
	for i := range n {
		signal[i] = rng.NormFloat64()
		forward[i] = 0.2*signal[i] + rng.NormFloat64()
		dates[i] = fmt.Sprintf("2022-01-%02d", i%28+1)
	}

	shuffled := study.Shuffle(signal, 99)
	ci := study.BootstrapByDate(dates, func(idx []int) float64 {
		a := make([]float64, len(idx))
		b := make([]float64, len(idx))
		for k, j := range idx {
			a[k], b[k] = shuffled[j], forward[j]
		}
		return study.SpearmanIC(a, b)
	}, 400, 7)

	if ci.Significant() {
		t.Errorf("a shuffled signal looked significant: IC %+.4f [%+.4f, %+.4f]",
			ci.Point, ci.Lo, ci.Hi)
	}
}

// TestBootstrapWidensWhenEventsShareADate is the reason dates are resampled
// rather than events. Cross-sectional correlation within a day is real, and
// treating each event as independent would understate the error.
func TestBootstrapWidensWhenEventsShareADate(t *testing.T) {
	t.Parallel()
	rng := rand.New(rand.NewPCG(5, 6))
	const n = 1200
	signal := make([]float64, n)
	forward := make([]float64, n)
	clustered := make([]string, n)
	distinct := make([]string, n)

	// A shared per-day shock, which is what a market move is.
	shock := map[int]float64{}
	for i := range n {
		day := i / 40
		if _, ok := shock[day]; !ok {
			shock[day] = rng.NormFloat64()
		}
		signal[i] = rng.NormFloat64()
		forward[i] = shock[day] + 0.3*rng.NormFloat64()
		clustered[i] = fmt.Sprintf("day-%d", day)
		distinct[i] = fmt.Sprintf("row-%d", i)
	}

	stat := func(idx []int) float64 {
		a := make([]float64, len(idx))
		b := make([]float64, len(idx))
		for k, j := range idx {
			a[k], b[k] = signal[j], forward[j]
		}
		return study.SpearmanIC(a, b)
	}

	wide := study.BootstrapByDate(clustered, stat, 400, 11)
	narrow := study.BootstrapByDate(distinct, stat, 400, 11)

	if (wide.Hi - wide.Lo) <= (narrow.Hi - narrow.Lo) {
		t.Errorf("clustering by date gave a %.4f wide interval, not wider than the "+
			"%.4f from treating rows as independent",
			wide.Hi-wide.Lo, narrow.Hi-narrow.Lo)
	}
}

func TestTercileSpreadFollowsTheSignal(t *testing.T) {
	t.Parallel()
	var signal, forward []float64
	for i := range 300 {
		signal = append(signal, float64(i))
		forward = append(forward, float64(i)/10000)
	}
	p := study.TercileSpread(signal, forward, 0)
	if p.Spread <= 0 {
		t.Errorf("spread = %v, want positive when the signal orders the outcome", p.Spread)
	}
	if p.NLong != 100 || p.NShort != 100 {
		t.Errorf("legs = %d/%d, want 100/100", p.NLong, p.NShort)
	}
}

func TestCostsComeOffTheSpread(t *testing.T) {
	t.Parallel()
	var signal, forward []float64
	for i := range 300 {
		signal = append(signal, float64(i))
		forward = append(forward, float64(i)/10000)
	}
	free := study.TercileSpread(signal, forward, 0)
	charged := study.TercileSpread(signal, forward, 10)

	// Ten bps a leg, two legs, so twenty bps off the pair.
	if diff := free.Net - charged.Net; math.Abs(diff-0.0020) > 1e-9 {
		t.Errorf("cost took off %.5f, want 0.00200", diff)
	}
}

func TestDetectableICShrinksWithSampleSize(t *testing.T) {
	t.Parallel()
	small, large := study.DetectableIC(250), study.DetectableIC(4000)
	if !(large < small) {
		t.Errorf("floor did not fall with n: %.4f then %.4f", small, large)
	}
	// At the size this study runs, only fairly large effects are visible, which
	// is the caveat the report has to carry.
	if got := study.DetectableIC(1000); got < 0.06 || got > 0.12 {
		t.Errorf("floor at n=1000 is %.4f, outside the expected 0.06 to 0.12", got)
	}
}
