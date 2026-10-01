package tui

import (
	"fmt"
	"image/color"
	"strings"
	"sync"
	"time"

	"charm.land/lipgloss/v2"
)

const statusHeight = 2

// barColors colours the progress bar by stage. While a run is active the
// gradient flows along the bar with each frame.
type barColors struct {
	mu    sync.Mutex
	steps []color.Color
	shift int
}

const barSteps = 96

func (c *barColors) set(th theme, s stage, frame int) {
	var stops []color.Color
	shift := 0
	switch s {
	case stageDone:
		stops = []color.Color{th.ok, th.accent3, th.ok}
	case stageFailed:
		stops = []color.Color{th.bad, th.accent2, th.bad}
	case stageApproval:
		stops = []color.Color{th.warn, th.accent2, th.warn}
		shift = frame
	case stageCancelled, stageIdle:
		stops = []color.Color{th.faint, th.muted, th.faint}
	default:
		stops = []color.Color{th.accent, th.accent2, th.accent3, th.accent}
		shift = frame * 3
	}
	c.mu.Lock()
	c.steps = lipgloss.Blend1D(barSteps, stops...)
	c.shift = shift
	c.mu.Unlock()
}

func (c *barColors) at(total, current float64) color.Color {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.steps) == 0 {
		return lipgloss.Color("#888888")
	}
	i := 0
	if total > 0 {
		i = int(current / total * float64(barSteps-1))
	}
	// Run the gradient back and forth so the flow has no seam.
	i = (i + c.shift) % (2 * (barSteps - 1))
	if i >= barSteps {
		i = 2*(barSteps-1) - i
	}
	return c.steps[i]
}

var pipeline = []struct {
	stage stage
	name  string
}{
	{stageRequest, "request"},
	{stageThinking, "think"},
	{stageTools, "tools"},
	{stageWriting, "write"},
	{stageDone, "done"},
}

// renderStatus draws the run's lifecycle: a stage pipeline and a progress bar.
func (m *model) renderStatus(w int) string {
	th := m.th
	r := m.run

	var left string
	switch {
	case r.stage == stageIdle:
		left = th.faintText.Render("● ready")
	case r.stage == stageApproval:
		pulse := []string{"◈", "◇"}[m.frame/4%2]
		left = lipgloss.NewStyle().Foreground(th.warn).Bold(true).Render(pulse + " waiting for your approval")
	case m.running:
		left = lipgloss.NewStyle().Foreground(th.accent3).Render(m.spinner()) + " " + m.renderPipeline()
	case r.stage == stageDone:
		left = lipgloss.NewStyle().Foreground(th.ok).Bold(true).Render("✓ answered")
	case r.stage == stageCancelled:
		left = th.dim.Render("■ interrupted")
	case r.stage == stageFailed:
		msg := "✗ failed"
		if r.outcome == "max_turns" {
			msg = "✗ hit the turn limit"
		} else if r.outcome == "refused" {
			msg = "⚑ refused"
		}
		left = lipgloss.NewStyle().Foreground(th.bad).Bold(true).Render(msg)
	}
	if m.flash != "" {
		left += "  " + lipgloss.NewStyle().Foreground(th.warn).Italic(true).Render(m.flash)
	}

	var facts []string
	if r.turn > 0 {
		facts = append(facts, fmt.Sprintf("turn %d/%d", r.turn, m.info.MaxTurns))
	}
	if !r.started.IsZero() {
		d := r.elapsed
		if m.running {
			d = time.Since(r.started)
		}
		facts = append(facts, "⏱ "+fmtDur(d))
	}
	if r.ttft > 0 {
		facts = append(facts, "⚡"+fmtDur(r.ttft))
	}
	if m.usage.In+m.usage.Out > 0 {
		facts = append(facts, "↑"+fmtTokens(m.usage.In)+" ↓"+fmtTokens(m.usage.Out))
	}
	if len(m.queue) > 0 {
		facts = append(facts, fmt.Sprintf("%d queued", len(m.queue)))
	}
	right := th.dim.Render(strings.Join(facts, "  ·  "))

	gap := w - 2 - lipgloss.Width(left) - lipgloss.Width(right)
	line := " " + left + strings.Repeat(" ", max(1, gap)) + right
	return line + "\n " + m.bar.View()
}

func (m *model) renderPipeline() string {
	th := m.th
	cur := m.run.stage
	rank := func(s stage) int {
		for i, p := range pipeline {
			if p.stage == s {
				return i
			}
		}
		return -1
	}
	now := rank(cur)
	parts := make([]string, 0, len(pipeline))
	for i, p := range pipeline {
		name := p.name
		if p.stage == stageTools && m.run.toolsTotal > 0 {
			name = fmt.Sprintf("tools %d/%d", m.run.toolsDone, m.run.toolsTotal)
		}
		switch {
		case i < now:
			parts = append(parts, lipgloss.NewStyle().Foreground(th.accent).Render("● "+name))
		case i == now:
			dot := []string{"◉", "○"}[m.frame/3%2]
			parts = append(parts, gradient(dot+" "+name, true, th.accent2, th.accent3))
		default:
			parts = append(parts, th.faintText.Render("○ "+name))
		}
	}
	sep := lipgloss.NewStyle().Foreground(th.border).Render(" ━ ")
	return strings.Join(parts, sep)
}
