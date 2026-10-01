package tui

import (
	"image/color"

	"charm.land/lipgloss/v2"
)

// theme holds the palette and the styles built from it.
type theme struct {
	dark bool

	bg, panel, border, borderFocus color.Color
	text, muted, faint             color.Color
	accent, accent2, accent3       color.Color
	ok, warn, bad, info            color.Color

	title, label, dim, faintText lipgloss.Style
	pill, pillAccent             lipgloss.Style
	userBox, userLabel           lipgloss.Style
	agentLabel, reasoning        lipgloss.Style
	card, cardTitle, cardBody    lipgloss.Style
	panelBox, panelFocus         lipgloss.Style
	key, keyHelp                 lipgloss.Style
	modal                        lipgloss.Style
}

func newTheme(dark bool) theme {
	ld := lipgloss.LightDark(dark)
	t := theme{
		dark:        dark,
		bg:          ld(lipgloss.Color("#FAFAFC"), lipgloss.Color("#0E0E14")),
		panel:       ld(lipgloss.Color("#F0EFF7"), lipgloss.Color("#16161F")),
		border:      ld(lipgloss.Color("#D5D3E3"), lipgloss.Color("#2B2A3A")),
		borderFocus: ld(lipgloss.Color("#7C5CFF"), lipgloss.Color("#8B6CFF")),
		text:        ld(lipgloss.Color("#1C1B29"), lipgloss.Color("#E6E4F2")),
		muted:       ld(lipgloss.Color("#6B6880"), lipgloss.Color("#9A97B0")),
		faint:       ld(lipgloss.Color("#A9A6BC"), lipgloss.Color("#55536A")),
		accent:      ld(lipgloss.Color("#6D4AFF"), lipgloss.Color("#9D7BFF")),
		accent2:     ld(lipgloss.Color("#E0467C"), lipgloss.Color("#FF6AA2")),
		accent3:     ld(lipgloss.Color("#0891B2"), lipgloss.Color("#3FD7F5")),
		ok:          ld(lipgloss.Color("#119A5B"), lipgloss.Color("#4ADE9A")),
		warn:        ld(lipgloss.Color("#C27803"), lipgloss.Color("#FFC24B")),
		bad:         ld(lipgloss.Color("#D1334B"), lipgloss.Color("#FF5C73")),
		info:        ld(lipgloss.Color("#2563EB"), lipgloss.Color("#6EA8FF")),
	}

	t.title = lipgloss.NewStyle().Foreground(t.text).Bold(true)
	t.label = lipgloss.NewStyle().Foreground(t.muted).Bold(true)
	t.dim = lipgloss.NewStyle().Foreground(t.muted)
	t.faintText = lipgloss.NewStyle().Foreground(t.faint)
	t.pill = lipgloss.NewStyle().Foreground(t.text).Background(t.border).Padding(0, 1)
	t.pillAccent = lipgloss.NewStyle().Foreground(lipgloss.Color("#FFFFFF")).Background(t.accent).Bold(true).Padding(0, 1)

	t.userLabel = lipgloss.NewStyle().Foreground(t.accent2).Bold(true)
	t.userBox = lipgloss.NewStyle().
		Border(lipgloss.ThickBorder(), false, false, false, true).
		BorderForeground(t.accent2).
		Foreground(t.text).
		PaddingLeft(1)
	t.agentLabel = lipgloss.NewStyle().Foreground(t.accent).Bold(true)
	t.reasoning = lipgloss.NewStyle().
		Border(lipgloss.NormalBorder(), false, false, false, true).
		BorderForeground(t.faint).
		Foreground(t.muted).
		Italic(true).
		PaddingLeft(1)

	t.card = lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(t.border).
		Padding(0, 1)
	t.cardTitle = lipgloss.NewStyle().Foreground(t.text).Bold(true)
	t.cardBody = lipgloss.NewStyle().Foreground(t.muted)

	t.panelBox = lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(t.border)
	t.panelFocus = t.panelBox.BorderForeground(t.borderFocus)

	t.key = lipgloss.NewStyle().Foreground(t.accent).Bold(true)
	t.keyHelp = lipgloss.NewStyle().Foreground(t.faint)

	t.modal = lipgloss.NewStyle().
		Border(lipgloss.ThickBorder()).
		BorderForeground(t.warn).
		Padding(1, 2)
	return t
}

// gradient renders s with its letters blended across colors.
func gradient(s string, bold bool, colors ...color.Color) string {
	runes := []rune(s)
	if len(runes) == 0 {
		return ""
	}
	steps := lipgloss.Blend1D(len(runes), colors...)
	var out string
	for i, r := range runes {
		out += lipgloss.NewStyle().Foreground(steps[i]).Bold(bold).Render(string(r))
	}
	return out
}
