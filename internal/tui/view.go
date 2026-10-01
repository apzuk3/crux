package tui

import (
	"fmt"
	"strings"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

func (m *model) View() tea.View {
	v := tea.NewView(m.render())
	v.AltScreen = true
	v.MouseMode = tea.MouseModeCellMotion
	v.WindowTitle = "crux · " + m.info.Agent
	v.BackgroundColor = m.th.bg
	switch {
	case m.run.stage == stageApproval:
		v.ProgressBar = tea.NewProgressBar(tea.ProgressBarWarning, int(m.bar.Percent()*100))
	case m.running:
		v.ProgressBar = tea.NewProgressBar(tea.ProgressBarDefault, int(m.bar.Percent()*100))
	}
	return v
}

func (m *model) render() string {
	if m.width == 0 {
		return ""
	}
	if m.width < 50 || m.height < 15 {
		return lipgloss.Place(m.width, m.height, lipgloss.Center, lipgloss.Center,
			m.th.dim.Render("Make the terminal a little bigger\n(at least 50×15)"))
	}

	sideW := m.sidebarWidth()
	mainW := m.width - sideW
	bodyH := m.height - 2

	chatStyle := m.th.panelBox
	chat := chatStyle.Width(mainW).Height(m.chat.Height() + 2).Render(m.chat.View())
	chat = m.scrollHint(chat, mainW)

	inputStyle := m.th.panelBox.Padding(0, 1)
	if m.focus == focusInput && len(m.pending) == 0 {
		inputStyle = m.th.panelFocus.Padding(0, 1)
	}
	input := inputStyle.Width(mainW).Render(m.input.View())

	main := lipgloss.JoinVertical(lipgloss.Left, chat, m.renderStatus(mainW), input)
	body := main
	if sideW > 0 {
		body = lipgloss.JoinHorizontal(lipgloss.Top, main, m.renderSidebar(sideW, bodyH))
	}

	screen := lipgloss.JoinVertical(lipgloss.Left, m.renderHeader(), body, m.renderFooter())
	switch {
	case len(m.pending) > 0:
		screen = m.overlay(screen, m.renderApproval())
	case m.picker != nil:
		screen = m.overlay(screen, m.renderPicker())
	}
	return screen
}

func (m *model) renderHeader() string {
	th := m.th
	logo := gradient(" crux ", true, th.accent, th.accent2, th.accent3)
	agent := th.title.Render(m.info.Agent)
	model := th.dim.Render(m.info.Provider + "/" + m.info.Model)
	left := logo + th.faintText.Render("│ ") + agent + "  " + model
	right := ""
	if m.running {
		right = lipgloss.NewStyle().Foreground(th.accent3).Render(m.spinner() + " working ")
	} else if len(m.pending) > 0 {
		right = lipgloss.NewStyle().Foreground(th.warn).Render("◈ approval needed ")
	}
	gap := max(1, m.width-lipgloss.Width(left)-lipgloss.Width(right))
	return left + strings.Repeat(" ", gap) + right
}

func (m *model) renderFooter() string {
	th := m.th
	var keys [][2]string
	switch {
	case m.picker != nil:
		keys = [][2]string{{"type", "filter"}, {"↑↓", "select"}, {"enter", "switch"}, {"esc", "cancel"}}
	case len(m.pending) > 0:
		keys = [][2]string{{"y", "approve"}, {"n", "reject"}, {"a", "approve all"}, {"esc", "reject all"}}
	case m.focus == focusSidebar:
		keys = [][2]string{{"↑↓", "select"}, {"tab", "back to chat"}, {"ctrl+b", "hide"}}
	default:
		keys = [][2]string{{"enter", "send"}, {"⇧enter", "newline"}}
		if m.running {
			keys = append(keys, [2]string{"esc", "interrupt"})
		}
		keys = append(keys, [2]string{"ctrl+o", "model"}, [2]string{"pgup/dn", "scroll"}, [2]string{"tab", "tools"}, [2]string{"ctrl+r", "thoughts"}, [2]string{"ctrl+b", "sidebar"})
	}
	keys = append(keys, [2]string{"ctrl+c", "quit"})
	var parts []string
	for _, k := range keys {
		parts = append(parts, th.key.Render(k[0])+" "+th.keyHelp.Render(k[1]))
	}
	return ansi.Truncate(" "+strings.Join(parts, th.faintText.Render("  ·  ")), m.width, "…")
}

// scrollHint marks the chat's bottom border when there is more below.
func (m *model) scrollHint(chat string, w int) string {
	if m.chat.AtBottom() {
		return chat
	}
	lines := strings.Split(chat, "\n")
	hint := lipgloss.NewStyle().Foreground(m.th.accent).Render(fmt.Sprintf(" ↓ more · %d%% ", int(m.chat.ScrollPercent()*100)))
	last := lines[len(lines)-1]
	at := max(2, w-lipgloss.Width(hint)-3)
	lines[len(lines)-1] = ansi.Truncate(last, at, "") + hint + ansi.TruncateLeft(last, at+lipgloss.Width(hint), "")
	return strings.Join(lines, "\n")
}

func (m *model) renderApproval() string {
	th := m.th
	c := m.pending[0]
	w := min(72, m.width-8)
	inner := w - 6

	title := lipgloss.NewStyle().Foreground(th.warn).Bold(true).Render("◈ Approval needed")
	if n := len(m.pending); n > 1 {
		title += th.dim.Render(fmt.Sprintf("   1 of %d", n))
	}
	who := c.Agent
	if who == "" {
		who = m.info.Agent
	}
	lines := []string{title, "", th.dim.Render(who + " wants to run"), th.title.Render(c.Name)}
	if t, ok := m.tools[c.Name]; ok && t.Description != "" {
		lines = append(lines, th.dim.Italic(true).Width(inner).Render(t.Description))
	}
	args := preview(prettyJSON(c.Args), inner-4, 12)
	lines = append(lines, "", lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(th.border).
		Foreground(th.accent3).
		Padding(0, 1).
		Width(inner).
		Render(args), "")
	if m.deciding {
		lines = append(lines, lipgloss.NewStyle().Foreground(th.accent3).Render(m.spinner()+" recording decision…"))
	} else {
		btn := func(k, label string, c lipgloss.Style) string {
			return c.Bold(true).Padding(0, 1).Render(k) + " " + th.dim.Render(label)
		}
		lines = append(lines, strings.Join([]string{
			btn("y", "approve", lipgloss.NewStyle().Background(th.ok).Foreground(th.bg)),
			btn("n", "reject", lipgloss.NewStyle().Background(th.bad).Foreground(th.bg)),
			btn("a", "approve all", lipgloss.NewStyle().Background(th.border).Foreground(th.text)),
		}, "   "))
	}
	return th.modal.Width(w).Background(th.panel).Render(strings.Join(lines, "\n"))
}

// overlay draws box centred over screen.
func (m *model) overlay(screen, box string) string {
	x := max(0, (m.width-lipgloss.Width(box))/2)
	y := max(0, (m.height-lipgloss.Height(box))/2)
	return lipgloss.NewCompositor(
		lipgloss.NewLayer(screen),
		lipgloss.NewLayer(box).X(x).Y(y).Z(1),
	).Render()
}
