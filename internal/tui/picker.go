package tui

import (
	"fmt"
	"strings"
	"unicode"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// picker is the model switcher: a filter and the models that match it.
type picker struct {
	filter    string
	cursor    int
	switching bool
}

type switchedMsg struct {
	to        Model
	sessionID string
	err       error
}

const pickerRows = 12

// openPicker shows the model switcher, filtered by filter.
func (m *model) openPicker(filter string) tea.Cmd {
	if m.running || m.deciding || len(m.pending) > 0 {
		m.flash = "finish the current run before switching models"
		return nil
	}
	m.flash = ""
	m.picker = &picker{filter: filter}
	m.input.Blur()
	return m.tick()
}

func (m *model) closePicker() {
	m.picker = nil
	m.setFocus(m.focus)
}

// pickerMatches lists the models whose provider/name contains every word of
// the filter.
func (m *model) pickerMatches() []Model {
	words := strings.Fields(strings.ToLower(m.picker.filter))
	var out []Model
	for _, mod := range m.info.Models {
		hay := strings.ToLower(mod.Provider + "/" + mod.Name)
		ok := true
		for _, w := range words {
			if !strings.Contains(hay, w) {
				ok = false
				break
			}
		}
		if ok {
			out = append(out, mod)
		}
	}
	return out
}

func (m *model) pickerKey(msg tea.KeyPressMsg) tea.Cmd {
	p := m.picker
	if p.switching {
		return nil
	}
	matches := m.pickerMatches()
	switch msg.String() {
	case "esc", "ctrl+o":
		m.closePicker()
		return nil
	case "up", "ctrl+p":
		p.cursor = max(0, p.cursor-1)
		return nil
	case "down", "ctrl+n":
		p.cursor = min(max(0, len(matches)-1), p.cursor+1)
		return nil
	case "pgup":
		p.cursor = max(0, p.cursor-pickerRows)
		return nil
	case "pgdown":
		p.cursor = min(max(0, len(matches)-1), p.cursor+pickerRows)
		return nil
	case "backspace":
		if r := []rune(p.filter); len(r) > 0 {
			p.filter = string(r[:len(r)-1])
			p.cursor = 0
		}
		return nil
	case "enter":
		if len(matches) == 0 {
			return nil
		}
		to := matches[p.cursor]
		if to == m.current() {
			m.closePicker()
			return nil
		}
		return m.switchModel(to)
	}
	if msg.Text != "" && unicode.IsPrint([]rune(msg.Text)[0]) {
		p.filter += msg.Text
		p.cursor = 0
	}
	return nil
}

func (m *model) current() Model {
	return Model{Provider: m.info.Provider, Name: m.info.Model}
}

func (m *model) switchModel(to Model) tea.Cmd {
	m.picker.switching = true
	m.runs.Add(1)
	ctx, backend := m.ctx, m.backend
	return tea.Batch(m.tick(), func() tea.Msg {
		defer m.runs.Done()
		id, err := backend.SwitchModel(ctx, to.Provider, to.Name)
		return switchedMsg{to: to, sessionID: id, err: err}
	})
}

func (m *model) switched(msg switchedMsg) {
	m.closePicker()
	if msg.err != nil {
		m.addNotice(noticeError, "Could not switch to "+msg.to.Name+": "+msg.err.Error())
		return
	}
	from := m.info.Model
	m.info.Provider, m.info.Model, m.info.SessionID = msg.to.Provider, msg.to.Name, msg.sessionID
	m.addNotice(noticeInfo, fmt.Sprintf("Switched from %s to %s/%s. The conversation continues in session %s; subagents keep their own models.",
		from, msg.to.Provider, msg.to.Name, shortID(msg.sessionID)))
}

func (m *model) renderPicker() string {
	th := m.th
	w := min(64, m.width-8)
	inner := w - 6

	title := lipgloss.NewStyle().Foreground(th.accent).Bold(true).Render("◆ Switch model")
	cur := th.dim.Render("now ") + th.title.Render(m.info.Model)
	lines := []string{spread(title, cur, inner), "", m.pickerFilterLine(), ""}
	lines = append(lines, m.pickerRows(inner)...)
	lines = append(lines, "", m.pickerFooter())
	return th.modal.BorderForeground(th.accent).Width(w).Background(th.panel).Render(strings.Join(lines, "\n"))
}

func (m *model) pickerFilterLine() string {
	th := m.th
	cursor := ""
	if m.frame/4%2 == 0 {
		cursor = "▍"
	}
	filter := m.picker.filter
	if filter == "" {
		filter = th.faintText.Italic(true).Render("type to filter, e.g. \"mini\" or \"anthropic\"")
	} else {
		filter = th.title.Render(filter)
	}
	return lipgloss.NewStyle().Foreground(th.accent2).Render("› ") + filter + lipgloss.NewStyle().Foreground(th.accent2).Render(cursor)
}

// pickerRows lists the window of matches around the cursor, grouped by
// provider, with how many are above and below it.
func (m *model) pickerRows(inner int) []string {
	th := m.th
	p := m.picker
	matches := m.pickerMatches()
	if p.cursor >= len(matches) {
		p.cursor = max(0, len(matches)-1)
	}
	var lines []string
	if len(matches) == 0 {
		lines = append(lines, th.faintText.Render("  no models match"))
	}
	start := min(max(0, p.cursor-pickerRows/2), max(0, len(matches)-pickerRows))
	end := min(len(matches), start+pickerRows)
	lastProvider := ""
	if start > 0 {
		lastProvider = matches[start-1].Provider
		lines = append(lines, th.faintText.Render(fmt.Sprintf("  ↑ %d more", start)))
	}
	for i := start; i < end; i++ {
		mod := matches[i]
		if mod.Provider != lastProvider {
			lastProvider = mod.Provider
			lines = append(lines, th.label.Render(strings.ToUpper(mod.Provider)))
		}
		lines = append(lines, m.pickerRow(mod, i == p.cursor, inner))
	}
	if rest := len(matches) - end; rest > 0 {
		lines = append(lines, th.faintText.Render(fmt.Sprintf("  ↓ %d more", rest)))
	}
	return lines
}

func (m *model) pickerRow(mod Model, selected bool, inner int) string {
	th := m.th
	mark := "  "
	if mod == m.current() {
		mark = lipgloss.NewStyle().Foreground(th.ok).Render("● ")
	}
	name := ansi.Truncate(mod.Name, inner-4, "…")
	if selected {
		return lipgloss.NewStyle().Background(th.accent).Foreground(lipgloss.Color("#FFFFFF")).Bold(true).Width(inner).Render(mark + name)
	}
	return mark + lipgloss.NewStyle().Foreground(th.text).Render(name)
}

func (m *model) pickerFooter() string {
	th := m.th
	if m.picker.switching {
		return lipgloss.NewStyle().Foreground(th.accent3).Render(m.spinner() + " switching…")
	}
	return th.key.Render("↑↓") + th.keyHelp.Render(" select  ") + th.key.Render("enter") + th.keyHelp.Render(" switch  ") + th.key.Render("esc") + th.keyHelp.Render(" cancel")
}

func shortID(id string) string {
	if len(id) > 8 {
		return id[:8]
	}
	return id
}
