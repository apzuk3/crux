package tui

import (
	"fmt"
	"slices"
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// sidebarRow is one selectable tool in the sidebar.
type sidebarRow struct {
	tool  Tool
	depth int
}

// sidebarTools lists the tools in display order, subagents' tools nested.
func (m *model) sidebarTools() []sidebarRow {
	var rows []sidebarRow
	var walk func([]Tool, int)
	walk = func(tools []Tool, depth int) {
		for _, t := range tools {
			rows = append(rows, sidebarRow{tool: t, depth: depth})
			walk(t.SubAgent, depth+1)
		}
	}
	walk(m.info.Tools, 0)
	return rows
}

func (m *model) renderSidebar(w, h int) string {
	th := m.th
	inner := w - 4
	rows := m.sidebarTools()
	out := slices.Concat(m.sidebarAgentSection(inner), m.sidebarToolsSection(rows, inner), m.sidebarSessionSection(inner))

	body := strings.Join(out, "\n")
	if m.focus == focusSidebar && m.selected < len(rows) {
		card := m.sidebarToolCard(rows[m.selected].tool, inner)
		space := h - 2 - lipgloss.Height(body) - lipgloss.Height(card)
		if space >= 1 {
			body += strings.Repeat("\n", space) + card
		}
	}

	style := th.panelBox
	if m.focus == focusSidebar {
		style = th.panelFocus
	}
	return style.Width(w).Height(h).Padding(0, 1).Render(body)
}

// sidebarSection is a section title with a rule filling the rest of inner.
func (m *model) sidebarSection(title, extra string, inner int) string {
	th := m.th
	t := th.label.Render(strings.ToUpper(title))
	if extra != "" {
		t += " " + th.faintText.Render(extra)
	}
	rule := max(0, inner-lipgloss.Width(t)-1)
	return t + " " + lipgloss.NewStyle().Foreground(th.border).Render(strings.Repeat("─", rule))
}

func (m *model) sidebarStat(k, v string) string {
	return m.th.dim.Render(fmt.Sprintf("%-8s", k)) + m.th.title.UnsetBold().Render(v)
}

func (m *model) sidebarAgentSection(inner int) []string {
	th := m.th
	switchHint := th.faintText.Render("^o")
	return []string{
		m.sidebarSection("agent", "", inner),
		th.agentLabel.Render("◆ ") + th.title.Render(ansi.Truncate(m.info.Agent, inner-2, "…")),
		"  " + spread(th.dim.Render(ansi.Truncate(m.info.Model, inner-6, "…")), switchHint, inner-2),
		"  " + th.faintText.Render(ansi.Truncate(fmt.Sprintf("%s · %d turns max", m.info.Provider, m.info.MaxTurns), inner-2, "…")),
		"",
	}
}

// sidebarToolsSection lists the tools, top-level ones grouped by toolset.
func (m *model) sidebarToolsSection(rows []sidebarRow, inner int) []string {
	th := m.th
	out := []string{m.sidebarSection("tools", fmt.Sprint(len(rows)), inner)}
	if len(rows) == 0 {
		out = append(out, th.faintText.Italic(true).Render("  no tools"))
	}
	lastSet := ""
	for i, r := range rows {
		if r.depth == 0 && r.tool.Toolset != lastSet {
			lastSet = r.tool.Toolset
			if lastSet != "" {
				out = append(out, th.faintText.Render("▾ "+lastSet))
			}
		}
		selected := i == m.selected && m.focus == focusSidebar
		out = append(out, m.sidebarTool(r, selected, inner))
	}
	return append(out, "")
}

func (m *model) sidebarSessionSection(inner int) []string {
	out := []string{
		m.sidebarSection("session", "", inner),
		m.sidebarStat("runs", fmt.Sprint(m.runCount)),
		m.sidebarStat("turns", fmt.Sprint(m.turns)),
		m.sidebarStat("tools", fmt.Sprintf("%d calls", m.toolCalls)),
		m.sidebarStat("tokens", "↑"+fmtTokens(m.usage.In)+"  ↓"+fmtTokens(m.usage.Out)),
	}
	if m.usage.In > 0 && m.usage.CacheRead > 0 {
		out = append(out, m.sidebarStat("cache", fmt.Sprintf("%d%% hit", m.usage.CacheRead*100/m.usage.In)))
	}
	if m.run.ttft > 0 {
		out = append(out, m.sidebarStat("ttft", fmtDur(m.run.ttft)))
	}
	if id := m.info.SessionID; id != "" {
		out = append(out, m.sidebarStat("id", ansi.Truncate(id, inner-8, "…")))
	}
	return out
}

// sidebarToolCard describes the selected tool.
func (m *model) sidebarToolCard(t Tool, inner int) string {
	th := m.th
	desc := t.Description
	if desc == "" {
		desc = "No description."
	}
	return lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(th.accent).
		Width(inner).
		Padding(0, 1).
		Render(th.title.Render(t.Name) + "\n" + th.dim.Render(desc))
}

func (m *model) sidebarTool(r sidebarRow, selected bool, w int) string {
	th := m.th
	t := r.tool
	st := m.stats[t.Name]
	indent := strings.Repeat("  ", r.depth)
	if r.depth > 0 {
		indent = strings.Repeat("  ", r.depth-1) + th.faintText.Render("└ ")
	}

	icon := th.faintText.Render("○")
	switch {
	case st != nil && st.running > 0:
		icon = lipgloss.NewStyle().Foreground(th.accent3).Render(m.spinner())
	case st != nil && st.calls > 0 && st.failed == st.calls:
		icon = lipgloss.NewStyle().Foreground(th.bad).Render("●")
	case st != nil && st.calls > 0:
		icon = lipgloss.NewStyle().Foreground(th.ok).Render("●")
	case t.IsSubAgent:
		icon = lipgloss.NewStyle().Foreground(th.accent2).Render("◇")
	}

	name := t.Name
	nameStyle := lipgloss.NewStyle().Foreground(th.text)
	if t.IsSubAgent {
		name = strings.TrimPrefix(name, "agent_")
		nameStyle = nameStyle.Foreground(th.accent2)
	}
	if st != nil && st.running > 0 {
		nameStyle = nameStyle.Bold(true)
	}

	var badge string
	if t.Approval {
		badge += lipgloss.NewStyle().Foreground(th.warn).Render(" ◈")
	}
	var right string
	if st != nil && st.calls > 0 {
		right = th.faintText.Render(fmt.Sprintf("%d× %s", st.calls, fmtDur(st.last)))
	}

	left := indent + icon + " "
	room := w - lipgloss.Width(left) - lipgloss.Width(badge) - lipgloss.Width(right) - 1
	line := left + nameStyle.Render(ansi.Truncate(name, max(4, room), "…")) + badge
	if gap := w - lipgloss.Width(line) - lipgloss.Width(right); gap > 0 && right != "" {
		line += strings.Repeat(" ", gap) + right
	}
	if selected {
		return lipgloss.NewStyle().Background(th.border).Width(w).Render(line)
	}
	return line
}
