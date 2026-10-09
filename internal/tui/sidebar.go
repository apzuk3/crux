package tui

import (
	"fmt"
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
	section := func(title, extra string) string {
		t := th.label.Render(strings.ToUpper(title))
		if extra != "" {
			t += " " + th.faintText.Render(extra)
		}
		rule := max(0, inner-lipgloss.Width(t)-1)
		return t + " " + lipgloss.NewStyle().Foreground(th.border).Render(strings.Repeat("─", rule))
	}

	var out []string
	out = append(out, section("agent", ""))
	out = append(out, th.agentLabel.Render("◆ ")+th.title.Render(ansi.Truncate(m.info.Agent, inner-2, "…")))
	switchHint := th.faintText.Render("^o")
	out = append(out, "  "+spread(th.dim.Render(ansi.Truncate(m.info.Model, inner-6, "…")), switchHint, inner-2))
	out = append(out, "  "+th.faintText.Render(ansi.Truncate(fmt.Sprintf("%s · %d turns max", m.info.Provider, m.info.MaxTurns), inner-2, "…")))
	out = append(out, "")

	rows := m.sidebarTools()
	out = append(out, section("tools", fmt.Sprint(len(rows))))
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
		out = append(out, m.sidebarTool(r, i == m.selected && m.focus == focusSidebar, inner))
	}
	out = append(out, "")

	out = append(out, section("session", ""))
	stat := func(k, v string) string {
		return th.dim.Render(fmt.Sprintf("%-8s", k)) + th.title.UnsetBold().Render(v)
	}
	out = append(out,
		stat("runs", fmt.Sprint(m.runCount)),
		stat("turns", fmt.Sprint(m.turns)),
		stat("tools", fmt.Sprintf("%d calls", m.toolCalls)),
		stat("tokens", "↑"+fmtTokens(m.usage.In)+"  ↓"+fmtTokens(m.usage.Out)),
	)
	if m.usage.In > 0 && m.usage.CacheRead > 0 {
		out = append(out, stat("cache", fmt.Sprintf("%d%% hit", m.usage.CacheRead*100/m.usage.In)))
	}
	if m.run.ttft > 0 {
		out = append(out, stat("ttft", fmtDur(m.run.ttft)))
	}
	if id := m.info.SessionID; id != "" {
		out = append(out, stat("id", ansi.Truncate(id, inner-8, "…")))
	}

	body := strings.Join(out, "\n")
	if m.focus == focusSidebar && m.selected < len(rows) {
		t := rows[m.selected].tool
		desc := t.Description
		if desc == "" {
			desc = "No description."
		}
		card := lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(th.accent).
			Width(inner).
			Padding(0, 1).
			Render(th.title.Render(t.Name) + "\n" + th.dim.Render(desc))
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
