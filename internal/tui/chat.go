package tui

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image/color"
	"sort"
	"strings"
	"time"

	"charm.land/glamour/v2"
	"charm.land/glamour/v2/styles"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

type blockKind int

const (
	blockUser blockKind = iota
	blockAssistant
	blockReasoning
	blockTool
	blockNotice
)

type toolState int

const (
	toolQueued toolState = iota
	toolAwaiting
	toolRunning
	toolOK
	toolFailed
	toolDenied
)

type noticeKind int

const (
	noticeInfo noticeKind = iota
	noticeError
	noticeCancel
	noticeRefusal
)

// block is one item of the transcript.
type block struct {
	kind  blockKind
	agent string
	text  string
	at    time.Time
	live  bool // still streaming
	took  time.Duration

	notice noticeKind

	call     Call
	state    toolState
	subagent bool
	started  time.Time
	output   string
	err      string
	dur      time.Duration
	children []*block
	turns    int
	activity string

	cache      string
	cacheWidth int
}

func (b *block) invalidate() { b.cacheWidth = 0 }

func (b *block) finish() {
	if b.live {
		b.live = false
		b.took = time.Since(b.at)
		b.invalidate()
	}
}

// busy reports whether the block animates, so it must be drawn every frame.
func (b *block) busy() bool {
	if b.live || b.state == toolRunning {
		return true
	}
	for _, c := range b.children {
		if c.busy() {
			return true
		}
	}
	return false
}

// markdown renders assistant answers, one glamour renderer per width.
type markdown struct {
	dark      bool
	renderers map[int]*glamour.TermRenderer
}

func newMarkdown(dark bool) *markdown {
	return &markdown{dark: dark, renderers: map[int]*glamour.TermRenderer{}}
}

func (md *markdown) render(text string, width int) string {
	r := md.renderers[width]
	if r == nil {
		style := styles.LightStyleConfig
		if md.dark {
			style = styles.DarkStyleConfig
		}
		zero := uint(0)
		style.Document.Margin = &zero
		style.Document.BlockPrefix = ""
		style.Document.BlockSuffix = ""
		var err error
		r, err = glamour.NewTermRenderer(glamour.WithStyles(style), glamour.WithWordWrap(width))
		if err != nil {
			return lipgloss.NewStyle().Width(width).Render(text)
		}
		md.renderers[width] = r
	}
	out, err := r.Render(text)
	if err != nil {
		return lipgloss.NewStyle().Width(width).Render(text)
	}
	return strings.Trim(out, "\n")
}

// renderTranscript draws the whole conversation for a chat of width w.
func (m *model) renderTranscript(w int) string {
	if len(m.blocks) == 0 {
		return m.renderWelcome(w)
	}
	parts := make([]string, 0, len(m.blocks))
	for _, b := range m.blocks {
		if b.busy() || b.cacheWidth != w {
			b.cache = m.renderBlock(b, w)
			b.cacheWidth = w
		}
		if b.cache != "" {
			parts = append(parts, b.cache)
		}
	}
	return "\n" + strings.Join(parts, "\n\n") + "\n"
}

func (m *model) renderBlock(b *block, w int) string {
	th := m.th
	inner := max(10, w-2)
	switch b.kind {
	case blockUser:
		head := th.userLabel.Render("you")
		if !b.at.IsZero() {
			head += "  " + th.faintText.Render(b.at.Format("15:04"))
		}
		body := lipgloss.NewStyle().Width(inner - 2).Foreground(th.text).Render(b.text)
		return pad(th.userBox.Render(head + "\n" + body))

	case blockAssistant:
		name := b.agent
		if name == "" {
			name = m.info.Agent
		}
		head := th.agentLabel.Render("◆ " + name)
		if b.live {
			head += "  " + lipgloss.NewStyle().Foreground(th.accent3).Render(m.spinner()+" writing")
		}
		text := b.text
		if b.live {
			text += "▍"
		}
		return pad(head + "\n" + m.md.render(text, inner))

	case blockReasoning:
		return m.renderReasoning(b, inner)

	case blockTool:
		return pad(m.renderTool(b, inner))

	case blockNotice:
		return pad(m.renderNotice(b, inner))
	}
	return ""
}

func (m *model) renderReasoning(b *block, w int) string {
	th := m.th
	icon := lipgloss.NewStyle().Foreground(th.accent).Render("✻")
	if b.live {
		label := th.dim.Italic(true).Render(" thinking… ")
		lines := strings.Split(strings.TrimSpace(ansi.Wordwrap(b.text, w-4, "")), "\n")
		if len(lines) > 4 && !m.showThoughts {
			lines = lines[len(lines)-4:]
		}
		body := th.reasoning.Render(strings.Join(lines, "\n"))
		return pad(icon + label + th.faintText.Render(m.spinner()) + "\n" + body)
	}
	if strings.TrimSpace(b.text) == "" {
		return ""
	}
	label := " thought"
	if b.took > 0 {
		label += " for " + fmtDur(b.took)
	}
	if !m.showThoughts {
		return pad(icon + th.dim.Italic(true).Render(label) + th.faintText.Render("  ctrl+r to expand"))
	}
	body := th.reasoning.Width(w - 2).Render(strings.TrimSpace(b.text))
	return pad(icon + th.dim.Italic(true).Render(label) + "\n" + body)
}

// toolLook is how a tool call's state is drawn.
func (m *model) toolLook(b *block) (icon, status string, border color.Color) {
	th := m.th
	border = th.border
	switch b.state {
	case toolQueued:
		icon = th.faintText.Render("○")
		status = th.faintText.Render("queued")
	case toolAwaiting:
		icon = lipgloss.NewStyle().Foreground(th.warn).Render("◈")
		status = lipgloss.NewStyle().Foreground(th.warn).Bold(true).Render("needs approval")
		border = th.warn
	case toolRunning:
		icon = lipgloss.NewStyle().Foreground(th.accent3).Render(m.spinner())
		status = lipgloss.NewStyle().Foreground(th.accent3).Render("running " + fmtDur(time.Since(b.started)))
		border = th.accent
	case toolOK:
		icon = lipgloss.NewStyle().Foreground(th.ok).Render("✓")
		status = th.dim.Render(fmtDur(b.dur))
	case toolFailed:
		icon = lipgloss.NewStyle().Foreground(th.bad).Render("✗")
		status = lipgloss.NewStyle().Foreground(th.bad).Render("failed")
		if b.dur > 0 {
			status += th.dim.Render(" " + fmtDur(b.dur))
		}
		border = th.bad
	case toolDenied:
		icon = th.faintText.Render("⊘")
		status = th.faintText.Render("rejected")
	}
	return icon, status, border
}

// renderTool draws a top-level tool call as a card. A subagent's card lists
// the calls the subagent makes, one row each.
func (m *model) renderTool(b *block, w int) string {
	th := m.th
	icon, status, border := m.toolLook(b)
	name := b.call.Name
	kindTag := ""
	if b.subagent {
		name = strings.TrimPrefix(name, "agent_")
		kindTag = lipgloss.NewStyle().Foreground(th.accent2).Render("◇ ")
	}
	cw := w - 4 // border and padding
	title := icon + " " + kindTag + th.cardTitle.Render(name)
	args := argsSummary(b.call.Args)
	room := cw - lipgloss.Width(title) - lipgloss.Width(status) - 3
	if room > 6 && args != "" {
		title += "  " + th.dim.Render(ansi.Truncate(args, room, "…"))
	}
	lines := []string{spread(title, status, cw)}

	if b.subagent {
		rail := lipgloss.NewStyle().Foreground(th.accent2).Render("│ ")
		for _, c := range b.children {
			lines = append(lines, rail+m.renderToolRow(c, cw-2))
		}
		if b.state == toolRunning {
			act := b.activity
			if act == "" {
				act = "starting"
			}
			turns := ""
			if b.turns > 0 {
				turns = fmt.Sprintf(" · turn %d", b.turns)
			}
			lines = append(lines, rail+lipgloss.NewStyle().Foreground(th.accent2).Italic(true).Render(act+"…")+th.faintText.Render(turns))
		}
	}

	switch {
	case b.state == toolDenied:
		lines = append(lines, th.faintText.Render(preview(b.err, cw, 2)))
	case b.err != "":
		lines = append(lines, lipgloss.NewStyle().Foreground(th.bad).Render(preview("✗ "+b.err, cw, 2)))
	case b.output != "":
		lines = append(lines, th.cardBody.Render(preview("→ "+b.output, cw, 2)))
	}
	return th.card.BorderForeground(border).Width(w).Render(strings.Join(lines, "\n"))
}

// renderToolRow draws a call made inside a subagent on one line, with the
// calls of a nested subagent below it.
func (m *model) renderToolRow(b *block, w int) string {
	th := m.th
	icon, status, _ := m.toolLook(b)
	name := b.call.Name
	if b.subagent {
		name = "◇ " + strings.TrimPrefix(name, "agent_")
	}
	left := icon + " " + lipgloss.NewStyle().Foreground(th.text).Render(name)
	detail := argsSummary(b.call.Args)
	switch {
	case b.err != "":
		detail = "✗ " + oneLine(b.err)
	case b.output != "":
		detail += "  → " + oneLine(b.output)
	}
	if room := w - lipgloss.Width(left) - lipgloss.Width(status) - 3; room > 6 && detail != "" {
		left += "  " + th.dim.Render(ansi.Truncate(detail, room, "…"))
	}
	row := spread(left, status, w)
	for _, c := range b.children {
		row += "\n  " + m.renderToolRow(c, w-2)
	}
	return row
}

// spread puts left and right on one line of width w.
func spread(left, right string, w int) string {
	gap := max(1, w-lipgloss.Width(left)-lipgloss.Width(right))
	return left + strings.Repeat(" ", gap) + right
}

func (m *model) renderNotice(b *block, w int) string {
	th := m.th
	var color = th.info
	icon := "ℹ"
	switch b.notice {
	case noticeError:
		color, icon = th.bad, "✗"
	case noticeCancel:
		color, icon = th.faint, "■"
	case noticeRefusal:
		color, icon = th.warn, "⚑"
	}
	return lipgloss.NewStyle().
		Border(lipgloss.ThickBorder(), false, false, false, true).
		BorderForeground(color).
		Foreground(color).
		PaddingLeft(1).
		Width(w).
		Render(icon + " " + b.text)
}

var logo = []string{
	"  ██████ ██████  ██   ██ ██   ██",
	" ██      ██   ██ ██   ██  ██ ██ ",
	" ██      ██████  ██   ██   ███  ",
	" ██      ██   ██ ██   ██  ██ ██ ",
	"  ██████ ██   ██  █████  ██   ██",
}

func (m *model) renderWelcome(w int) string {
	th := m.th
	var art []string
	for i, l := range logo {
		stops := []color.Color{th.accent, th.accent2, th.accent3}
		shift := i % len(stops)
		stops = append(stops[shift:], stops[:shift]...)
		art = append(art, gradient(l, true, stops...))
	}
	lines := []string{strings.Join(art, "\n"), ""}
	lines = append(lines, th.title.Render(m.info.Agent)+th.dim.Render("  is ready"))
	lines = append(lines, th.pillAccent.Render(m.info.Provider)+" "+th.pill.Render(m.info.Model))
	lines = append(lines, "")
	n := len(m.info.Tools)
	lines = append(lines, th.dim.Render(fmt.Sprintf("%d tools · up to %d turns per run", n, m.info.MaxTurns)))
	lines = append(lines, "")
	for _, hint := range [][2]string{{"enter", "send"}, {"shift+enter", "new line"}, {"ctrl+o", "switch model"}, {"tab", "browse tools"}, {"/help", "commands"}} {
		lines = append(lines, th.key.Render(fmt.Sprintf("%12s", hint[0]))+"  "+th.keyHelp.Render(hint[1]))
	}
	content := lipgloss.JoinVertical(lipgloss.Center, lines...)
	h := max(lipgloss.Height(content)+2, m.chat.Height())
	return lipgloss.Place(w, h, lipgloss.Center, lipgloss.Center, content)
}

const helpText = `Commands: /model [filter] switches models · /compact summarises older messages to save context · /clear clears the screen · /quit exits · /help shows this.
Keys: enter send · ctrl+o switch model · shift+enter or ctrl+j new line · ↑ recall last prompt · esc interrupt · pgup/pgdn scroll · tab browse tools · ctrl+r show thoughts · ctrl+b toggle sidebar · ctrl+c quit.
While approving: y approve · n reject · a approve all · esc reject all.`

// spinner is the current frame of the shared spinner.
func (m *model) spinner() string {
	frames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
	return frames[m.frame%len(frames)]
}

func pad(s string) string {
	return lipgloss.NewStyle().PaddingLeft(1).Render(s)
}

// argsSummary shows JSON arguments as key=value pairs on one line.
func argsSummary(args string) string {
	var obj map[string]any
	if err := json.Unmarshal([]byte(args), &obj); err != nil {
		return oneLine(args)
	}
	keys := make([]string, 0, len(obj))
	for k := range obj {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		v, _ := json.Marshal(obj[k])
		s := string(v)
		if str, ok := obj[k].(string); ok {
			s = str
		}
		parts = append(parts, k+"="+oneLine(s))
	}
	return strings.Join(parts, " ")
}

// prettyJSON indents JSON, or returns s unchanged when it is not JSON.
func prettyJSON(s string) string {
	var buf bytes.Buffer
	if json.Indent(&buf, []byte(s), "", "  ") != nil {
		return s
	}
	return buf.String()
}

func oneLine(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// preview shows at most n lines of s, each cut to width w.
func preview(s string, w, n int) string {
	s = strings.TrimSpace(s)
	lines := strings.Split(s, "\n")
	more := len(lines) > n
	if more {
		lines = lines[:n]
	}
	for i, l := range lines {
		lines[i] = ansi.Truncate(strings.TrimRight(l, " \t"), w, "…")
	}
	if more {
		lines[len(lines)-1] = ansi.Truncate(lines[len(lines)-1]+" …", w, "…")
	}
	return strings.Join(lines, "\n")
}

func fmtDur(d time.Duration) string {
	switch {
	case d <= 0:
		return "0ms"
	case d < time.Second:
		return fmt.Sprintf("%dms", d.Milliseconds())
	case d < time.Minute:
		return fmt.Sprintf("%.1fs", d.Seconds())
	}
	return d.Round(time.Second).String()
}

func fmtTokens(n int) string {
	switch {
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1e6)
	case n >= 1_000:
		return fmt.Sprintf("%.1fk", float64(n)/1e3)
	}
	return fmt.Sprint(n)
}
