package tui

import (
	"context"
	"errors"
	"strings"
	"sync"
	"time"

	"charm.land/bubbles/v2/progress"
	"charm.land/bubbles/v2/textarea"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
)

type (
	eventMsg   Event
	runDoneMsg struct{ err error }
	decidedMsg struct {
		id  string
		err error
	}
	tickMsg struct{}
)

const tickEvery = 90 * time.Millisecond

// stage is where the top-level run is in its lifecycle.
type stage int

const (
	stageIdle stage = iota
	stageRequest
	stageThinking
	stageTools
	stageWriting
	stageApproval
	stageDone
	stageFailed
	stageCancelled
)

// runState is the lifecycle of the current (or last) top-level run.
type runState struct {
	stage      stage
	started    time.Time
	elapsed    time.Duration
	turn       int
	toolsTotal int
	toolsDone  int
	ttft       time.Duration
	outcome    string
}

// toolStat is what the sidebar shows for one tool.
type toolStat struct {
	calls, running, failed int
	last                   time.Duration
}

type focus int

const (
	focusInput focus = iota
	focusSidebar
)

type model struct {
	info    Info
	backend Backend
	ctx     context.Context
	runs    *sync.WaitGroup
	cancel  context.CancelFunc // cancels the active run

	th       theme
	themeSet bool
	md       *markdown

	width, height int
	chat          viewport.Model
	input         textarea.Model
	bar           progress.Model
	barColors     *barColors

	blocks []*block
	calls  map[string]*block // tool call ID -> its card
	stats  map[string]*toolStat
	tools  map[string]Tool // every tool, including subagents' own

	running  bool
	run      runState
	frame    int
	ticking  bool
	dirty    bool
	follow   bool
	queue    []string
	history  []string
	pending  []Call
	deciding bool
	flash    string
	picker   *picker // open model switcher

	runCount, turns, toolCalls int
	usage                      Usage

	focus        focus
	selected     int
	showSidebar  bool
	showThoughts bool
}

func newModel(info Info, backend Backend, ctx context.Context, runs *sync.WaitGroup) *model {
	m := &model{
		info:        info,
		backend:     backend,
		ctx:         ctx,
		runs:        runs,
		calls:       map[string]*block{},
		stats:       map[string]*toolStat{},
		tools:       map[string]Tool{},
		follow:      true,
		showSidebar: true,
		barColors:   &barColors{},
	}
	var index func([]Tool)
	index = func(tools []Tool) {
		for _, t := range tools {
			m.tools[t.Name] = t
			index(t.SubAgent)
		}
	}
	index(info.Tools)

	m.setTheme(true)
	m.chat = viewport.New()
	m.chat.MouseWheelEnabled = true
	m.chat.MouseWheelDelta = 3
	m.chat.KeyMap.Up.SetEnabled(false)
	m.chat.KeyMap.Down.SetEnabled(false)
	m.chat.KeyMap.Left.SetEnabled(false)
	m.chat.KeyMap.Right.SetEnabled(false)
	m.chat.KeyMap.HalfPageUp.SetEnabled(false)
	m.chat.KeyMap.HalfPageDown.SetEnabled(false)

	m.input = textarea.New()
	m.input.Placeholder = "Ask " + info.Agent + " anything…"
	m.input.ShowLineNumbers = false
	m.input.CharLimit = 0
	m.input.DynamicHeight = true
	m.input.MinHeight = 1
	m.input.MaxHeight = 8
	m.input.SetPromptFunc(2, func(p textarea.PromptInfo) string {
		if p.LineNumber == 0 {
			return lipgloss.NewStyle().Foreground(m.th.accent2).Bold(true).Render("❯ ")
		}
		return "  "
	})
	m.input.KeyMap.InsertNewline.SetKeys("shift+enter", "ctrl+j", "alt+enter")
	m.input.Focus()
	m.styleInput()

	m.bar = progress.New(
		progress.WithoutPercentage(),
		progress.WithFillCharacters('━', '━'),
		progress.WithColorFunc(m.barColors.at),
		progress.WithSpringOptions(18, 1),
	)
	m.barColors.set(m.th, stageIdle, 0)

	for _, e := range info.History {
		m.apply(e, true)
	}
	return m
}

func (m *model) setTheme(dark bool) {
	m.th = newTheme(dark)
	m.md = newMarkdown(dark)
	for _, b := range m.blocks {
		b.invalidate()
	}
	m.styleInput()
	m.bar.EmptyColor = m.th.border
	m.barColors.set(m.th, m.run.stage, m.frame)
	m.dirty = true
}

func (m *model) styleInput() {
	s := textarea.DefaultStyles(m.th.dark)
	for _, st := range []*textarea.StyleState{&s.Focused, &s.Blurred} {
		st.Base = lipgloss.NewStyle()
		st.CursorLine = lipgloss.NewStyle()
		st.Text = lipgloss.NewStyle().Foreground(m.th.text)
		st.Placeholder = lipgloss.NewStyle().Foreground(m.th.faint).Italic(true)
	}
	s.Cursor.Color = m.th.accent2
	m.input.SetStyles(s)
}

func (m *model) Init() tea.Cmd {
	return tea.Batch(tea.RequestBackgroundColor, m.input.Focus())
}

func (m *model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd
	switch msg := msg.(type) {
	case tea.BackgroundColorMsg:
		if !m.themeSet {
			m.themeSet = true
			m.setTheme(msg.IsDark())
		}

	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.layout()

	case eventMsg:
		cmds = append(cmds, m.apply(Event(msg), false))

	case runDoneMsg:
		cmds = append(cmds, m.runDone(msg.err))

	case decidedMsg:
		cmds = append(cmds, m.decided(msg))

	case switchedMsg:
		m.switched(msg)

	case tickMsg:
		m.ticking = false
		m.frame++
		m.barColors.set(m.th, m.run.stage, m.frame)
		m.dirty = true
		if m.animating() {
			cmds = append(cmds, m.tick())
		}

	case progress.FrameMsg:
		var cmd tea.Cmd
		m.bar, cmd = m.bar.Update(msg)
		cmds = append(cmds, cmd)

	case tea.MouseWheelMsg:
		var cmd tea.Cmd
		m.chat, cmd = m.chat.Update(msg)
		m.follow = m.chat.AtBottom()
		cmds = append(cmds, cmd)

	case tea.PasteMsg:
		if m.picker != nil {
			m.picker.filter += oneLine(msg.Content)
			m.picker.cursor = 0
		} else if m.focus == focusInput && len(m.pending) == 0 {
			var cmd tea.Cmd
			m.input, cmd = m.input.Update(msg)
			cmds = append(cmds, cmd)
			m.layout()
		}

	case tea.KeyPressMsg:
		cmds = append(cmds, m.key(msg))

	default:
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		cmds = append(cmds, cmd)
	}

	if m.dirty {
		m.refresh()
	}
	return m, tea.Batch(cmds...)
}

func (m *model) key(msg tea.KeyPressMsg) tea.Cmd {
	k := msg.String()
	switch k {
	case "ctrl+c":
		if m.cancel != nil {
			m.cancel()
		}
		return tea.Quit
	}
	if m.picker != nil {
		return m.pickerKey(msg)
	}
	switch k {
	case "ctrl+o":
		return m.openPicker("")
	case "ctrl+b":
		m.showSidebar = !m.showSidebar
		m.layout()
		return nil
	case "ctrl+r":
		m.showThoughts = !m.showThoughts
		for _, b := range m.blocks {
			b.invalidate()
		}
		m.dirty = true
		return nil
	case "pgup":
		m.chat.PageUp()
		m.follow = false
		return nil
	case "pgdown":
		m.chat.PageDown()
		m.follow = m.chat.AtBottom()
		return nil
	case "ctrl+home":
		m.chat.GotoTop()
		m.follow = false
		return nil
	case "ctrl+end":
		m.chat.GotoBottom()
		m.follow = true
		return nil
	}

	if len(m.pending) > 0 {
		return m.approvalKey(k)
	}

	switch k {
	case "esc":
		if m.running && m.cancel != nil {
			m.cancel()
			m.flash = "interrupting…"
			return nil
		}
		if m.focus == focusSidebar {
			m.setFocus(focusInput)
		}
		return nil
	case "tab":
		if m.focus == focusInput {
			m.setFocus(focusSidebar)
		} else {
			m.setFocus(focusInput)
		}
		return nil
	}

	if m.focus == focusSidebar {
		switch k {
		case "up", "k":
			m.selected = max(0, m.selected-1)
		case "down", "j":
			m.selected = min(len(m.sidebarTools())-1, m.selected+1)
		case "enter":
			m.setFocus(focusInput)
		}
		return nil
	}

	switch k {
	case "enter":
		return m.submit()
	case "up":
		if m.input.Value() == "" && len(m.history) > 0 {
			m.input.SetValue(m.history[len(m.history)-1])
			m.layout()
			return nil
		}
	}
	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	m.layout()
	return cmd
}

func (m *model) setFocus(f focus) {
	m.focus = f
	if f == focusInput {
		m.input.Focus()
	} else {
		m.input.Blur()
	}
}

func (m *model) submit() tea.Cmd {
	text := strings.TrimSpace(m.input.Value())
	if text == "" {
		return nil
	}
	m.input.Reset()
	m.layout()
	switch text {
	case "/quit", "/exit", "/q":
		return tea.Quit
	case "/clear":
		m.blocks = nil
		m.calls = map[string]*block{}
		m.dirty = true
		return nil
	case "/help":
		m.addNotice(noticeInfo, helpText)
		return nil
	}
	if text == "/model" || strings.HasPrefix(text, "/model ") {
		return m.openPicker(strings.TrimSpace(strings.TrimPrefix(text, "/model")))
	}
	m.history = append(m.history, text)
	if m.running || m.deciding {
		m.queue = append(m.queue, text)
		m.flash = "queued — sent when the agent finishes"
		return nil
	}
	return m.send(text)
}

// send shows the prompt and starts a run with it.
func (m *model) send(text string) tea.Cmd {
	m.blocks = append(m.blocks, &block{kind: blockUser, text: text, at: time.Now()})
	m.follow = true
	m.dirty = true
	return m.start(text)
}

// start runs the backend in the background. Empty input resumes.
func (m *model) start(input string) tea.Cmd {
	ctx, cancel := context.WithCancel(m.ctx)
	m.cancel = cancel
	m.running = true
	m.flash = ""
	if input != "" || m.run.started.IsZero() {
		m.run = runState{started: time.Now()}
	}
	m.run.stage, m.run.elapsed = stageRequest, 0
	m.runs.Add(1)
	backend := m.backend
	return tea.Batch(
		func() tea.Msg {
			defer m.runs.Done()
			defer cancel()
			return runDoneMsg{err: backend.Run(ctx, input)}
		},
		m.bar.SetPercent(0.04),
		m.tick(),
	)
}

func (m *model) runDone(err error) tea.Cmd {
	m.running = false
	m.cancel = nil
	m.flash = ""
	if m.run.elapsed == 0 {
		m.run.elapsed = time.Since(m.run.started)
	}
	for _, b := range m.calls {
		if b.state == toolRunning {
			b.state = toolFailed
			b.err = "interrupted"
			b.invalidate()
		}
	}
	m.finishLive()
	switch {
	case errors.Is(err, context.Canceled):
		m.run.stage = stageCancelled
		m.addNotice(noticeCancel, "Interrupted. Send a message to continue.")
	case err != nil:
		m.run.stage = stageFailed
		m.addNotice(noticeError, err.Error())
	}
	m.barColors.set(m.th, m.run.stage, m.frame)

	var cmds []tea.Cmd
	if m.pending = m.backend.Pending(); len(m.pending) > 0 {
		m.run.stage = stageApproval
		for _, c := range m.pending {
			if b := m.calls[c.ID]; b != nil {
				b.state = toolAwaiting
				b.invalidate()
			}
		}
		m.barColors.set(m.th, stageApproval, m.frame)
		cmds = append(cmds, m.bar.SetPercent(0.6))
		m.input.Blur()
	} else if len(m.queue) > 0 && err == nil {
		next := m.queue[0]
		m.queue = m.queue[1:]
		cmds = append(cmds, m.send(next))
	}
	m.dirty = true
	switch m.run.stage {
	case stageDone, stageFailed:
		cmds = append(cmds, m.bar.SetPercent(1))
	}
	return tea.Batch(cmds...)
}

func (m *model) approvalKey(k string) tea.Cmd {
	if m.deciding {
		return nil
	}
	switch k {
	case "y", "enter":
		return m.decide(m.pending[:1], true)
	case "n", "backspace":
		return m.decide(m.pending[:1], false)
	case "a":
		return m.decide(m.pending, true)
	case "esc":
		return m.decide(m.pending, false)
	}
	return nil
}

func (m *model) decide(calls []Call, approve bool) tea.Cmd {
	m.deciding = true
	calls = append([]Call(nil), calls...)
	m.runs.Add(1)
	ctx, backend := m.ctx, m.backend
	return func() tea.Msg {
		defer m.runs.Done()
		for _, c := range calls {
			if err := backend.Decide(ctx, c.ID, approve); err != nil {
				return decidedMsg{id: c.ID, err: err}
			}
		}
		return decidedMsg{}
	}
}

func (m *model) decided(msg decidedMsg) tea.Cmd {
	m.deciding = false
	if msg.err != nil {
		m.addNotice(noticeError, "Could not record the decision: "+msg.err.Error())
	}
	m.pending = m.backend.Pending()
	if len(m.pending) > 0 {
		return nil
	}
	m.setFocus(m.focus)
	return m.start("")
}

func (m *model) tick() tea.Cmd {
	if m.ticking {
		return nil
	}
	m.ticking = true
	return tea.Tick(tickEvery, func(time.Time) tea.Msg { return tickMsg{} })
}

func (m *model) animating() bool {
	return m.running || m.deciding || m.picker != nil
}

// layout sizes every component to the window.
func (m *model) layout() {
	if m.width == 0 {
		return
	}
	mainW := m.width - m.sidebarWidth()
	m.input.SetWidth(max(10, mainW-4))
	m.bar.SetWidth(max(10, mainW-2))
	chatH := m.height - 1 - 1 - statusHeight - (m.input.Height() + 2)
	m.chat.SetWidth(max(10, mainW-2))
	m.chat.SetHeight(max(3, chatH-2))
	for _, b := range m.blocks {
		b.invalidate()
	}
	m.dirty = true
}

func (m *model) sidebarWidth() int {
	if !m.showSidebar || m.width < 90 {
		return 0
	}
	return 34
}

// refresh re-renders the transcript into the chat viewport.
func (m *model) refresh() {
	m.dirty = false
	if m.width == 0 {
		return
	}
	atBottom := m.follow || m.chat.AtBottom()
	m.chat.SetContent(m.renderTranscript(m.chat.Width()))
	if atBottom {
		m.chat.GotoBottom()
	}
}
