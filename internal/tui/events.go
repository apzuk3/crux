package tui

import (
	"time"

	tea "charm.land/bubbletea/v2"
)

// apply folds an event into the transcript, the sidebar and the run's
// lifecycle. replay is set for history, which shows no lifecycle.
func (m *model) apply(e Event, replay bool) tea.Cmd {
	m.dirty = true
	if e.Usage != nil {
		m.usage.In += e.Usage.In
		m.usage.Out += e.Usage.Out
		m.usage.CacheRead += e.Usage.CacheRead
	}
	if e.top() && e.FirstToken > 0 {
		m.run.ttft = e.FirstToken
	}

	switch e.Kind {
	case EventRunStarted, EventRunFinished:
		return m.applyRun(e)
	case EventTurnStarted:
		return m.applyTurn(e)
	case EventUser, EventCompacted, EventRefusal:
		m.applyNotice(e, replay)
	case EventText, EventReasoningText:
		return m.applyDelta(e)
	case EventReasoning:
		m.applyReasoning(e)
	case EventAssistant:
		m.applyAssistant(e)
	case EventToolCall:
		return m.applyToolCall(e)
	case EventToolStarted:
		return m.applyToolStarted(e)
	case EventToolResult:
		return m.applyToolResult(e, replay)
	case EventApproval:
		m.applyApproval(e)
	}
	return nil
}

// top reports whether the event comes from the top-level session rather
// than a subagent's.
func (e Event) top() bool { return e.Parent == "" }

func (m *model) applyRun(e Event) tea.Cmd {
	if !e.top() {
		return nil
	}
	if e.Kind == EventRunStarted {
		m.runCount++
		return nil
	}
	m.run.outcome = e.Outcome
	m.run.elapsed = time.Since(m.run.started)
	m.run.stage = finishedStage(e.Outcome)
	m.barColors.set(m.th, m.run.stage, m.frame)
	switch m.run.stage {
	case stageDone, stageFailed:
		return m.bar.SetPercent(1)
	}
	return nil
}

func finishedStage(outcome string) stage {
	switch outcome {
	case "answered":
		return stageDone
	case "approval_needed":
		return stageApproval
	case "cancelled":
		return stageCancelled
	}
	return stageFailed
}

func (m *model) applyTurn(e Event) tea.Cmd {
	m.turns++
	if !e.top() {
		if parent := m.calls[e.Parent]; parent != nil {
			parent.turns++
			parent.activity = "thinking"
			parent.invalidate()
		}
		return nil
	}
	m.run.turn++
	m.run.toolsTotal, m.run.toolsDone = 0, 0
	return m.setStage(stageRequest)
}

// applyNotice shows the top-level events that become a block of their own.
func (m *model) applyNotice(e Event, replay bool) {
	if !e.top() {
		return
	}
	switch e.Kind {
	case EventUser:
		// Live prompts are shown when sent; a subagent's task shows on its card.
		if replay {
			m.blocks = append(m.blocks, &block{kind: blockUser, text: e.Text})
		}
	case EventCompacted:
		m.addNotice(noticeInfo, e.Text)
	case EventRefusal:
		m.finishLive()
		m.addNotice(noticeRefusal, e.Text)
	}
}

// applyDelta appends streamed text to the live block of its kind.
func (m *model) applyDelta(e Event) tea.Cmd {
	kind, s := blockAssistant, stageWriting
	if e.Kind == EventReasoningText {
		kind, s = blockReasoning, stageThinking
	}
	b := m.live(kind, e.Agent)
	b.text += e.Text
	b.invalidate()
	return m.setStage(s)
}

func (m *model) applyReasoning(e Event) {
	if !e.top() {
		return
	}
	if b := m.liveBlock(blockReasoning); b != nil {
		if e.Text != "" {
			b.text = e.Text
		}
		b.finish()
		return
	}
	if e.Text != "" {
		m.blocks = append(m.blocks, &block{kind: blockReasoning, text: e.Text})
	}
}

func (m *model) applyAssistant(e Event) {
	if !e.top() {
		if parent := m.calls[e.Parent]; parent != nil && e.Text != "" {
			parent.activity = "answering"
			parent.invalidate()
		}
		return
	}
	m.finishReasoning()
	if b := m.liveBlock(blockAssistant); b != nil {
		b.text = e.Text
		b.finish()
		return
	}
	if e.Text != "" {
		m.blocks = append(m.blocks, &block{kind: blockAssistant, agent: e.Agent, text: e.Text})
	}
}

func (m *model) applyToolCall(e Event) tea.Cmd {
	m.finishLive()
	parent := m.calls[e.Parent]
	b := &block{kind: blockTool, call: e.Call, state: toolQueued, agent: e.Agent}
	if t, ok := m.tools[e.Call.Name]; ok {
		b.subagent = t.IsSubAgent
	}
	m.calls[e.Call.ID] = b
	if parent != nil {
		parent.children = append(parent.children, b)
		parent.activity = "calling " + e.Call.Name
		parent.invalidate()
	} else {
		m.blocks = append(m.blocks, b)
	}
	if e.top() {
		m.run.toolsTotal++
		return m.setStage(stageTools)
	}
	return nil
}

func (m *model) applyToolStarted(e Event) tea.Cmd {
	m.toolCalls++
	b := m.calls[e.Call.ID]
	if b == nil {
		return nil
	}
	b.state = toolRunning
	b.started = time.Now()
	b.invalidate()
	m.stat(b.call.Name).running++
	if parent := m.calls[e.Parent]; parent != nil {
		parent.activity = "running " + b.call.Name
		parent.invalidate()
	}
	if e.top() {
		return m.setStage(stageTools)
	}
	return nil
}

func (m *model) applyToolResult(e Event, replay bool) tea.Cmd {
	b := m.calls[e.Call.ID]
	if b == nil {
		return nil
	}
	m.recordToolStat(b, e, replay)
	b.output, b.err, b.dur = e.Output, e.Err, e.Duration
	b.activity = ""
	b.state = resultState(e)
	b.invalidate()
	if e.top() {
		m.run.toolsDone++
		return m.setStage(stageTools)
	}
	return nil
}

// recordToolStat updates the sidebar's counters for a finished call. Replayed
// results only release the running count.
func (m *model) recordToolStat(b *block, e Event, replay bool) {
	if b.state == toolRunning {
		st := m.stat(b.call.Name)
		st.running = max(0, st.running-1)
	}
	if replay {
		return
	}
	st := m.stat(b.call.Name)
	st.calls++
	st.last = e.Duration
	if e.Err != "" && !e.Denied {
		st.failed++
	}
}

func resultState(e Event) toolState {
	switch {
	case e.Denied:
		return toolDenied
	case e.Err != "":
		return toolFailed
	}
	return toolOK
}

func (m *model) applyApproval(e Event) {
	b := m.calls[e.Call.ID]
	if b == nil {
		return
	}
	if e.Approved {
		b.state = toolQueued
	} else {
		b.state = toolDenied
	}
	b.invalidate()
}

// setStage moves the top-level run to s and advances the progress bar.
func (m *model) setStage(s stage) tea.Cmd {
	if !m.running {
		return nil
	}
	m.run.stage = s
	m.barColors.set(m.th, s, m.frame)
	return m.bar.SetPercent(m.run.progress())
}

// progress is how far through the current turn the run is.
func (r runState) progress() float64 {
	switch r.stage {
	case stageRequest:
		return 0.15
	case stageThinking:
		return 0.35
	case stageTools:
		if r.toolsTotal == 0 {
			return 0.45
		}
		return 0.45 + 0.4*float64(r.toolsDone)/float64(r.toolsTotal)
	case stageWriting:
		return 0.9
	case stageApproval:
		return 0.6
	case stageDone, stageFailed:
		return 1
	}
	return 0
}

func (m *model) stat(name string) *toolStat {
	st := m.stats[name]
	if st == nil {
		st = &toolStat{}
		m.stats[name] = st
	}
	return st
}

// live returns the streaming block of kind, starting one if needed.
func (m *model) live(kind blockKind, agent string) *block {
	if b := m.liveBlock(kind); b != nil {
		return b
	}
	if kind == blockAssistant {
		m.finishReasoning()
	}
	b := &block{kind: kind, agent: agent, live: true, at: time.Now()}
	m.blocks = append(m.blocks, b)
	return b
}

// liveBlock returns the last block if it is still streaming and of kind.
func (m *model) liveBlock(kind blockKind) *block {
	for i := len(m.blocks) - 1; i >= 0; i-- {
		b := m.blocks[i]
		if b.live && b.kind == kind {
			return b
		}
		if b.kind != blockReasoning {
			break
		}
	}
	return nil
}

func (m *model) finishReasoning() {
	if b := m.liveBlock(blockReasoning); b != nil {
		b.finish()
	}
}

// finishLive ends any streaming block, keeping what streamed so far.
func (m *model) finishLive() {
	for _, b := range m.blocks {
		if b.live {
			b.finish()
		}
	}
}

func (m *model) addNotice(kind noticeKind, text string) {
	m.blocks = append(m.blocks, &block{kind: blockNotice, notice: kind, text: text})
	m.dirty = true
}
