package tui

import (
	"context"
	"strings"
	"sync"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
	"github.com/stretchr/testify/require"
)

type fakeBackend struct {
	mu       sync.Mutex
	pending  []Call
	decided  map[string]bool
	inputs   []string
	switched []string
}

func (f *fakeBackend) Run(ctx context.Context, input string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.inputs = append(f.inputs, input)
	return nil
}

func (f *fakeBackend) Compact(context.Context) (bool, error) { return true, nil }

func (f *fakeBackend) Pending() []Call {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []Call
	for _, c := range f.pending {
		if _, ok := f.decided[c.ID]; !ok {
			out = append(out, c)
		}
	}
	return out
}

func (f *fakeBackend) Decide(ctx context.Context, id string, approve bool) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.decided == nil {
		f.decided = map[string]bool{}
	}
	f.decided[id] = approve
	return nil
}

func (f *fakeBackend) SwitchModel(ctx context.Context, provider, model string) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.switched = append(f.switched, provider+"/"+model)
	return "7f3c2a10-0000-4000-8000-000000000000", nil
}

var testInfo = Info{
	Agent:    "support",
	Provider: "anthropic",
	Model:    "claude-sonnet-5-5",
	MaxTurns: 10,
	Models: []Model{
		{Provider: "anthropic", Name: "claude-sonnet-5-5"},
		{Provider: "anthropic", Name: "claude-opus-5-5"},
		{Provider: "openai", Name: "gpt-5.4-mini"},
	},
	Tools: []Tool{
		{Name: "refund_order", Description: "Refund an order", Approval: true},
		{Name: "agent_orders", Description: "Looks up orders", IsSubAgent: true, SubAgent: []Tool{
			{Name: "lookup_order", Description: "Look up an order"},
		}},
	},
}

func newTestModel(t *testing.T, info Info, b Backend) *model {
	t.Helper()
	m := newModel(info, b, context.Background(), &sync.WaitGroup{})
	m.Update(tea.WindowSizeMsg{Width: 140, Height: 44})
	return m
}

func screen(m *model) string {
	return ansi.Strip(m.render())
}

func TestModelShowsRunLifecycle(t *testing.T) {
	m := newTestModel(t, testInfo, &fakeBackend{})
	require.Contains(t, screen(m), "is ready")
	require.Len(t, strings.Split(m.render(), "\n"), 44)

	m.running = true
	m.run = runState{stage: stageRequest}
	m.blocks = append(m.blocks, &block{kind: blockUser, text: "Check A-1001"})
	events := []Event{
		{Kind: EventRunStarted, Agent: "support"},
		{Kind: EventTurnStarted, Agent: "support"},
		{Kind: EventToolCall, Agent: "support", Call: Call{ID: "c1", Name: "agent_orders", Args: `{"task":"look up A-1001"}`}},
		{Kind: EventToolStarted, Agent: "support", Call: Call{ID: "c1"}},
		{Kind: EventTurnStarted, Agent: "orders", Parent: "c1"},
		{Kind: EventToolCall, Agent: "orders", Parent: "c1", Call: Call{ID: "s1", Name: "lookup_order", Args: `{"order_id":"A-1001"}`}},
		{Kind: EventToolStarted, Agent: "orders", Parent: "c1", Call: Call{ID: "s1"}},
	}
	for _, e := range events {
		m.Update(eventMsg(e))
	}
	require.Equal(t, stageTools, m.run.stage)
	require.Equal(t, 1, m.stats["lookup_order"].running)
	out := screen(m)
	require.Contains(t, out, "◇ orders")
	require.Contains(t, out, "lookup_order  order_id=A-1001")
	require.Contains(t, out, "tools 0/1")

	for _, e := range []Event{
		{Kind: EventToolResult, Agent: "orders", Parent: "c1", Call: Call{ID: "s1"}, Output: `{"status":"delivered"}`},
		{Kind: EventToolResult, Agent: "support", Call: Call{ID: "c1"}, Output: "A-1001 was delivered"},
		{Kind: EventTurnStarted, Agent: "support"},
		{Kind: EventText, Agent: "support", Text: "It was "},
		{Kind: EventText, Agent: "support", Text: "delivered."},
	} {
		m.Update(eventMsg(e))
	}
	require.Equal(t, stageWriting, m.run.stage)
	require.Equal(t, 2, m.run.turn)
	require.Contains(t, screen(m), "It was delivered.")

	m.Update(eventMsg{Kind: EventAssistant, Agent: "support", Text: "It was **delivered**.", Usage: &Usage{In: 1500, Out: 40}})
	m.Update(eventMsg{Kind: EventRunFinished, Agent: "support", Outcome: "answered"})
	m.Update(runDoneMsg{})

	require.False(t, m.running)
	require.Equal(t, stageDone, m.run.stage)
	require.Equal(t, 1, m.stats["lookup_order"].calls)
	require.Equal(t, Usage{In: 1500, Out: 40}, m.usage)
	out = screen(m)
	require.Contains(t, out, "✓ answered")
	require.Contains(t, out, "It was delivered.")
	require.Contains(t, out, "↑1.5k ↓40")
	require.Equal(t, 1, strings.Count(out, "It was delivered."), "the streamed preview is replaced by the stored answer")
}

func TestModelApprovalFlow(t *testing.T) {
	backend := &fakeBackend{pending: []Call{{ID: "r1", Name: "refund_order", Args: `{"order_id":"A-1002"}`, Agent: "support"}}}
	m := newTestModel(t, testInfo, backend)
	m.running = true
	m.Update(eventMsg{Kind: EventToolCall, Agent: "support", Call: Call{ID: "r1", Name: "refund_order", Args: `{"order_id":"A-1002"}`}})
	m.Update(runDoneMsg{})

	require.Len(t, m.pending, 1)
	require.Equal(t, toolAwaiting, m.calls["r1"].state)
	out := screen(m)
	require.Contains(t, out, "Approval needed")
	require.Contains(t, out, `"order_id": "A-1002"`)
	require.Contains(t, out, "Refund an order")

	_, cmd := m.Update(tea.KeyPressMsg{Code: 'y', Text: "y"})
	require.NotNil(t, cmd)
	msg := cmd()
	require.Equal(t, map[string]bool{"r1": true}, backend.decided)

	m.Update(msg)
	require.Empty(t, m.pending)
	require.True(t, m.running, "deciding the last call resumes the run")
}

func TestModelQueuesPromptsWhileRunning(t *testing.T) {
	m := newTestModel(t, testInfo, &fakeBackend{})
	m.running = true
	m.input.SetValue("and A-1003?")
	m.Update(tea.KeyPressMsg{Code: tea.KeyEnter})
	require.Equal(t, []string{"and A-1003?"}, m.queue)
	require.Contains(t, screen(m), "1 queued")

	m.Update(runDoneMsg{})
	require.Empty(t, m.queue)
	require.True(t, m.running, "the queued prompt is sent when the run ends")
}

func TestModelReplaysHistory(t *testing.T) {
	info := testInfo
	info.History = []Event{
		{Kind: EventUser, Agent: "support", Text: "Where is A-1001?"},
		{Kind: EventToolCall, Agent: "support", Call: Call{ID: "c1", Name: "refund_order", Args: `{"order_id":"A-1001"}`}},
		{Kind: EventToolResult, Agent: "support", Call: Call{ID: "c1"}, Denied: true, Err: "tool execution declined by user"},
		{Kind: EventAssistant, Agent: "support", Text: "I did not refund it."},
	}
	m := newTestModel(t, info, &fakeBackend{})
	out := screen(m)
	require.Contains(t, out, "Where is A-1001?")
	require.Contains(t, out, "rejected")
	require.Contains(t, out, "I did not refund it.")
	require.Equal(t, stageIdle, m.run.stage)
}

func TestModelNarrowTerminal(t *testing.T) {
	m := newTestModel(t, testInfo, &fakeBackend{})
	m.Update(tea.WindowSizeMsg{Width: 70, Height: 30})
	require.Zero(t, m.sidebarWidth())
	lines := strings.Split(m.render(), "\n")
	require.Len(t, lines, 30)
	for _, line := range lines {
		require.LessOrEqual(t, ansi.StringWidth(line), 70)
	}
	m.Update(tea.WindowSizeMsg{Width: 30, Height: 10})
	require.Contains(t, screen(m), "bigger")
}

func TestModelSwitchesModels(t *testing.T) {
	backend := &fakeBackend{}
	m := newTestModel(t, testInfo, backend)
	m.Update(tea.KeyPressMsg{Code: 'o', Mod: tea.ModCtrl})
	require.NotNil(t, m.picker)
	out := screen(m)
	require.Contains(t, out, "Switch model")
	require.Contains(t, out, "gpt-5.4-mini")

	for _, r := range "mini" {
		m.Update(tea.KeyPressMsg{Code: r, Text: string(r)})
	}
	require.Equal(t, []Model{{Provider: "openai", Name: "gpt-5.4-mini"}}, m.pickerMatches())

	_, cmd := m.Update(tea.KeyPressMsg{Code: tea.KeyEnter})
	require.NotNil(t, cmd)
	msg := cmd()
	if batch, ok := msg.(tea.BatchMsg); ok {
		for _, c := range batch {
			if got, ok := c().(switchedMsg); ok {
				msg = got
			}
		}
	}
	require.IsType(t, switchedMsg{}, msg)
	m.Update(msg)

	require.Nil(t, m.picker)
	require.Equal(t, []string{"openai/gpt-5.4-mini"}, backend.switched)
	require.Equal(t, Model{Provider: "openai", Name: "gpt-5.4-mini"}, m.current())
	out = screen(m)
	require.Contains(t, out, "Switched from claude-sonnet-5-5 to openai/gpt-5.4-mini")
	require.Contains(t, out, "openai/gpt-5.4-mini")
}

func TestModelPickerWaitsForRun(t *testing.T) {
	m := newTestModel(t, testInfo, &fakeBackend{})
	m.running = true
	m.input.SetValue("/model opus")
	m.Update(tea.KeyPressMsg{Code: tea.KeyEnter})
	require.Nil(t, m.picker)
	require.Contains(t, screen(m), "finish the current run")

	m.running = false
	m.input.SetValue("/model opus")
	m.Update(tea.KeyPressMsg{Code: tea.KeyEnter})
	require.NotNil(t, m.picker)
	require.Equal(t, []Model{{Provider: "anthropic", Name: "claude-opus-5-5"}}, m.pickerMatches())
}
