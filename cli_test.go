package crux

import (
	"context"
	"testing"
	"time"

	"crux.foo/internal/tui"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

func TestCLIToolsDescribeSubagents(t *testing.T) {
	reg := NewToolsRegistry()
	noop := func(ctx context.Context, in struct{}) (string, *StateDelta, error) { return "", nil, nil }
	RegisterToolWithRegistry(reg, "lookup", "Looks up an order", noop)
	RegisterToolWithRegistry(reg, "refund", "Refunds an order", noop, WithApprovalNeeded(true))
	orders := Must(New("orders", OpenAIGPT4, WithToolsRegistry([]string{"lookup"}, reg)))
	support := Must(New("support", OpenAIGPT4, WithToolsRegistry([]string{"refund"}, reg), WithSubAgent(orders, "Answers order questions")))

	require.Equal(t, []tui.Tool{
		{Name: "refund", Description: "Refunds an order", Approval: true},
		{Name: "agent_orders", Description: "Answers order questions", IsSubAgent: true, SubAgent: []tui.Tool{
			{Name: "lookup", Description: "Looks up an order"},
		}},
	}, cliTools(support.tools))
}

func TestCLIEventsNestSubagentWork(t *testing.T) {
	reg := NewToolsRegistry()
	RegisterToolWithRegistry(reg, "lookup", "", func(ctx context.Context, in struct{}) (string, *StateDelta, error) { return "", nil, nil })
	orders := Must(New("orders", OpenAIGPT4, WithToolsRegistry([]string{"lookup"}, reg)))
	support := Must(New("support", OpenAIGPT4, WithSubAgent(orders, "orders")))

	ctx := context.Background()
	c := &cli{children: map[uuid.UUID]string{}}
	parent := MustSession(NewSession(ctx, support))
	c.session = parent

	call := Entry{Seq: 3, Kind: KindToolCall, ToolCall: &ToolCall{ID: "c1", Name: "agent_orders", Args: []byte(`{"task":"find A-1"}`)}}
	c.track(parent, call)
	ev, ok := c.event(parent, call)
	require.True(t, ok)
	require.Equal(t, tui.Event{Kind: tui.EventToolCall, Agent: "support", Call: tui.Call{ID: "c1", Name: "agent_orders", Args: `{"task":"find A-1"}`, Agent: "support"}}, ev)

	childID := childSessionID(parent.id, openToolCall{call: call.ToolCall, seq: call.Seq}.key())
	child := MustSession(NewSession(ctx, orders, WithSessionID(childID)))
	ev, ok = c.event(child, Entry{Kind: KindToolResult, Duration: time.Second, ToolResult: &ToolResult{CallID: "s1", Output: "ok"}})
	require.True(t, ok)
	require.Equal(t, tui.Event{Kind: tui.EventToolResult, Agent: "orders", Parent: "c1", Call: tui.Call{ID: "s1"}, Output: "ok", Duration: time.Second}, ev)

	ev, ok = c.event(parent, Entry{Kind: KindRunFinished, Run: &RunStatus{Outcome: RunMaxTurns, Error: "max turns reached"}})
	require.True(t, ok)
	require.Equal(t, "max_turns", ev.Outcome)

	ev, ok = c.event(parent, Entry{Kind: KindAssistant, Content: []ContentPart{{Kind: ContentKindRefusal, Text: "no"}}})
	require.True(t, ok)
	require.Equal(t, tui.EventRefusal, ev.Kind)

	_, ok = c.event(parent, Entry{Kind: KindStateDelta, Delta: &StateDelta{}})
	require.False(t, ok, "state deltas are not shown")
	ev, ok = c.event(parent, Entry{Kind: KindProviderTool, Usage: &Usage{InputTokens: 5}})
	require.True(t, ok, "usage is counted even on entries the chat does not show")
	require.Equal(t, &tui.Usage{In: 5}, ev.Usage)
}

func TestCLISwitchModelKeepsHistoryAndHandlers(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test-key")
	ctx := context.Background()
	var seen []Kind
	agent := Must(New("support", OpenAIGPT4, WithAPIKey("openai-key")))
	user, _ := NewUserEntry("hello")
	answer := Entry{Kind: KindAssistant, Content: []ContentPart{{Kind: ContentKindText, Text: "hi"}}}
	session := MustSession(NewSession(ctx, agent,
		WithSessionLogs([]Entry{user, answer}),
		WithEntryHandler(func(_ context.Context, _ *Session, e Entry) { seen = append(seen, e.Kind) }),
	))
	c := &cli{session: session, children: map[uuid.UUID]string{}}

	id, err := c.SwitchModel(ctx, string(ProviderAnthropic), string(ClaudeSonnet5_5))
	require.NoError(t, err)
	require.NotEqual(t, session.id.String(), id)
	require.Equal(t, id, c.session.id.String())
	require.Equal(t, ProviderAnthropic, c.session.agent.provider)
	require.Equal(t, string(ClaudeSonnet5_5), c.session.agent.model)
	require.Equal(t, "hi", c.session.logs[1].Text())

	require.NoError(t, c.session.appendLogs(ctx, Entry{Kind: KindRunStarted}))
	require.Equal(t, []Kind{KindRunStarted}, seen, "the fork keeps the session's entry handlers")
}

func TestCLIModelsListCurrentFirst(t *testing.T) {
	for _, spec := range providerSpecs {
		for _, name := range spec.envVars {
			t.Setenv(name, "")
		}
	}
	t.Setenv("ANTHROPIC_API_KEY", "test-key")
	agent := Must(New("support", OpenAIGPT5_4, WithAPIKey("openai-key")))

	models := cliModels(agent)
	require.Equal(t, tui.Model{Provider: "openai", Name: string(OpenAIGPT5_4)}, models[0])
	providers := map[string]bool{}
	for _, m := range models {
		providers[m.Provider] = true
		require.False(t, snapshot.MatchString(m.Name), "dated snapshot %s is listed", m.Name)
	}
	require.Equal(t, map[string]bool{"openai": true, "anthropic": true}, providers)
	require.Contains(t, models, tui.Model{Provider: "anthropic", Name: string(ClaudeOpus5_5)})
}
