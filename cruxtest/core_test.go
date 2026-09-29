package cruxtest_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/apzuk3/crux"
	"github.com/apzuk3/crux/cruxtest"
	"github.com/glebarez/sqlite"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

func newMockAgent(t *testing.T, mock *cruxtest.Mock, model string, opts ...crux.AgentOption) *crux.Agent {
	t.Helper()
	agent, err := crux.New("test-agent", model, append(mock.AgentOptions(), opts...)...)
	require.NoError(t, err)
	return agent
}

func TestToolPanicIsReportedToModel(t *testing.T) {
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "explode", "Always panics", func(ctx context.Context, in struct{}) (string, *crux.StateDelta, error) {
		panic("boom")
	})

	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("explode", map[string]any{})
	mock.Expect().ReturnText("The tool failed.")

	agent := newMockAgent(t, mock, crux.ChatModelGPT5_6Sol, crux.WithToolsRegistry([]string{"explode"}, reg))
	session := crux.MustSession(crux.NewSession(t.Context(), agent))

	out, err := session.Run(t.Context(), "go")
	require.NoError(t, err)
	require.Equal(t, "The tool failed.", out)

	var result *crux.ToolResult
	for _, entry := range session.Logs() {
		if entry.Kind == crux.KindToolResult {
			result = entry.ToolResult
		}
	}
	require.NotNil(t, result)
	require.Contains(t, result.Error, `tool "explode" panicked: boom`)
}

func TestMaxTurnsIsSentinel(t *testing.T) {
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "noop", "Does nothing", func(ctx context.Context, in struct{}) (string, *crux.StateDelta, error) {
		return "ok", nil, nil
	})

	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("noop", map[string]any{})
	mock.Expect().ReturnToolCall("noop", map[string]any{})

	agent := newMockAgent(t, mock, crux.ChatModelGPT5_6Sol,
		crux.WithToolsRegistry([]string{"noop"}, reg), crux.WithMaxTurns(2))
	session := crux.MustSession(crux.NewSession(t.Context(), agent))

	_, err := session.Run(t.Context(), "go")
	require.ErrorIs(t, err, crux.ErrMaxTurns)
}

func TestRefusalIsSentinel(t *testing.T) {
	for _, model := range []string{crux.ChatModelGPT5_6Sol, crux.ClaudeHaiku4_5} {
		t.Run(model, func(t *testing.T) {
			mock := cruxtest.NewMock()
			mock.Expect().ReturnRefusal("no")

			session := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, model)))
			_, err := session.Run(t.Context(), "do something bad")
			require.ErrorIs(t, err, crux.ErrRefused)
		})
	}
}

func TestSessionUsageSumsAllRequests(t *testing.T) {
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "noop", "Does nothing", func(ctx context.Context, in struct{}) (string, *crux.StateDelta, error) {
		return "ok", nil, nil
	})

	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("noop", map[string]any{}).WithUsage(cruxtest.TokenUsage{InputTokens: 10, OutputTokens: 2})
	mock.Expect().ReturnText("done").WithUsage(cruxtest.TokenUsage{InputTokens: 15, OutputTokens: 3})

	agent := newMockAgent(t, mock, crux.ChatModelGPT5_6Sol, crux.WithToolsRegistry([]string{"noop"}, reg))
	session := crux.MustSession(crux.NewSession(t.Context(), agent))
	_, err := session.Run(t.Context(), "go")
	require.NoError(t, err)

	usage := session.Usage()
	require.Equal(t, 25, usage.InputTokens)
	require.Equal(t, 5, usage.OutputTokens)
}

func TestMaxTokensAndTemperatureReachTheWire(t *testing.T) {
	cases := []struct {
		model       string
		maxTokens   func(body map[string]any) any
		temperature func(body map[string]any) any
	}{
		{
			model:       crux.ChatModelGPT5_6Sol,
			maxTokens:   func(b map[string]any) any { return b["max_output_tokens"] },
			temperature: func(b map[string]any) any { return b["temperature"] },
		},
		{
			model:       crux.ClaudeHaiku4_5,
			maxTokens:   func(b map[string]any) any { return b["max_tokens"] },
			temperature: func(b map[string]any) any { return b["temperature"] },
		},
		{
			model: crux.Gemini3_8Flash,
			maxTokens: func(b map[string]any) any {
				return b["generationConfig"].(map[string]any)["maxOutputTokens"]
			},
			temperature: func(b map[string]any) any {
				return b["generationConfig"].(map[string]any)["temperature"]
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			mock := cruxtest.NewMock()
			mock.Expect().ReturnText("hi")

			agent := newMockAgent(t, mock, tc.model, crux.WithMaxTokens(321), crux.WithTemperature(0.25))
			session := crux.MustSession(crux.NewSession(t.Context(), agent))
			_, err := session.Run(t.Context(), "hello")
			require.NoError(t, err)

			var body map[string]any
			require.NoError(t, mock.Requests()[0].UnmarshalBody(&body))
			require.EqualValues(t, 321, tc.maxTokens(body))
			require.EqualValues(t, 0.25, tc.temperature(body))
		})
	}
}

func TestAnthropicDefaultMaxTokens(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnText("hi")

	session := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, crux.ClaudeHaiku4_5)))
	_, err := session.Run(t.Context(), "hello")
	require.NoError(t, err)

	var body map[string]any
	require.NoError(t, mock.Requests()[0].UnmarshalBody(&body))
	require.EqualValues(t, 16384, body["max_tokens"])
}

func TestSubAgentTakesTaskAndReturnsOutput(t *testing.T) {
	type Brief struct {
		Text string `json:"text"`
	}

	child := cruxtest.NewMock()
	child.Expect().ReturnJSON(Brief{Text: "brief"})
	subAgent, err := crux.New("researcher", crux.ChatModelGPT5_6Sol,
		append(child.AgentOptions(), crux.WithOutputSchemaFrom[Brief]())...)
	require.NoError(t, err)

	parent := cruxtest.NewMock()
	parent.Expect().ReturnToolCall("agent_researcher", map[string]any{"task": "research the lantern"})
	parent.Expect().ReturnText("final")
	agent := newMockAgent(t, parent, crux.ChatModelGPT5_6Sol, crux.WithSubAgent(subAgent, "Researches things"))

	session := crux.MustSession(crux.NewSession(t.Context(), agent))
	out, err := session.Run(t.Context(), "go")
	require.NoError(t, err)
	require.Equal(t, "final", out)

	// The parent sees a task parameter, not the child's output schema.
	var parentBody struct {
		Tools []struct {
			Name       string         `json:"name"`
			Parameters map[string]any `json:"parameters"`
		} `json:"tools"`
	}
	require.NoError(t, parent.Requests()[0].UnmarshalBody(&parentBody))
	require.Len(t, parentBody.Tools, 1)
	require.Equal(t, "agent_researcher", parentBody.Tools[0].Name)
	require.Contains(t, parentBody.Tools[0].Parameters["properties"], "task")
	require.NotContains(t, parentBody.Tools[0].Parameters["properties"], "text")

	// The child receives the task as plain text.
	require.Contains(t, child.Requests()[0].BodyString(), `research the lantern`)
	require.NotContains(t, child.Requests()[0].BodyString(), `\"task\"`)

	var result *crux.ToolResult
	for _, entry := range session.Logs() {
		if entry.Kind == crux.KindToolResult {
			result = entry.ToolResult
		}
	}
	require.NotNil(t, result)
	var brief Brief
	require.NoError(t, json.Unmarshal([]byte(result.Output), &brief))
	require.Equal(t, "brief", brief.Text)
}

func TestSubAgentSessionIsStoredAsChild(t *testing.T) {
	db, err := gorm.Open(sqlite.Open("file:"+uuid.NewString()+"?mode=memory&cache=shared"), &gorm.Config{})
	require.NoError(t, err)
	store, err := crux.NewGORMStore(db)
	require.NoError(t, err)

	child := cruxtest.NewMock()
	child.Expect().ReturnText("brief")
	subAgent, err := crux.New("researcher", crux.ChatModelGPT5_6Sol, child.AgentOptions()...)
	require.NoError(t, err)

	parent := cruxtest.NewMock()
	parent.Expect().ReturnToolCall("agent_researcher", map[string]any{"task": "research"})
	parent.Expect().ReturnText("final")
	agent := newMockAgent(t, parent, crux.ChatModelGPT5_6Sol, crux.WithSubAgent(subAgent, "Researches things"))

	session := crux.MustSession(crux.NewSession(t.Context(), agent, crux.WithStore(store)))
	_, err = session.Run(t.Context(), "go")
	require.NoError(t, err)

	var children []struct {
		ID      uuid.UUID
		AgentID uuid.UUID
	}
	require.NoError(t, db.Table("crux_sessions").Where("parent_id = ?", session.ID()).Find(&children).Error)
	require.Len(t, children, 1)
	require.Equal(t, subAgent.ID(), children[0].AgentID)

	// The subagent's conversation is kept in the parent's store.
	entries, err := store.Get(t.Context(), children[0].ID)
	require.NoError(t, err)
	require.Equal(t, "research", entries[0].Text())
	require.Equal(t, "brief", entries[len(entries)-1].Text())
}

func TestMalformedToolArgumentsAreReportedAndStored(t *testing.T) {
	db, err := gorm.Open(sqlite.Open("file:"+uuid.NewString()+"?mode=memory&cache=shared"), &gorm.Config{})
	require.NoError(t, err)
	store, err := crux.NewGORMStore(db)
	require.NoError(t, err)

	reg := crux.NewToolsRegistry()
	called := false
	crux.RegisterToolWithRegistry(reg, "lookup", "Looks up a city", func(ctx context.Context, in struct {
		City string `json:"city"`
	}) (string, *crux.StateDelta, error) {
		called = true
		return in.City, nil, nil
	})

	const truncated = `{"city": "Par`
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("lookup", truncated)
	mock.Expect().ReturnText("Sorry, try again.")
	agent := newMockAgent(t, mock, crux.ChatModelGPT5_6Sol, crux.WithToolsRegistry([]string{"lookup"}, reg))

	session := crux.MustSession(crux.NewSession(t.Context(), agent, crux.WithStore(store)))
	out, err := session.Run(t.Context(), "weather?")
	require.NoError(t, err)
	require.Equal(t, "Sorry, try again.", out)
	require.False(t, called, "a tool must not run with arguments it could not decode")

	var result *crux.ToolResult
	for _, entry := range session.Logs() {
		if entry.Kind == crux.KindToolResult {
			result = entry.ToolResult
		}
	}
	require.NotNil(t, result)
	require.Contains(t, result.Error, "not valid JSON")

	// The model sees its own arguments replayed unchanged.
	var body struct {
		Input []map[string]any `json:"input"`
	}
	require.NoError(t, mock.Requests()[1].UnmarshalBody(&body))
	var replayed string
	for _, item := range body.Input {
		if item["type"] == "function_call" {
			replayed, _ = item["arguments"].(string)
		}
	}
	require.Equal(t, truncated, replayed)

	// The whole conversation was persisted and reloads.
	stored, err := store.Get(t.Context(), session.ID())
	require.NoError(t, err)
	require.Len(t, stored, len(session.Logs()))
}

func TestAnthropicLargeMaxTokensRuns(t *testing.T) {
	// The SDK rejects non-streaming requests whose max_tokens may take over ten minutes.
	mock := cruxtest.NewMock()
	mock.Expect().ReturnText("long answer")

	agent := newMockAgent(t, mock, crux.ClaudeHaiku4_5, crux.WithMaxTokens(64000))
	session := crux.MustSession(crux.NewSession(t.Context(), agent))
	out, err := session.Run(t.Context(), "write a lot")
	require.NoError(t, err)
	require.Equal(t, "long answer", out)

	var body map[string]any
	require.NoError(t, mock.Requests()[0].UnmarshalBody(&body))
	require.EqualValues(t, 64000, body["max_tokens"])
	require.Equal(t, true, body["stream"])
}

type failingStore struct {
	*crux.MemoryStore
	failToolResults bool
}

func (f *failingStore) Append(ctx context.Context, session *crux.Session, entries ...crux.Entry) error {
	for _, entry := range entries {
		if f.failToolResults && entry.Kind == crux.KindToolResult {
			return errors.New("database unavailable")
		}
	}
	return f.MemoryStore.Append(ctx, session, entries...)
}

func TestUnsavedToolResultRunsAgainOnResume(t *testing.T) {
	reg := crux.NewToolsRegistry()
	calls := 0
	crux.RegisterToolWithRegistry(reg, "count", "Counts calls", func(ctx context.Context, in struct{}) (int, *crux.StateDelta, error) {
		calls++
		return calls, nil, nil
	})

	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("count", map[string]any{})
	mock.Expect().ReturnText("done")
	agent := newMockAgent(t, mock, crux.ChatModelGPT5_6Sol, crux.WithToolsRegistry([]string{"count"}, reg))

	store := &failingStore{MemoryStore: crux.NewMemoryStore(), failToolResults: true}
	session := crux.MustSession(crux.NewSession(t.Context(), agent, crux.WithStore(store)))

	_, err := session.Run(t.Context(), "go")
	require.ErrorContains(t, err, "database unavailable")
	for _, entry := range session.Logs() {
		require.NotEqual(t, crux.KindToolResult, entry.Kind, "the unsaved result must not be in the session")
	}

	store.failToolResults = false
	out, err := session.Resume(t.Context())
	require.NoError(t, err)
	require.Equal(t, "done", out)
	require.Equal(t, 2, calls, "the tool runs again because its first result was never stored")

	stored, err := store.Get(t.Context(), session.ID())
	require.NoError(t, err)
	require.Equal(t, session.Logs(), stored)
}
