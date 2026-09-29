package crux

import (
	"context"
	"encoding/json"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/glebarez/sqlite"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

func TestStoresRejectEntriesAnotherWriterAppended(t *testing.T) {
	ctx := t.Context()
	db, err := gorm.Open(sqlite.Open("file:"+uuid.NewString()+"?mode=memory&cache=shared"), &gorm.Config{})
	require.NoError(t, err)
	gormStore, err := NewGORMStore(db)
	require.NoError(t, err)

	for name, store := range map[string]Store{"memory": NewMemoryStore(), "gorm": gormStore} {
		t.Run(name, func(t *testing.T) {
			agent := Must(New("a", ChatModelGPT5, WithAPIKey("k")))
			first := MustSession(NewSession(ctx, agent, WithStore(store)))
			hello, _ := NewUserEntry("hello")
			require.NoError(t, first.appendLogs(ctx, hello))

			// Two handlers load the same conversation; the second write loses.
			a := MustSession(NewSession(ctx, agent, WithStore(store), WithSessionID(first.ID())))
			b := MustSession(NewSession(ctx, agent, WithStore(store), WithSessionID(first.ID())))
			fromA, _ := NewUserEntry("from a")
			fromB, _ := NewUserEntry("from b")
			require.NoError(t, a.appendLogs(ctx, fromA))
			err := b.appendLogs(ctx, fromB)
			require.ErrorIs(t, err, ErrSessionConflict)
			require.Len(t, b.Logs(), 1, "a failed write leaves the session unchanged")

			stored, err := store.Get(ctx, first.ID())
			require.NoError(t, err)
			require.Len(t, stored, 2)
			require.Equal(t, "from a", stored[1].Text())
		})
	}
}

func TestAppendContinuesAfterSeededSeq(t *testing.T) {
	ctx := t.Context()
	agent := Must(New("a", ChatModelGPT5, WithAPIKey("k")))
	seeded := []Entry{
		{Seq: 5, Kind: KindUser, Content: []ContentPart{{Kind: ContentKindText, Text: "hi"}}},
		{Seq: 9, Kind: KindAssistant, Content: []ContentPart{{Kind: ContentKindText, Text: "hello"}}},
	}
	session := MustSession(NewSession(ctx, agent, WithSessionLogs(seeded)))
	next, _ := NewUserEntry("again")
	require.NoError(t, session.appendLogs(ctx, next))
	require.Equal(t, uint64(10), session.Logs()[2].Seq)
}

func TestToolRegistrationPanicsOnInvalidOrDuplicateNames(t *testing.T) {
	noop := func(ctx context.Context, in struct{}) (string, *StateDelta, error) { return "", nil, nil }

	reg := NewToolsRegistry()
	RegisterToolWithRegistry(reg, "ok_tool-1", "fine", noop)
	require.Panics(t, func() { RegisterToolWithRegistry(reg, "ok_tool-1", "again", noop) })
	require.Panics(t, func() { RegisterToolWithRegistry(reg, "has space", "bad", noop) })
	require.Panics(t, func() { RegisterToolWithRegistry(reg, "", "bad", noop) })
	require.Panics(t, func() { RegisterToolWithRegistry(reg, strings.Repeat("a", 65), "bad", noop) })

	err := AddToolsetWithRegistry(reg, toolsetFunc(func(r ToolsRegistry) error {
		RegisterToolWithRegistry(r, "ok_tool-1", "clash", noop)
		return nil
	}))
	require.ErrorContains(t, err, `tool "ok_tool-1" is already registered`)
}

type toolsetFunc func(ToolsRegistry) error

func (f toolsetFunc) Register(r ToolsRegistry) error { return f(r) }

func TestAgentRejectsInvalidToolSetup(t *testing.T) {
	sub := Must(New("my helper", ChatModelGPT5, WithAPIKey("k")))
	_, err := New("p", ChatModelGPT5, WithAPIKey("k"), WithSubAgent(sub, "d"))
	require.ErrorContains(t, err, "invalid tool name")

	helper := Must(New("helper", ChatModelGPT5, WithAPIKey("k")))
	_, err = New("p", ChatModelGPT5, WithAPIKey("k"), WithSubAgent(helper, "d"), WithSubAgent(helper, "d"))
	require.ErrorContains(t, err, `two tools named "agent_helper"`)

	_, err = New("p", ChatModelGPT5, WithAPIKey("k"), WithMaxTurns(0))
	require.Error(t, err)
	_, err = New("p", ChatModelGPT5, WithAPIKey("k"), WithMaxRepairs(-1))
	require.Error(t, err)
}

func TestToolArgumentsAreChecked(t *testing.T) {
	type nested struct {
		Name string `json:"name"`
	}
	type input struct {
		Path  string  `json:"path"`
		Note  *string `json:"note"`
		Inner nested  `json:"inner"`
	}
	reg := NewToolsRegistry()
	var got input
	RegisterToolWithRegistry(reg, "write", "w", func(ctx context.Context, in input) (string, *StateDelta, error) {
		got = in
		return "ok", nil, nil
	})
	tools, err := reg.selected([]string{"write"})
	require.NoError(t, err)
	tool := tools[0]
	call := func(args string) error {
		_, _, err := tool.invoke(t.Context(), json.RawMessage(args))
		return err
	}

	require.NoError(t, call(`{"path":"a.txt","note":null,"inner":{"name":"x"}}`))
	require.Equal(t, "a.txt", got.Path)

	require.ErrorContains(t, call(`{"path":"safe.txt","path":"evil.txt","inner":{"name":"x"}}`), "more than once")
	require.ErrorContains(t, call(`{"path":"safe.txt","PATH":"evil.txt","inner":{"name":"x"}}`), `"PATH" must be spelled "path"`)
	require.ErrorContains(t, call(`{"path":"a","inner":{"NAME":"x"}}`), `"inner.NAME" must be spelled "inner.name"`)
	require.ErrorContains(t, call(`{"inner":{"name":"x"}}`), "path")
	require.ErrorContains(t, call(`{"path":"a","inner":{}}`), "name")
	require.ErrorContains(t, call(`{"path":1,"inner":{"name":"x"}}`), "path")

	var mapIn map[string]any
	RegisterToolWithRegistry(reg, "anything", "a", func(ctx context.Context, in map[string]any) (string, *StateDelta, error) {
		mapIn = in
		return "ok", nil, nil
	})
	tools, err = reg.selected([]string{"anything"})
	require.NoError(t, err)
	_, _, err = tools[0].invoke(t.Context(), json.RawMessage(`{"Key":1,"key":2}`))
	require.NoError(t, err, "maps may hold keys that differ only in case")
	require.Len(t, mapIn, 2)
}

func TestToolCallsFromOneTurnRunConcurrently(t *testing.T) {
	var running, peak atomic.Int32
	slow := Tool{
		name: "slow",
		kind: toolKindTool,
		invoke: func(ctx context.Context, args json.RawMessage) (string, *StateDelta, error) {
			n := running.Add(1)
			for {
				p := peak.Load()
				if n <= p || peak.CompareAndSwap(p, n) {
					break
				}
			}
			time.Sleep(50 * time.Millisecond)
			running.Add(-1)
			return "done", nil, nil
		},
	}
	agent := &Agent{tools: []Tool{slow}}
	var logs []Entry
	for _, id := range []string{"c1", "c2", "c3"} {
		logs = append(logs, Entry{Kind: KindToolCall, ToolCall: &ToolCall{ID: id, Name: "slow"}})
	}
	session := MustSession(NewSession(t.Context(), agent, WithSessionLogs(logs)))

	results := session.executeUnexecutedToolCalls(t.Context())
	require.Len(t, results, 3)
	for i, id := range []string{"c1", "c2", "c3"} {
		require.Equal(t, id, results[i].ToolResult.CallID)
	}
	require.Equal(t, int32(3), peak.Load())
}

func TestZeroToolsRegistryPanicsWithAClearMessage(t *testing.T) {
	var reg ToolsRegistry
	require.PanicsWithValue(t, "crux: ToolsRegistry must be created with NewToolsRegistry", func() {
		RegisterToolWithRegistry(reg, "x", "x", func(ctx context.Context, in struct{}) (string, *StateDelta, error) { return "", nil, nil })
	})
}

func TestToolRegistrationRejectsNonObjectInput(t *testing.T) {
	reg := NewToolsRegistry()
	require.PanicsWithError(t, `tool "echo": input type string must be a struct or a map`, func() {
		RegisterToolWithRegistry(reg, "echo", "e", func(ctx context.Context, in string) (string, *StateDelta, error) { return in, nil, nil })
	})
	require.NotPanics(t, func() {
		RegisterToolWithRegistry(reg, "any_map", "m", func(ctx context.Context, in map[string]any) (string, *StateDelta, error) { return "", nil, nil })
	})
}

func TestCanonicalDataRedactsBaseURLCredentials(t *testing.T) {
	agent, err := New("a", "some-model", WithProvider(ProviderOpenAI), WithAPIKey("k"),
		WithBaseURL("https://user:secret@gateway.example/v1?key=token123&region=eu"))
	require.NoError(t, err)
	data := string(agent.CanonicalData())
	require.NotContains(t, data, "secret")
	require.NotContains(t, data, "token123")
	require.Contains(t, data, "gateway.example/v1")
	require.Equal(t, "https://user:secret@gateway.example/v1?key=token123&region=eu", agent.BaseURL())
}

func TestReplaySkipsEmptyText(t *testing.T) {
	log := []Entry{
		{Kind: KindUser, Content: []ContentPart{{Kind: ContentKindText, Text: "hi"}}},
		{Kind: KindAssistant, Content: []ContentPart{{Kind: ContentKindText, Text: " "}}},
		{Kind: KindToolCall, ToolCall: &ToolCall{ID: "c1", Name: "t", Args: []byte(`{}`)}},
		{Kind: KindToolResult, ToolResult: &ToolResult{CallID: "c1", Output: "ok"}},
	}

	messages, err := toAnthropicMessages(log)
	require.NoError(t, err)
	require.Len(t, messages, 3)
	require.Len(t, messages[1].Content, 1)
	require.NotNil(t, messages[1].Content[0].OfToolUse)

	contents, err := toGeminiContents(log)
	require.NoError(t, err)
	require.Len(t, contents, 3)
	require.Len(t, contents[1].Parts, 1)
	call := contents[1].Parts[0]
	require.NotNil(t, call.FunctionCall)
	require.NotEmpty(t, call.ThoughtSignature, "unsigned calls need Gemini 3's stand-in signature")
}
