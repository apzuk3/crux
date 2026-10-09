package cruxtest_test

import (
	"context"
	"encoding/json"
	"slices"
	"testing"

	"crux.foo"
	"crux.foo/cruxtest"
	"github.com/stretchr/testify/require"
)

var searchSchema = json.RawMessage(`{
	"type": "object",
	"properties": {"query": {"type": "string", "description": "What to look for"}, "limit": {"type": "integer"}},
	"required": ["query"],
	"additionalProperties": false
}`)

func TestWithInputSchema(t *testing.T) {
	reg := crux.NewToolsRegistry()
	var got []json.RawMessage
	crux.RegisterToolWithRegistry(reg, "search", "Search the docs", func(ctx context.Context, args json.RawMessage) (string, *crux.StateDelta, error) {
		got = append(got, args)
		return "found", nil, nil
	}, crux.WithInputSchema(searchSchema))

	for _, provider := range []struct {
		provider crux.Provider
		model    string
	}{
		{crux.ProviderOpenAI, crux.OpenAIGPT5_4},
		{crux.ProviderAnthropic, crux.ClaudeHaiku4_5},
		{crux.ProviderGoogle, crux.Gemini3_8Flash},
	} {
		t.Run(string(provider.provider), func(t *testing.T) {
			got = nil
			mock := cruxtest.NewMock()
			mock.Expect().ReturnToolCall("search", map[string]any{"query": "caching", "limit": 3})
			mock.Expect().ReturnToolCall("search", map[string]any{"limit": 3})
			mock.Expect().ReturnToolCall("search", map[string]any{"query": "x", "extra": true})
			mock.Expect().ReturnText("done")

			a, err := crux.New("docs", provider.model, append(mock.AgentOptions(), crux.WithProvider(provider.provider),
				crux.WithToolsRegistry([]string{"search"}, reg))...)
			require.NoError(t, err)
			s, err := crux.NewSession(t.Context(), a)
			require.NoError(t, err)
			_, err = s.Run(t.Context(), "look it up")
			require.NoError(t, err)

			require.Len(t, got, 1, "only the valid call reaches the tool")
			require.JSONEq(t, `{"query":"caching","limit":3}`, string(got[0]))
			require.Contains(t, mock.Requests()[0].BodyString(), "What to look for", "the given schema is sent")

			var errors []string
			for _, e := range s.Logs() {
				if e.Kind == crux.KindToolResult && e.ToolResult.Error != "" {
					errors = append(errors, e.ToolResult.Error)
				}
			}
			require.Len(t, errors, 2)
			require.Contains(t, errors[0], "query")
			require.Contains(t, errors[1], "extra")
		})
	}
}

func TestWithInputSchemaMapInput(t *testing.T) {
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "echo", "Echo", func(ctx context.Context, args map[string]any) (any, *crux.StateDelta, error) {
		return args, nil, nil
	}, crux.WithInputSchema(map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "number"}}}))

	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("echo", map[string]any{"a": 1})
	mock.Expect().ReturnText("done")
	a := crux.Must(crux.New("e", crux.OpenAIGPT5_4, append(mock.AgentOptions(), crux.WithToolsRegistry([]string{"echo"}, reg))...))
	s := crux.MustSession(crux.NewSession(t.Context(), a))
	_, err := s.Run(t.Context(), "go")
	require.NoError(t, err)
	logs := s.Logs()
	result := logs[slices.IndexFunc(logs, func(e crux.Entry) bool { return e.Kind == crux.KindToolResult })]
	require.Equal(t, `{"a":1}`, result.ToolResult.Output)
}

func TestWithInputSchemaRejectsBadSchemas(t *testing.T) {
	noop := func(ctx context.Context, args json.RawMessage) (string, *crux.StateDelta, error) { return "", nil, nil }
	tests := map[string]any{
		"not an object":    json.RawMessage(`[1]`),
		"nil":              nil,
		"not type object":  map[string]any{"type": "string"},
		"does not compile": map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": 5}}},
	}
	for name, schema := range tests {
		t.Run(name, func(t *testing.T) {
			require.Panics(t, func() {
				crux.RegisterToolWithRegistry(crux.NewToolsRegistry(), "t", "", noop, crux.WithInputSchema(schema))
			})
		})
	}

	t.Run("input type cannot hold an object", func(t *testing.T) {
		require.Panics(t, func() {
			crux.RegisterToolWithRegistry(crux.NewToolsRegistry(), "t", "", func(ctx context.Context, n int) (string, *crux.StateDelta, error) {
				return "", nil, nil
			}, crux.WithInputSchema(searchSchema))
		})
	})

	t.Run("AddToolset returns it as an error", func(t *testing.T) {
		err := crux.AddToolsetWithRegistry(crux.NewToolsRegistry(), toolsetFunc(func(reg crux.ToolsRegistry) error {
			crux.RegisterToolWithRegistry(reg, "t", "", noop, crux.WithInputSchema(map[string]any{"type": "array"}))
			return nil
		}))
		require.ErrorContains(t, err, `"type": "object"`)
	})
}

type toolsetFunc func(crux.ToolsRegistry) error

func (f toolsetFunc) Register(reg crux.ToolsRegistry) error { return f(reg) }
