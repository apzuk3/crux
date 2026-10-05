package cruxtest_test

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/apzuk3/crux"
	"github.com/apzuk3/crux/cruxtest"
	"github.com/stretchr/testify/require"
)

type lookupArgs struct {
	Query string `json:"query"`
}

// choiceRun runs one tool call and a final answer on an agent with the
// lookup tool and returns the bodies of both requests.
func choiceRun(t *testing.T, provider crux.Provider, model string, opts ...crux.AgentOption) []map[string]any {
	t.Helper()
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "lookup", "Look something up", func(ctx context.Context, in lookupArgs) (string, *crux.StateDelta, error) {
		return "found " + in.Query, nil, nil
	})
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("lookup", map[string]any{"query": "go"})
	mock.Expect().ReturnText("done")
	opts = append([]crux.AgentOption{crux.WithProvider(provider), crux.WithToolsRegistry([]string{"lookup"}, reg)}, opts...)
	a, err := crux.New("chooser", model, append(mock.AgentOptions(), opts...)...)
	require.NoError(t, err)
	s, err := crux.NewSession(t.Context(), a)
	require.NoError(t, err)
	out, err := s.Run(t.Context(), "look up go")
	require.NoError(t, err)
	require.Equal(t, "done", out)
	var bodies []map[string]any
	for _, req := range mock.Requests() {
		var body map[string]any
		require.NoError(t, json.Unmarshal([]byte(req.BodyString()), &body))
		bodies = append(bodies, body)
	}
	require.Len(t, bodies, 2)
	return bodies
}

func TestToolChoiceOpenAI(t *testing.T) {
	tests := []struct {
		choice crux.ToolChoice
		want   any
	}{
		{crux.ToolChoiceRequired, "required"},
		{crux.ToolChoiceNone, "none"},
		{crux.ToolChoiceAuto, "auto"},
		{crux.ToolChoiceTool("lookup"), map[string]any{"type": "function", "name": "lookup"}},
	}
	for _, tt := range tests {
		t.Run(string(tt.choice), func(t *testing.T) {
			bodies := choiceRun(t, crux.ProviderOpenAI, crux.OpenAIGPT5_4, crux.WithToolChoice(tt.choice))
			require.Equal(t, tt.want, bodies[0]["tool_choice"])
			require.NotContains(t, bodies[1], "tool_choice", "only the first request of a run is forced")
		})
	}

	t.Run("parallel", func(t *testing.T) {
		bodies := choiceRun(t, crux.ProviderOpenAI, crux.OpenAIGPT5_4, crux.WithParallelToolCalls(false))
		for _, body := range bodies {
			require.Equal(t, false, body["parallel_tool_calls"])
			require.NotContains(t, body, "tool_choice")
		}
	})

	t.Run("unset sends nothing", func(t *testing.T) {
		bodies := choiceRun(t, crux.ProviderOpenAI, crux.OpenAIGPT5_4)
		require.NotContains(t, bodies[0], "tool_choice")
		require.NotContains(t, bodies[0], "parallel_tool_calls")
	})
}

func TestToolChoiceAnthropic(t *testing.T) {
	tests := []struct {
		name   string
		opts   []crux.AgentOption
		first  any
		second any
	}{
		{"required", []crux.AgentOption{crux.WithToolChoice(crux.ToolChoiceRequired)},
			map[string]any{"type": "any"}, nil},
		{"none", []crux.AgentOption{crux.WithToolChoice(crux.ToolChoiceNone)},
			map[string]any{"type": "none"}, nil},
		{"tool", []crux.AgentOption{crux.WithToolChoice(crux.ToolChoiceTool("lookup"))},
			map[string]any{"type": "tool", "name": "lookup"}, nil},
		{"tool without parallel calls", []crux.AgentOption{crux.WithToolChoice(crux.ToolChoiceTool("lookup")), crux.WithParallelToolCalls(false)},
			map[string]any{"type": "tool", "name": "lookup", "disable_parallel_tool_use": true},
			map[string]any{"type": "auto", "disable_parallel_tool_use": true}},
		{"unset", nil, nil, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bodies := choiceRun(t, crux.ProviderAnthropic, crux.ClaudeSonnet5, tt.opts...)
			for i, want := range []any{tt.first, tt.second} {
				if want == nil {
					require.NotContains(t, bodies[i], "tool_choice")
				} else {
					require.Equal(t, want, bodies[i]["tool_choice"])
				}
			}
		})
	}

	t.Run("forcing a tool while reasoning is rejected", func(t *testing.T) {
		sub := crux.Must(crux.New("helper", crux.ClaudeSonnet5))
		_, err := crux.New("r", crux.ClaudeSonnet5, crux.WithReasoning(crux.ReasoningLow), crux.WithSubAgent(sub, "Helps"),
			crux.WithToolChoice(crux.ToolChoiceRequired))
		require.ErrorContains(t, err, "force a tool call")
		_, err = crux.New("r", crux.ClaudeSonnet5, crux.WithReasoning(crux.ReasoningLow), crux.WithSubAgent(sub, "Helps"),
			crux.WithToolChoice(crux.ToolChoiceNone))
		require.NoError(t, err)
	})
}

func TestToolChoiceGemini(t *testing.T) {
	tests := []struct {
		choice crux.ToolChoice
		want   map[string]any
	}{
		{crux.ToolChoiceRequired, map[string]any{"mode": "ANY"}},
		{crux.ToolChoiceNone, map[string]any{"mode": "NONE"}},
		{crux.ToolChoiceTool("lookup"), map[string]any{"mode": "ANY", "allowedFunctionNames": []any{"lookup"}}},
	}
	for _, tt := range tests {
		t.Run(string(tt.choice), func(t *testing.T) {
			bodies := choiceRun(t, crux.ProviderGoogle, crux.Gemini3_8Flash, crux.WithToolChoice(tt.choice))
			config, _ := bodies[0]["toolConfig"].(map[string]any)
			require.Equal(t, tt.want, config["functionCallingConfig"])
			require.NotContains(t, bodies[1], "toolConfig")
		})
	}

	t.Run("parallel calls cannot be turned off", func(t *testing.T) {
		_, err := crux.New("g", crux.Gemini3_8Flash, crux.WithParallelToolCalls(false))
		require.ErrorContains(t, err, "parallel")
		_, err = crux.New("g", crux.Gemini3_8Flash, crux.WithParallelToolCalls(true))
		require.NoError(t, err)
	})
}

func TestToolChoiceValidation(t *testing.T) {
	_, err := crux.New("v", crux.OpenAIGPT5_4, crux.WithToolChoice("sometimes"))
	require.ErrorContains(t, err, "unknown tool choice")

	_, err = crux.New("v", crux.OpenAIGPT5_4, crux.WithToolChoice(crux.ToolChoiceTool("bad name")))
	require.ErrorContains(t, err, "invalid tool name")

	_, err = crux.New("v", crux.OpenAIGPT5_4, crux.WithToolChoice(crux.ToolChoiceTool("missing")))
	require.ErrorContains(t, err, "not one of its tools")

	_, err = crux.New("v", crux.OpenAIGPT5_4, crux.WithToolChoice(crux.ToolChoiceRequired))
	require.ErrorContains(t, err, "at least one tool")

	sub := crux.Must(crux.New("helper", crux.OpenAIGPT5_4))
	_, err = crux.New("v", crux.OpenAIGPT5_4, crux.WithSubAgent(sub, "Helps"),
		crux.WithToolChoice(crux.ToolChoiceTool("agent_helper")))
	require.NoError(t, err)

	a := crux.Must(crux.New("v", crux.OpenAIGPT5_4))
	b := crux.Must(crux.New("v", crux.OpenAIGPT5_4, crux.WithParallelToolCalls(true)))
	c := crux.Must(crux.New("v", crux.OpenAIGPT5_4, crux.WithMaxRetries(5)))
	require.NotEqual(t, a.ID(), b.ID())
	require.Equal(t, a.ID(), c.ID(), "retries do not change what the model sees")
}

func TestToolChoiceNotRepeatedOnResume(t *testing.T) {
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "lookup", "Look something up", func(ctx context.Context, in lookupArgs) (string, *crux.StateDelta, error) {
		return "found", nil, nil
	}, crux.WithApprovalNeeded(true))
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("lookup", map[string]any{"query": "go"})
	mock.Expect().ReturnText("done")
	a, err := crux.New("chooser", crux.OpenAIGPT5_4, append(mock.AgentOptions(),
		crux.WithToolsRegistry([]string{"lookup"}, reg), crux.WithToolChoice(crux.ToolChoiceRequired))...)
	require.NoError(t, err)
	s, err := crux.NewSession(t.Context(), a)
	require.NoError(t, err)
	_, err = s.Run(t.Context(), "look up go")
	require.ErrorIs(t, err, crux.ErrApprovalNeeded)
	require.NoError(t, s.Approve(t.Context(), s.PendingApprovals()[0].ID))
	out, err := s.Resume(t.Context())
	require.NoError(t, err)
	require.Equal(t, "done", out)

	requests := mock.Requests()
	require.Len(t, requests, 2)
	require.Contains(t, requests[0].BodyString(), `"tool_choice":"required"`)
	require.NotContains(t, requests[1].BodyString(), "tool_choice")
}

func TestMaxRetries(t *testing.T) {
	for _, tt := range []struct {
		provider crux.Provider
		model    string
	}{
		{crux.ProviderOpenAI, crux.OpenAIGPT5_4},
		{crux.ProviderAnthropic, crux.ClaudeSonnet5},
		{crux.ProviderGoogle, crux.Gemini3_8Flash},
	} {
		t.Run(string(tt.provider), func(t *testing.T) {
			for _, retries := range []int{0, 1} {
				mock := cruxtest.NewMock()
				for range retries + 1 {
					mock.Expect().ReturnError(http.StatusInternalServerError, `{"error":{"message":"boom"}}`)
				}
				a, err := crux.New("retry", tt.model, append(mock.AgentOptions(),
					crux.WithProvider(tt.provider), crux.WithMaxRetries(retries))...)
				require.NoError(t, err)
				s, err := crux.NewSession(t.Context(), a)
				require.NoError(t, err)
				_, err = s.Run(t.Context(), "hi")
				require.Error(t, err)
				require.Equal(t, retries+1, mock.Calls())
			}
		})
	}

	_, err := crux.New("retry", crux.OpenAIGPT5_4, crux.WithMaxRetries(-1))
	require.ErrorContains(t, err, "negative")
}

func TestToolTimeout(t *testing.T) {
	reg := crux.NewToolsRegistry()
	release := make(chan struct{})
	defer close(release)
	crux.RegisterToolWithRegistry(reg, "ignores_ctx", "Hangs and ignores its context", func(ctx context.Context, in lookupArgs) (string, *crux.StateDelta, error) {
		<-release
		return "late", nil, nil
	}, crux.WithToolTimeout(20*time.Millisecond))
	crux.RegisterToolWithRegistry(reg, "honours_ctx", "Waits for its context", func(ctx context.Context, in lookupArgs) (string, *crux.StateDelta, error) {
		<-ctx.Done()
		return "", nil, ctx.Err()
	}, crux.WithToolTimeout(20*time.Millisecond))
	crux.RegisterToolWithRegistry(reg, "quick", "Returns at once", func(ctx context.Context, in lookupArgs) (string, *crux.StateDelta, error) {
		return "quick " + in.Query, nil, nil
	}, crux.WithToolTimeout(time.Minute))

	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCalls(
		cruxtest.ToolCall{Name: "ignores_ctx", Args: map[string]any{"query": "a"}},
		cruxtest.ToolCall{Name: "honours_ctx", Args: map[string]any{"query": "b"}},
		cruxtest.ToolCall{Name: "quick", Args: map[string]any{"query": "c"}},
	)
	mock.Expect().ReturnText("done")
	a, err := crux.New("timeouts", crux.OpenAIGPT5_4, append(mock.AgentOptions(),
		crux.WithToolsRegistry([]string{"ignores_ctx", "honours_ctx", "quick"}, reg))...)
	require.NoError(t, err)
	s, err := crux.NewSession(t.Context(), a)
	require.NoError(t, err)
	out, err := s.Run(t.Context(), "go")
	require.NoError(t, err)
	require.Equal(t, "done", out)

	var results []*crux.ToolResult
	for _, e := range s.Logs() {
		if e.Kind == crux.KindToolResult {
			results = append(results, e.ToolResult)
		}
	}
	require.Len(t, results, 3)
	require.Equal(t, `tool "ignores_ctx" timed out after 20ms`, results[0].Error)
	require.Equal(t, `tool "honours_ctx" timed out after 20ms`, results[1].Error)
	require.Empty(t, results[2].Error)
	require.Equal(t, "quick c", results[2].Output)
}
