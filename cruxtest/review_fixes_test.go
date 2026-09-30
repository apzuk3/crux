package cruxtest_test

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/apzuk3/crux"
	"github.com/apzuk3/crux/cruxtest"
	"github.com/stretchr/testify/require"
)

// A turn with only reasoning is an empty final answer, not a reason to ask
// the provider again with history that ends in an assistant message.
func TestReasoningOnlyTurnIsFinal(t *testing.T) {
	cases := map[string]string{
		crux.ClaudeHaiku4_5:  `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"thinking","thinking":"hmm","signature":"sig"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
		crux.Gemini2_5Flash:  `{"candidates":[{"content":{"role":"model","parts":[{"text":"","thoughtSignature":"c2ln"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`,
		crux.OpenAIGPT5_6Sol: `{"id":"resp_1","status":"completed","output":[{"id":"rs_1","type":"reasoning","summary":[],"encrypted_content":"enc"}],"usage":{"input_tokens":1,"output_tokens":1}}`,
	}
	for model, body := range cases {
		t.Run(model, func(t *testing.T) {
			mock := cruxtest.NewMock()
			mock.Expect().ReturnRaw(200, []byte(body))
			s := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, model)))
			out, err := s.Run(t.Context(), "hi")
			require.NoError(t, err)
			require.Empty(t, out)
			require.Equal(t, 1, mock.Calls())
			require.NotNil(t, lastEntry(t, s).Usage)
		})
	}
}

// Output items from one OpenAI-compatible provider are not replayed to another
// when a stored session is loaded with a different agent.
func TestStoredSessionDropsForeignOpenAIItems(t *testing.T) {
	store := crux.NewMemoryStore()
	openaiMock := cruxtest.NewMock()
	openaiMock.Expect().ReturnRaw(200, []byte(`{"id":"resp_1","status":"completed","output":[{"id":"rs_secret","type":"reasoning","summary":[],"encrypted_content":"openai-only"},{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"first","annotations":[]}]}],"usage":{"input_tokens":1,"output_tokens":1}}`))
	s := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, openaiMock, crux.OpenAIGPT5_6Sol), crux.WithStore(store)))
	_, err := s.Run(t.Context(), "hi")
	require.NoError(t, err)

	xaiMock := cruxtest.NewMock(cruxtest.WithProvider(crux.ProviderOpenAI))
	xaiMock.Expect().ReturnText("second")
	xai := newMockAgent(t, xaiMock, "grok-test", crux.WithProvider(crux.ProviderXAI))
	resumed := crux.MustSession(crux.NewSession(t.Context(), xai, crux.WithSessionID(s.ID()), crux.WithStore(store)))
	out, err := resumed.Run(t.Context(), "again")
	require.NoError(t, err)
	require.Equal(t, "second", out)

	body := xaiMock.Requests()[0].BodyString()
	require.NotContains(t, body, "openai-only")
	require.NotContains(t, body, "rs_secret")
	require.Contains(t, body, "first")
}

type probeArgs struct {
	Q string `json:"q"`
}

func TestPendingApprovalsReturnsCopies(t *testing.T) {
	reg := crux.NewToolsRegistry()
	var got string
	crux.RegisterToolWithRegistry(reg, "danger", "d", func(ctx context.Context, in probeArgs) (string, *crux.StateDelta, error) {
		got = in.Q
		return "ok", nil, nil
	}, crux.WithApprovalNeeded(true))
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("danger", map[string]any{"q": "safe"})
	mock.Expect().ReturnText("done")
	s := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, crux.ClaudeHaiku4_5, crux.WithToolsRegistry([]string{"danger"}, reg))))
	_, err := s.Run(t.Context(), "go")
	require.ErrorIs(t, err, crux.ErrApprovalNeeded)

	pending := s.PendingApprovals()
	require.Len(t, pending, 1)
	pending[0].Args = []byte(`{"q":"evil"}`)
	pending[0].Name = "other"
	require.NoError(t, s.Approve(t.Context(), pending[0].ID))
	_, err = s.Resume(t.Context())
	require.NoError(t, err)
	require.Equal(t, "safe", got)
}

// State the store cannot encode is reported to the model, so the tool's
// result is still stored and the tool does not run again on Resume.
func TestUnencodableStateIsReportedNotRetried(t *testing.T) {
	reg := crux.NewToolsRegistry()
	runs := 0
	crux.RegisterToolWithRegistry(reg, "bad_state", "b", func(ctx context.Context, in probeArgs) (string, *crux.StateDelta, error) {
		runs++
		return "ok", &crux.StateDelta{Set: map[string]any{"x": math.NaN()}}, nil
	})
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("bad_state", map[string]any{"q": "x"})
	mock.Expect().ReturnText("done")
	s := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, crux.ClaudeHaiku4_5, crux.WithToolsRegistry([]string{"bad_state"}, reg))))
	_, err := s.Run(t.Context(), "go")
	require.NoError(t, err)
	require.Equal(t, 1, runs)
	var result *crux.ToolResult
	for _, e := range s.Logs() {
		require.NotEqual(t, crux.KindStateDelta, e.Kind)
		if e.ToolResult != nil {
			result = e.ToolResult
		}
	}
	require.NotNil(t, result)
	require.True(t, strings.Contains(result.Error, "cannot be stored as JSON"), result.Error)
}
