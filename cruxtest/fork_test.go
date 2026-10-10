package cruxtest_test

import (
	"testing"

	"crux.foo"
	"crux.foo/cruxtest"
	"github.com/stretchr/testify/require"
)

func TestForkKeepsHTTPClient(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnText("from parent")
	mock.Expect().ReturnText("from fork")
	session := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, crux.OpenAIGPT5_6Sol), crux.WithHTTPClient(mock.Client())))
	_, err := session.Run(t.Context(), "hi")
	require.NoError(t, err)

	forked, err := session.Fork(t.Context())
	require.NoError(t, err)
	out, err := forked.Run(t.Context(), "again")
	require.NoError(t, err)
	require.Equal(t, "from fork", out)
	mock.AssertAllConsumed(t)
	require.Contains(t, mock.Requests()[1].BodyString(), "again")
}

func TestForkUsageCountsOnlyOwnRequests(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnText("one").WithUsage(cruxtest.TokenUsage{InputTokens: 10, OutputTokens: 2})
	mock.Expect().ReturnText("two").WithUsage(cruxtest.TokenUsage{InputTokens: 7, OutputTokens: 1})
	session := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, crux.OpenAIGPT5_6Sol), crux.WithHTTPClient(mock.Client())))
	_, err := session.Run(t.Context(), "hi")
	require.NoError(t, err)

	forked, err := session.Fork(t.Context())
	require.NoError(t, err)
	require.Equal(t, crux.Usage{}, forked.Usage())

	_, err = forked.Run(t.Context(), "again")
	require.NoError(t, err)
	require.Equal(t, 7, forked.Usage().InputTokens)
	require.Equal(t, 10, session.Usage().InputTokens)
}

// A fork to another provider drops the reasoning and parallel-call settings
// chosen for the old one, which the new provider might reject.
func TestForkToOtherProviderDropsProviderSettings(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnText("first")
	mock.Expect().ReturnText("second")
	mock.Expect().ReturnText("third")
	a, err := crux.New("r", crux.ClaudeOpus4_8, crux.WithReasoning(crux.ReasoningLow), crux.WithParallelToolCalls(false))
	require.NoError(t, err)
	s, err := crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client()))
	require.NoError(t, err)
	_, err = s.Run(t.Context(), "hi")
	require.NoError(t, err)

	xai, err := s.Fork(t.Context(), crux.WithModel(crux.XAIGrok4_20Reasoning))
	require.NoError(t, err, "grok-4.20 rejects a reasoning effort, so the fork must drop it")
	_, err = xai.Run(t.Context(), "again")
	require.NoError(t, err)
	require.NotContains(t, mock.Requests()[1].BodyString(), `"reasoning"`)

	gemini, err := s.Fork(t.Context(), crux.WithModel(crux.Gemini2_5Flash))
	require.NoError(t, err, "gemini rejects WithParallelToolCalls(false), so the fork must drop it")
	_, err = gemini.Run(t.Context(), "again")
	require.NoError(t, err)

	same, err := s.Fork(t.Context(), crux.WithModel(crux.ClaudeHaiku4_5))
	require.NoError(t, err)
	require.Equal(t, crux.ClaudeHaiku4_5, same.Agent().Model())
}
