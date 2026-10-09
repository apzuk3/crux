package cruxtest_test

import (
	"testing"

	"crux.foo"
	"crux.foo/cruxtest"
	"github.com/stretchr/testify/require"
)

func TestForkUsesNewHTTPClient(t *testing.T) {
	parentMock := cruxtest.NewMock()
	parentMock.Expect().ReturnText("from parent")
	session := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, parentMock, crux.OpenAIGPT5_6Sol)))
	_, err := session.Run(t.Context(), "hi")
	require.NoError(t, err)

	forkMock := cruxtest.NewMock()
	forkMock.Expect().ReturnText("from fork")
	forked, err := session.Fork(t.Context(), crux.WithHTTPClient(forkMock.Client()))
	require.NoError(t, err)
	out, err := forked.Run(t.Context(), "again")
	require.NoError(t, err)
	require.Equal(t, "from fork", out)
	parentMock.AssertTurnCount(t, 1)
	forkMock.AssertAllConsumed(t)
}

func TestForkUsageCountsOnlyOwnRequests(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnText("one").WithUsage(cruxtest.TokenUsage{InputTokens: 10, OutputTokens: 2})
	mock.Expect().ReturnText("two").WithUsage(cruxtest.TokenUsage{InputTokens: 7, OutputTokens: 1})
	session := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, crux.OpenAIGPT5_6Sol)))
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
