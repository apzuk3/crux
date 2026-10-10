package cruxtest_test

import (
	"context"
	"testing"

	"crux.foo"
	"crux.foo/cruxtest"
	"github.com/stretchr/testify/require"
)

// A session serves one operation at a time: a second one started while a
// run is in progress fails with ErrSessionBusy instead of racing.
func TestSessionBusy(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "block", "Block until released", func(ctx context.Context, in struct{}) (string, *crux.StateDelta, error) {
		close(started)
		<-release
		return "released", nil, nil
	})
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("block", map[string]any{})
	mock.Expect().ReturnText("done")
	mock.Expect().ReturnText("again")
	a, err := crux.New("busy", crux.OpenAIGPT5_4, crux.WithToolsRegistry([]string{"block"}, reg))
	require.NoError(t, err)
	s := crux.MustSession(crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client())))

	result := make(chan error, 1)
	go func() {
		_, err := s.Run(t.Context(), "go")
		result <- err
	}()
	<-started

	_, err = s.Run(t.Context(), "another")
	require.ErrorIs(t, err, crux.ErrSessionBusy)
	_, err = s.Resume(t.Context())
	require.ErrorIs(t, err, crux.ErrSessionBusy)
	require.ErrorIs(t, s.Approve(t.Context(), "call_1"), crux.ErrSessionBusy)
	require.ErrorIs(t, s.Compact(t.Context()), crux.ErrSessionBusy)
	_, err = s.Fork(t.Context())
	require.ErrorIs(t, err, crux.ErrSessionBusy)
	for _, chunkErr := range s.Stream(t.Context(), "streamed") {
		require.ErrorIs(t, chunkErr, crux.ErrSessionBusy)
	}

	close(release)
	require.NoError(t, <-result)

	out, err := s.Run(t.Context(), "free again")
	require.NoError(t, err)
	require.Equal(t, "again", out)
	mock.AssertAllConsumed(t)
}
