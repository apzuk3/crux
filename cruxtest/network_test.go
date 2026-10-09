package cruxtest_test

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"crux.foo"
	"crux.foo/cruxtest"
	"github.com/stretchr/testify/require"
)

func TestNetworkToolsInARun(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "service is healthy")
	}))
	t.Cleanup(ts.Close)

	network := crux.Network()
	t.Cleanup(func() { network.Close() })
	reg := crux.NewToolsRegistry()
	require.NoError(t, crux.AddToolsetWithRegistry(reg, network))

	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall(crux.NetList, map[string]any{})
	mock.Expect().ReturnToolCall(crux.NetHTTPRequest, map[string]any{"url": ts.URL + "/health"})
	agent := crux.Must(crux.New("netops", crux.OpenAIGPT5_4,
		append(mock.AgentOptions(), crux.WithToolsetsRegistry(reg, crux.ToolsetNetwork))...))

	s := crux.MustSession(crux.NewSession(t.Context(), agent))
	_, err := s.Run(t.Context(), "is the service up?")
	require.ErrorIs(t, err, crux.ErrApprovalNeeded, "http_request needs approval; net_list does not")
	require.Contains(t, mock.Requests()[1].BodyString(), "no open handles")

	pending := s.PendingApprovals()
	require.Len(t, pending, 1)
	require.Equal(t, crux.NetHTTPRequest, pending[0].Name)
	require.NoError(t, s.Approve(t.Context(), pending[0].ID))

	mock.Expect().ReturnText("It is up.")
	out, err := s.Resume(t.Context())
	require.NoError(t, err)
	require.Equal(t, "It is up.", out)
	require.Contains(t, mock.Requests()[2].BodyString(), "service is healthy")
}
