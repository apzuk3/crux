package cruxtest_test

import (
	"context"
	"sync/atomic"
	"testing"

	"crux.foo"
	"crux.foo/cruxtest"
	"github.com/stretchr/testify/require"
)

type deleteArgs struct {
	Path string `json:"path"`
}

// approvalAgents returns a coordinator whose "worker" subagent has a
// delete_file tool that needs approval, and counts the deletions.
func approvalAgents(t *testing.T, mock *cruxtest.Mock) (*crux.Agent, *atomic.Int32) {
	t.Helper()
	var deleted atomic.Int32
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "delete_file", "Delete a file", func(ctx context.Context, in deleteArgs) (string, *crux.StateDelta, error) {
		deleted.Add(1)
		return "deleted " + in.Path, nil, nil
	}, crux.WithApprovalNeeded(true))

	worker := crux.Must(crux.New("worker", crux.OpenAIGPT5_4, append(mock.AgentOptions(),
		crux.WithToolsRegistry([]string{"delete_file"}, reg))...))
	coordinator := crux.Must(crux.New("coordinator", crux.OpenAIGPT5_4, append(mock.AgentOptions(),
		crux.WithSubAgent(worker, "Does file work"))...))
	return coordinator, &deleted
}

func TestSubagentApprovalReachesParent(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("agent_worker", map[string]any{"task": "delete a.txt"})
	mock.Expect().ReturnToolCall("delete_file", map[string]any{"path": "a.txt"})
	coordinator, deleted := approvalAgents(t, mock)

	s := crux.MustSession(crux.NewSession(t.Context(), coordinator))
	_, err := s.Run(t.Context(), "clean up")
	require.ErrorIs(t, err, crux.ErrApprovalNeeded)
	require.Zero(t, deleted.Load())

	pending := s.PendingApprovals()
	require.Len(t, pending, 1)
	require.Equal(t, "delete_file", pending[0].Name)
	require.Equal(t, "worker", pending[0].Agent)
	require.JSONEq(t, `{"path":"a.txt"}`, string(pending[0].Args))

	_, err = s.Run(t.Context(), "more input")
	require.ErrorIs(t, err, crux.ErrApprovalNeeded, "new input waits for the decision")

	require.NoError(t, s.Approve(t.Context(), pending[0].ID))
	require.Empty(t, s.PendingApprovals())

	mock.Expect().ReturnText("worker: deleted a.txt")
	mock.Expect().ReturnText("all clean")
	out, err := s.Resume(t.Context())
	require.NoError(t, err)
	require.Equal(t, "all clean", out)
	require.EqualValues(t, 1, deleted.Load())

	var result *crux.ToolResult
	for _, e := range s.Logs() {
		if e.Kind == crux.KindToolResult {
			require.Nil(t, result, "the subagent call is answered once")
			result = e.ToolResult
		}
	}
	require.Equal(t, "worker: deleted a.txt", result.Output)
	require.Len(t, mock.Requests(), 4, "the worker continues its session instead of starting again")
}

func TestSubagentRejection(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("agent_worker", map[string]any{"task": "delete a.txt"})
	mock.Expect().ReturnToolCall("delete_file", map[string]any{"path": "a.txt"})
	coordinator, deleted := approvalAgents(t, mock)

	s := crux.MustSession(crux.NewSession(t.Context(), coordinator))
	_, err := s.Run(t.Context(), "clean up")
	require.ErrorIs(t, err, crux.ErrApprovalNeeded)
	require.NoError(t, s.Reject(t.Context(), s.PendingApprovals()[0].ID, "keep it"))

	mock.Expect().ReturnText("worker: not allowed")
	mock.Expect().ReturnText("left it alone")
	out, err := s.Resume(t.Context())
	require.NoError(t, err)
	require.Equal(t, "left it alone", out)
	require.Zero(t, deleted.Load())
	require.Contains(t, mock.Requests()[2].BodyString(), "keep it", "the worker's model sees the reason")
}

func TestSubagentApprovalSurvivesReload(t *testing.T) {
	store := crux.NewMemoryStore()
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCalls(
		cruxtest.ToolCall{Name: "agent_worker", Args: map[string]any{"task": "delete a.txt"}},
		cruxtest.ToolCall{Name: "agent_worker", Args: map[string]any{"task": "delete b.txt"}},
	)
	mock.Expect().ReturnToolCall("delete_file", map[string]any{"path": "a.txt"})
	mock.Expect().ReturnToolCall("delete_file", map[string]any{"path": "b.txt"})
	coordinator, deleted := approvalAgents(t, mock)

	first := crux.MustSession(crux.NewSession(t.Context(), coordinator, crux.WithStore(store)))
	_, err := first.Run(t.Context(), "clean up")
	require.ErrorIs(t, err, crux.ErrApprovalNeeded)
	require.Len(t, first.PendingApprovals(), 2)

	// Another process loads the session and decides.
	loaded := crux.MustSession(crux.NewSession(t.Context(), coordinator, crux.WithStore(store), crux.WithSessionID(first.ID())))
	pending := loaded.PendingApprovals()
	require.Len(t, pending, 2)
	var paths []string
	for _, call := range pending {
		require.Equal(t, "worker", call.Agent)
		paths = append(paths, string(call.Args))
	}
	require.ElementsMatch(t, []string{`{"path":"a.txt"}`, `{"path":"b.txt"}`}, paths)

	require.NoError(t, loaded.Approve(t.Context(), pending[0].ID))
	_, err = loaded.Resume(t.Context())
	require.ErrorIs(t, err, crux.ErrApprovalNeeded, "one call is still undecided")
	require.Zero(t, deleted.Load(), "nothing runs until every call is decided")

	require.NoError(t, loaded.Approve(t.Context(), pending[1].ID))
	mock.Expect().ReturnText("worker done")
	mock.Expect().ReturnText("worker done")
	mock.Expect().ReturnText("all clean")
	out, err := loaded.Resume(t.Context())
	require.NoError(t, err)
	require.Equal(t, "all clean", out)
	require.EqualValues(t, 2, deleted.Load())
}

func TestSubagentWithoutApprovalsIsUnchanged(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("agent_helper", map[string]any{"task": "say hi"})
	mock.Expect().ReturnText("hi")
	mock.Expect().ReturnText("helper said hi")

	helper := crux.Must(crux.New("helper", crux.OpenAIGPT5_4, mock.AgentOptions()...))
	parent := crux.Must(crux.New("parent", crux.OpenAIGPT5_4, append(mock.AgentOptions(), crux.WithSubAgent(helper, "Helps"))...))
	s := crux.MustSession(crux.NewSession(t.Context(), parent))
	out, err := s.Run(t.Context(), "go")
	require.NoError(t, err)
	require.Equal(t, "helper said hi", out)
	require.Equal(t, "hi", s.StateSnapshot()["helper"])
}
