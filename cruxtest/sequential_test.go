package cruxtest_test

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"crux.foo"
	"crux.foo/cruxtest"
	"github.com/stretchr/testify/require"
)

type stepArgs struct {
	N int `json:"n"`
}

// TestSequentialToolOrder checks that calls of a sequential tool run one at a
// time in the model's order, while a call of another tool in the same turn
// runs alongside them.
func TestSequentialToolOrder(t *testing.T) {
	reg := crux.NewToolsRegistry()
	var (
		mu      sync.Mutex
		order   []int
		running int
		overlap bool
	)
	parallelRan := make(chan struct{})
	crux.RegisterToolWithRegistry(reg, "step", "A step whose order matters", func(ctx context.Context, in stepArgs) (string, *crux.StateDelta, error) {
		mu.Lock()
		running++
		overlap = overlap || running > 1
		mu.Unlock()
		if in.N == 1 {
			// The first step outlasts the others and waits for the parallel
			// tool, so the other steps would overtake it if they ran concurrently.
			select {
			case <-parallelRan:
			case <-time.After(5 * time.Second):
				return "", nil, context.DeadlineExceeded
			}
			time.Sleep(20 * time.Millisecond)
		}
		mu.Lock()
		running--
		order = append(order, in.N)
		mu.Unlock()
		return "ok", nil, nil
	}, crux.WithSequential())
	crux.RegisterToolWithRegistry(reg, "other", "An independent tool", func(ctx context.Context, in stepArgs) (string, *crux.StateDelta, error) {
		close(parallelRan)
		return "ok", nil, nil
	})

	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCalls(
		cruxtest.ToolCall{Name: "step", Args: map[string]any{"n": 1}},
		cruxtest.ToolCall{Name: "step", Args: map[string]any{"n": 2}},
		cruxtest.ToolCall{Name: "other", Args: map[string]any{"n": 0}},
		cruxtest.ToolCall{Name: "step", Args: map[string]any{"n": 3}},
	)
	mock.Expect().ReturnText("done")
	a, err := crux.New("seq", crux.OpenAIGPT5_4, append(mock.AgentOptions(),
		crux.WithToolsRegistry([]string{"step", "other"}, reg))...)
	require.NoError(t, err)
	s, err := crux.NewSession(t.Context(), a)
	require.NoError(t, err)
	_, err = s.Run(t.Context(), "go")
	require.NoError(t, err)

	require.Equal(t, []int{1, 2, 3}, order)
	require.False(t, overlap)
	for _, e := range s.Logs() {
		if e.Kind == crux.KindToolResult {
			require.Empty(t, e.ToolResult.Error)
		}
	}
}

// TestSequentialToolState checks that each sequential call sees the state
// changes of the calls before it in the same turn.
func TestSequentialToolState(t *testing.T) {
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "increment", "Add one to the counter", func(ctx context.Context, in stepArgs) (string, *crux.StateDelta, error) {
		state, _ := crux.StateFromContext(ctx)
		count, _ := state["count"].(int)
		return "ok", &crux.StateDelta{Set: map[string]any{"count": count + 1}}, nil
	}, crux.WithSequential())

	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCalls(
		cruxtest.ToolCall{Name: "increment", Args: map[string]any{"n": 1}},
		cruxtest.ToolCall{Name: "increment", Args: map[string]any{"n": 2}},
		cruxtest.ToolCall{Name: "increment", Args: map[string]any{"n": 3}},
	)
	mock.Expect().ReturnText("done")
	a, err := crux.New("counter", crux.OpenAIGPT5_4, append(mock.AgentOptions(),
		crux.WithToolsRegistry([]string{"increment"}, reg))...)
	require.NoError(t, err)
	s, err := crux.NewSession(t.Context(), a)
	require.NoError(t, err)
	_, err = s.Run(t.Context(), "count to three")
	require.NoError(t, err)
	require.Equal(t, 3, s.StateSnapshot()["count"])
}

// TestFilesystemCallsKeepOrder writes a file into a new directory and edits
// it in one turn: each call depends on the one before it.
func TestFilesystemCallsKeepOrder(t *testing.T) {
	root := t.TempDir()
	reg := crux.NewToolsRegistry()
	require.NoError(t, crux.AddToolsetWithRegistry(reg, crux.Filesystem(root)))

	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCalls(
		cruxtest.ToolCall{Name: crux.FsCreateDirectory, Args: map[string]any{"paths": []string{"notes"}}},
		cruxtest.ToolCall{Name: crux.FsWriteFile, Args: map[string]any{"path": "notes/todo.txt", "content": "one"}},
		cruxtest.ToolCall{Name: crux.FsEditFile, Args: map[string]any{"path": "notes/todo.txt",
			"edits": []map[string]any{{"old_text": "one", "new_text": "two"}}}},
		cruxtest.ToolCall{Name: crux.FsReadFile, Args: map[string]any{"path": "notes/todo.txt"}},
	)
	mock.Expect().ReturnText("done")
	a, err := crux.New("writer", crux.OpenAIGPT5_4, append(mock.AgentOptions(),
		crux.WithToolsetsRegistry(reg, crux.ToolsetFilesystem))...)
	require.NoError(t, err)
	s, err := crux.NewSession(t.Context(), a)
	require.NoError(t, err)
	_, err = s.Run(t.Context(), "write a todo")
	require.ErrorIs(t, err, crux.ErrApprovalNeeded)
	for _, call := range s.PendingApprovals() {
		require.NoError(t, s.Approve(t.Context(), call.ID))
	}
	_, err = s.Resume(t.Context())
	require.NoError(t, err)

	var results []*crux.ToolResult
	for _, e := range s.Logs() {
		if e.Kind == crux.KindToolResult {
			require.Empty(t, e.ToolResult.Error)
			results = append(results, e.ToolResult)
		}
	}
	require.Len(t, results, 4)
	require.Contains(t, results[3].Output, "two")
	content, err := os.ReadFile(filepath.Join(root, "notes", "todo.txt"))
	require.NoError(t, err)
	require.Equal(t, "two", string(content))
}
