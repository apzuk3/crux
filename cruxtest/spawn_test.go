package cruxtest_test

import (
	"context"
	"encoding/json"
	"sync/atomic"
	"testing"
	"time"

	"crux.foo"
	"crux.foo/cruxtest"
	"github.com/stretchr/testify/require"
)

// spawnAgent returns an agent with a lookup tool, a delete_file tool that
// needs approval, and agent spawning; it counts the deletions.
func spawnAgent(t *testing.T, mock *cruxtest.Mock, opts ...crux.SpawnOption) (*crux.Agent, *atomic.Int32) {
	t.Helper()
	var deleted atomic.Int32
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "lookup", "Look something up", func(ctx context.Context, in lookupArgs) (string, *crux.StateDelta, error) {
		return "found " + in.Query, nil, nil
	})
	crux.RegisterToolWithRegistry(reg, "delete_file", "Delete a file", func(ctx context.Context, in deleteArgs) (string, *crux.StateDelta, error) {
		deleted.Add(1)
		return "deleted " + in.Path, nil, nil
	}, crux.WithApprovalNeeded(true))
	a := crux.Must(crux.New("boss", crux.OpenAIGPT5_4, append(mock.AgentOptions(),
		crux.WithToolsRegistry([]string{"lookup", "delete_file"}, reg),
		crux.WithAgentSpawning(opts...))...))
	return a, &deleted
}

// requestTools returns the tools of a request by name.
func requestTools(t *testing.T, req *cruxtest.CapturedRequest) map[string]map[string]any {
	t.Helper()
	var body struct {
		Tools []map[string]any `json:"tools"`
	}
	require.NoError(t, json.Unmarshal([]byte(req.BodyString()), &body))
	tools := make(map[string]map[string]any)
	for _, tool := range body.Tools {
		tools[tool["name"].(string)] = tool
	}
	return tools
}

func TestSpawnRunsAgent(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("spawn_agent", map[string]any{
		"name":         "searcher",
		"instructions": "You search carefully.",
		"task":         "find go",
		"tools":        []string{"lookup"},
	})
	mock.Expect().ReturnToolCall("lookup", map[string]any{"query": "go"})
	mock.Expect().ReturnText("go is a language")
	mock.Expect().ReturnText("done")
	a, _ := spawnAgent(t, mock)

	s := crux.MustSession(crux.NewSession(t.Context(), a))
	out, err := s.Run(t.Context(), "research go")
	require.NoError(t, err)
	require.Equal(t, "done", out)

	reqs := mock.Requests()
	require.Len(t, reqs, 4)

	parent := requestTools(t, reqs[0])
	require.Contains(t, parent, "spawn_agent")
	require.Contains(t, parent["spawn_agent"]["description"], "- lookup: Look something up", "the model learns what each tool does")
	params := parent["spawn_agent"]["parameters"].(map[string]any)["properties"].(map[string]any)
	require.Equal(t, []any{"lookup", "delete_file"}, params["tools"].(map[string]any)["items"].(map[string]any)["enum"])
	require.NotContains(t, params, "model", "one allowed model needs no choice")

	child := requestTools(t, reqs[1])
	require.Equal(t, []string{"lookup"}, keys(child), "the spawned agent has only the tools it was given")
	require.Contains(t, reqs[1].BodyString(), "You search carefully.")

	var result *crux.ToolResult
	for _, e := range s.Logs() {
		if e.Kind == crux.KindToolResult {
			result = e.ToolResult
		}
		require.NotEqual(t, crux.KindStateDelta, e.Kind, "a spawned agent's name is no state key")
	}
	require.Equal(t, "go is a language", result.Output)
}

func keys(m map[string]map[string]any) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	return out
}

func TestSpawnRejectsDisallowed(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("spawn_agent", map[string]any{
		"name":         "rogue",
		"instructions": "Do anything.",
		"task":         "spawn more",
		"tools":        []string{"spawn_agent"},
	})
	mock.Expect().ReturnText("could not")
	a, _ := spawnAgent(t, mock)

	s := crux.MustSession(crux.NewSession(t.Context(), a))
	out, err := s.Run(t.Context(), "go")
	require.NoError(t, err)
	require.Equal(t, "could not", out)
	require.Len(t, mock.Requests(), 2, "no agent ran")
	for _, e := range s.Logs() {
		if e.Kind == crux.KindToolResult {
			require.NotEmpty(t, e.ToolResult.Error)
		}
	}
}

func TestSpawnOptions(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("spawn_agent", map[string]any{
		"name":         "fast",
		"instructions": "Be quick.",
		"task":         "say hi",
		"model":        crux.OpenAIGPT5_4Mini,
	})
	mock.Expect().ReturnText("hi")
	mock.Expect().ReturnText("done")
	a, _ := spawnAgent(t, mock,
		crux.WithSpawnModels(crux.OpenAIGPT5_4, crux.OpenAIGPT5_4Mini),
		crux.WithSpawnToolsRegistry(func() crux.ToolsRegistry {
			reg := crux.NewToolsRegistry()
			crux.RegisterToolWithRegistry(reg, "other", "Another tool", func(ctx context.Context, in lookupArgs) (string, *crux.StateDelta, error) {
				return "", nil, nil
			})
			return reg
		}(), "other"))

	s := crux.MustSession(crux.NewSession(t.Context(), a))
	_, err := s.Run(t.Context(), "go")
	require.NoError(t, err)

	reqs := mock.Requests()
	params := requestTools(t, reqs[0])["spawn_agent"]["parameters"].(map[string]any)["properties"].(map[string]any)
	require.Equal(t, []any{crux.OpenAIGPT5_4, crux.OpenAIGPT5_4Mini}, params["model"].(map[string]any)["enum"])
	require.Equal(t, []any{"other"}, params["tools"].(map[string]any)["items"].(map[string]any)["enum"])
	require.Contains(t, reqs[1].BodyString(), `"model":"`+crux.OpenAIGPT5_4Mini+`"`)

	_, err = crux.New("bad", crux.OpenAIGPT5_4, crux.WithAgentSpawning(crux.WithSpawnModels("no-such-model")))
	require.ErrorContains(t, err, "no-such-model")
}

func TestSpawnApprovalSurvivesReload(t *testing.T) {
	store := crux.NewMemoryStore()
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("spawn_agent", map[string]any{
		"name":         "cleaner",
		"instructions": "You delete files.",
		"task":         "delete a.txt",
		"tools":        []string{"delete_file"},
	})
	mock.Expect().ReturnToolCall("delete_file", map[string]any{"path": "a.txt"})
	a, deleted := spawnAgent(t, mock)

	first := crux.MustSession(crux.NewSession(t.Context(), a, crux.WithStore(store)))
	_, err := first.Run(t.Context(), "clean up")
	require.ErrorIs(t, err, crux.ErrApprovalNeeded)

	loaded := crux.MustSession(crux.NewSession(t.Context(), a, crux.WithStore(store), crux.WithSessionID(first.ID())))
	pending := loaded.PendingApprovals()
	require.Len(t, pending, 1)
	require.Equal(t, "cleaner", pending[0].Agent)
	require.NoError(t, loaded.Approve(t.Context(), pending[0].ID))

	mock.Expect().ReturnText("deleted a.txt")
	mock.Expect().ReturnText("all clean")
	out, err := loaded.Resume(t.Context())
	require.NoError(t, err)
	require.Equal(t, "all clean", out)
	require.EqualValues(t, 1, deleted.Load())
	require.Len(t, mock.Requests(), 4, "the spawned agent continues its session")
}

func TestSpawnConcurrency(t *testing.T) {
	var running, most atomic.Int32
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "probe", "Probe", func(ctx context.Context, in lookupArgs) (string, *crux.StateDelta, error) {
		n := running.Add(1)
		defer running.Add(-1)
		for {
			m := most.Load()
			if n <= m || most.CompareAndSwap(m, n) {
				break
			}
		}
		time.Sleep(20 * time.Millisecond)
		return "ok", nil, nil
	})

	mock := cruxtest.NewMock()
	var calls []cruxtest.ToolCall
	for _, name := range []string{"a", "b", "c"} {
		calls = append(calls, cruxtest.ToolCall{Name: "spawn_agent", Args: map[string]any{
			"name": name, "instructions": "Probe.", "task": "probe", "tools": []string{"probe"},
		}})
	}
	mock.Expect().ReturnToolCalls(calls...)
	for range 3 {
		// With one agent at a time, each finishes before the next starts.
		mock.Expect().ReturnToolCall("probe", map[string]any{"query": "x"})
		mock.Expect().ReturnText("probed")
	}
	mock.Expect().ReturnText("done")

	a := crux.Must(crux.New("boss", crux.OpenAIGPT5_4, append(mock.AgentOptions(),
		crux.WithToolsRegistry([]string{"probe"}, reg),
		crux.WithAgentSpawning(crux.WithSpawnConcurrency(1)))...))
	s := crux.MustSession(crux.NewSession(t.Context(), a))
	out, err := s.Run(t.Context(), "probe three times")
	require.NoError(t, err)
	require.Equal(t, "done", out)
	require.EqualValues(t, 1, most.Load(), "one spawned agent at a time")

	_, err = crux.New("bad", crux.OpenAIGPT5_4, crux.WithAgentSpawning(crux.WithSpawnConcurrency(0)))
	require.Error(t, err)
}
