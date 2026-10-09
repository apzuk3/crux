package cruxtest_test

import (
	"context"
	"testing"

	"crux.foo"
	"crux.foo/cruxtest"
	"github.com/stretchr/testify/require"
)

// The agents in these tests are built without any test options, as an
// application builds them; only the session gets the mock's client.

func TestSessionClientReachesSubAgents(t *testing.T) {
	helper := crux.Must(crux.New("helper", crux.OpenAIGPT5_4))
	parent := crux.Must(crux.New("parent", crux.OpenAIGPT5_4, crux.WithSubAgent(helper, "Helps")))

	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("agent_helper", map[string]any{"task": "say hi"})
	mock.Expect().ReturnText("hi")
	mock.Expect().ReturnText("helper said hi")

	s := crux.MustSession(crux.NewSession(t.Context(), parent, crux.WithHTTPClient(mock.Client())))
	out, err := s.Run(t.Context(), "go")
	require.NoError(t, err)
	require.Equal(t, "helper said hi", out)
	mock.AssertAllConsumed(t)
}

func TestSessionClientReachesSpawnedAgents(t *testing.T) {
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "lookup", "Look something up", func(ctx context.Context, in lookupArgs) (string, *crux.StateDelta, error) {
		return "found " + in.Query, nil, nil
	})
	boss := crux.Must(crux.New("boss", crux.OpenAIGPT5_4, crux.WithToolsRegistry([]string{"lookup"}, reg), crux.WithAgentSpawning()))

	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("spawn_agent", map[string]any{"name": "searcher", "instructions": "Search.", "task": "find go"})
	mock.Expect().ReturnText("go is a language")
	mock.Expect().ReturnText("done")

	s := crux.MustSession(crux.NewSession(t.Context(), boss, crux.WithHTTPClient(mock.Client())))
	out, err := s.Run(t.Context(), "go")
	require.NoError(t, err)
	require.Equal(t, "done", out)
	mock.AssertAllConsumed(t)
}

func TestSessionClientReachesCompaction(t *testing.T) {
	a := crux.Must(crux.New("chat", crux.OpenAIGPT5_4))
	mock := cruxtest.NewMock()
	mock.Expect().ReturnText("first answer")
	mock.Expect().ReturnText("second answer")
	mock.Expect().ReturnText("## Summary\nThe user asked twice.")

	s := crux.MustSession(crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client())))
	_, err := s.Run(t.Context(), "first")
	require.NoError(t, err)
	_, err = s.Run(t.Context(), "second")
	require.NoError(t, err)
	require.NoError(t, s.Compact(t.Context()))
	mock.AssertAllConsumed(t)
}

func TestForkKeepsSessionClient(t *testing.T) {
	a := crux.Must(crux.New("chat", crux.OpenAIGPT5_4))
	mock := cruxtest.NewMock()
	mock.Expect().ReturnText("one")
	mock.Expect().ReturnText("two")

	s := crux.MustSession(crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client())))
	_, err := s.Run(t.Context(), "hi")
	require.NoError(t, err)
	fork, err := s.Fork(t.Context(), crux.WithInstructions("Be brief."))
	require.NoError(t, err)
	out, err := fork.Run(t.Context(), "again")
	require.NoError(t, err)
	require.Equal(t, "two", out)
	mock.AssertAllConsumed(t)
}

func TestGeminiSessionClientWithoutKey(t *testing.T) {
	for _, name := range []string{"GOOGLE_API_KEY", "GOOGLE_APIKEY", "GOOGLE_KEY", "GEMINI_API_KEY", "GEMINI_APIKEY", "GEMINI_KEY"} {
		t.Setenv(name, "")
	}
	a := crux.Must(crux.New("gem", crux.Gemini2_5Flash))
	mock := cruxtest.NewMock(cruxtest.WithProvider(crux.ProviderGoogle))
	mock.Expect().ReturnText("hello")

	s := crux.MustSession(crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client())))
	out, err := s.Run(t.Context(), "hi")
	require.NoError(t, err)
	require.Equal(t, "hello", out)
}

func TestDeciderWithHTTPClientCopies(t *testing.T) {
	d := crux.MustDecider(crux.NewDecider(crux.Jev))
	mock := cruxtest.NewMock()
	mock.Expect().ReturnDecision(map[string]cruxtest.Answer{"team": cruxtest.Choice("web", 0.9, nil)})

	bound := d.WithHTTPClient(mock.Client())
	require.NotSame(t, d, bound)
	res, err := crux.Decide[routed](t.Context(), bound, "blank page")
	require.NoError(t, err)
	require.Equal(t, team("web"), res.Value.Team)
	mock.AssertAllConsumed(t)
}
