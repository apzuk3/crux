package cruxtest_test

import (
	"context"
	"testing"
	"time"

	"crux.foo"
	"crux.foo/cruxtest"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

type cityArgs struct {
	City string `json:"city"`
}

// weatherRegistry has a "weather" tool that reports the city it was asked about.
func weatherRegistry(opts ...crux.ToolOption) crux.ToolsRegistry {
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "weather", "Weather for a city", func(ctx context.Context, in cityArgs) (string, *crux.StateDelta, error) {
		return "sunny in " + in.City, nil, nil
	}, opts...)
	return reg
}

// recorder is an entry handler that keeps what it receives. It is not safe
// for concurrent use, so it also checks that calls are serialised.
type recorder struct {
	entries  []crux.Entry
	sessions []*crux.Session
}

func (r *recorder) handle(ctx context.Context, s *crux.Session, e crux.Entry) {
	r.entries = append(r.entries, e)
	r.sessions = append(r.sessions, s)
}

func kinds(entries []crux.Entry) []crux.Kind {
	var out []crux.Kind
	for _, e := range entries {
		out = append(out, e.Kind)
	}
	return out
}

func TestRunIsRecordedInTheLog(t *testing.T) {
	for _, model := range wireModels {
		t.Run(model, func(t *testing.T) {
			mock := cruxtest.NewMock()
			mock.Expect().ReturnToolCall("weather", map[string]any{"city": "Paris"})
			mock.Expect().ReturnText("It is sunny.")
			agent := newMockAgent(t, mock, model, crux.WithToolsRegistry([]string{"weather"}, weatherRegistry()))
			var rec recorder
			s := crux.MustSession(crux.NewSession(t.Context(), agent, crux.WithEntryHandler(rec.handle), crux.WithHTTPClient(mock.Client())))

			out, err := s.Run(t.Context(), "Weather?")
			require.NoError(t, err)
			require.Equal(t, "It is sunny.", out)

			logs := s.Logs()
			require.Equal(t, []crux.Kind{
				crux.KindRunStarted, crux.KindUser,
				crux.KindTurnStarted, crux.KindToolCall, crux.KindToolStarted, crux.KindToolResult,
				crux.KindTurnStarted, crux.KindAssistant,
				crux.KindRunFinished,
			}, kinds(logs))
			require.Equal(t, logs, rec.entries, "the handler sees exactly what was stored")

			require.Equal(t, crux.RunStatus{Outcome: crux.RunAnswered}, *logs[8].Run)
			turn := logs[2].Turn
			require.Equal(t, agent.ID(), turn.AgentID)
			require.Equal(t, agent.Provider(), turn.Provider)
			require.Equal(t, model, turn.Model)
			require.Equal(t, logs[3].ToolCall.ID, logs[4].ToolCall.ID)
			require.Equal(t, "weather", logs[4].ToolCall.Name)
			require.NotEmpty(t, logs[7].Response.ID)
			require.Zero(t, logs[7].Response.FirstTokenAfter, "a response that was not streamed has no first token")

			// Lifecycle records are not sent to the model, and do not hide the answer.
			answer, ok := s.FinalOutput()
			require.True(t, ok)
			require.Equal(t, "It is sunny.", answer)
			mock.Expect().ReturnText("Still sunny.")
			_, err = s.Run(t.Context(), "And now?")
			require.NoError(t, err)
			require.NotContains(t, mock.Requests()[2].BodyString(), string(crux.RunAnswered))
		})
	}
}

func TestStreamRecordsTimeToFirstToken(t *testing.T) {
	for _, model := range wireModels {
		t.Run(model, func(t *testing.T) {
			mock := cruxtest.NewMock()
			mock.Expect().ReturnText("Hello there")
			s := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, model), crux.WithHTTPClient(mock.Client())))
			for _, err := range s.Stream(t.Context(), "Hi") {
				require.NoError(t, err)
			}
			answer := lastEntry(t, s)
			require.NotNil(t, answer.Response)
			require.NotEmpty(t, answer.Response.ID)
			require.GreaterOrEqual(t, answer.Response.FirstTokenAfter, time.Duration(0))
			require.LessOrEqual(t, answer.Response.FirstTokenAfter, answer.Duration)
		})
	}
}

func TestTurnAgentIDChangesWithToolDescription(t *testing.T) {
	agentID := func(description string) uuid.UUID {
		reg := crux.NewToolsRegistry()
		crux.RegisterToolWithRegistry(reg, "weather", description, func(ctx context.Context, in cityArgs) (string, *crux.StateDelta, error) {
			return "", nil, nil
		})
		mock := cruxtest.NewMock()
		mock.Expect().ReturnText("ok")
		s := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, crux.OpenAIGPT5_4, crux.WithToolsRegistry([]string{"weather"}, reg)), crux.WithHTTPClient(mock.Client())))
		_, err := s.Run(t.Context(), "hi")
		require.NoError(t, err)
		return s.Logs()[2].Turn.AgentID
	}
	require.Equal(t, agentID("Weather for a city"), agentID("Weather for a city"))
	require.NotEqual(t, agentID("Weather for a city"), agentID("Weather anywhere"))
}

func TestApprovalIsRecordedInTheLog(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("weather", map[string]any{"city": "Paris"})
	mock.Expect().ReturnText("Sunny.")
	agent := newMockAgent(t, mock, crux.OpenAIGPT5_4,
		crux.WithToolsRegistry([]string{"weather"}, weatherRegistry(crux.WithApprovalNeeded(true))))
	var rec recorder
	s := crux.MustSession(crux.NewSession(t.Context(), agent, crux.WithEntryHandler(rec.handle), crux.WithHTTPClient(mock.Client())))

	_, err := s.Run(t.Context(), "Weather?")
	require.ErrorIs(t, err, crux.ErrApprovalNeeded)
	logs := s.Logs()
	last := logs[len(logs)-1]
	require.Equal(t, crux.KindRunFinished, last.Kind)
	require.Equal(t, crux.RunApprovalNeeded, last.Run.Outcome)
	require.Equal(t, crux.ErrApprovalNeeded.Error(), last.Run.Error)

	pending := s.PendingApprovals()
	require.Len(t, pending, 1)
	require.NoError(t, s.Approve(t.Context(), pending[0].ID))
	require.Equal(t, crux.KindApproval, rec.entries[len(rec.entries)-1].Kind)

	_, err = s.Resume(t.Context())
	require.NoError(t, err)
	logs = s.Logs()
	require.Equal(t, []crux.Kind{
		crux.KindApproval, crux.KindRunStarted, crux.KindToolStarted, crux.KindToolResult,
		crux.KindTurnStarted, crux.KindAssistant, crux.KindRunFinished,
	}, kinds(logs[len(logs)-7:]))
	require.Equal(t, logs, rec.entries)
}

func TestRejectedCallIsDenied(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("weather", map[string]any{"city": "Paris"})
	mock.Expect().ReturnText("Okay.")
	agent := newMockAgent(t, mock, crux.OpenAIGPT5_4,
		crux.WithToolsRegistry([]string{"weather"}, weatherRegistry(crux.WithApprovalNeeded(true))))
	s := crux.MustSession(crux.NewSession(t.Context(), agent, crux.WithHTTPClient(mock.Client())))
	_, err := s.Run(t.Context(), "Weather?")
	require.ErrorIs(t, err, crux.ErrApprovalNeeded)
	require.NoError(t, s.Reject(t.Context(), s.PendingApprovals()[0].ID, "not now"))
	_, err = s.Resume(t.Context())
	require.NoError(t, err)

	result := toolResult(t, s.Logs())
	require.True(t, result.Denied)
	require.Equal(t, "not now", result.Error)
	require.NotContains(t, kinds(s.Logs()), crux.KindToolStarted, "a rejected tool never starts")
}

func TestSubagentEntriesReachParent(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("agent_researcher", map[string]any{"task": "Paris"})
	mock.Expect().ReturnToolCall("weather", map[string]any{"city": "Paris"})
	mock.Expect().ReturnText("brief")
	mock.Expect().ReturnText("final")
	researcher := crux.Must(crux.New("researcher", crux.OpenAIGPT5_4,
		crux.WithToolsRegistry([]string{"weather"}, weatherRegistry())))
	agent := newMockAgent(t, mock, crux.OpenAIGPT5_4, crux.WithSubAgent(researcher, "Researches"))

	var rec recorder
	s := crux.MustSession(crux.NewSession(t.Context(), agent, crux.WithEntryHandler(rec.handle), crux.WithHTTPClient(mock.Client())))
	out, err := s.Run(t.Context(), "Research Paris")
	require.NoError(t, err)
	require.Equal(t, "final", out)

	var child *crux.Session
	for i, e := range rec.entries {
		if rec.sessions[i] != s {
			child = rec.sessions[i]
			require.Equal(t, "researcher", child.Agent().Name())
		}
		if e.Kind == crux.KindRunFinished && rec.sessions[i] == child {
			require.Equal(t, crux.RunAnswered, e.Run.Outcome)
		}
	}
	require.NotNil(t, child)
	require.Equal(t, child.Logs(), entriesOf(rec, child))
	require.Equal(t, s.Logs(), entriesOf(rec, s))
}

func entriesOf(rec recorder, s *crux.Session) []crux.Entry {
	var out []crux.Entry
	for i, e := range rec.entries {
		if rec.sessions[i] == s {
			out = append(out, e)
		}
	}
	return out
}

func TestParallelSubagentsReportSerially(t *testing.T) {
	mock := cruxtest.NewMock()
	calls := make([]cruxtest.ToolCall, 3)
	for i := range calls {
		calls[i] = cruxtest.ToolCall{Name: "agent_researcher", Args: map[string]any{"task": "Paris"}}
	}
	mock.Expect().ReturnToolCalls(calls...)
	for range calls {
		mock.Expect().ReturnText("brief")
	}
	mock.Expect().ReturnText("done")
	researcher := crux.Must(crux.New("researcher", crux.OpenAIGPT5_4))
	agent := newMockAgent(t, mock, crux.OpenAIGPT5_4, crux.WithSubAgent(researcher, "Researches"))

	// The subagents run concurrently; the recorder appends without a lock,
	// so -race reports handler calls that are not serialised.
	var rec recorder
	s := crux.MustSession(crux.NewSession(t.Context(), agent, crux.WithEntryHandler(rec.handle), crux.WithHTTPClient(mock.Client())))
	out, err := s.Run(t.Context(), "go")
	require.NoError(t, err)
	require.Equal(t, "done", out)
	require.Equal(t, s.Logs(), entriesOf(rec, s))
	children := map[*crux.Session]bool{}
	for _, sess := range rec.sessions {
		if sess != s {
			children[sess] = true
		}
	}
	require.Len(t, children, 3)
}

func TestCancelledRunIsRecorded(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("slow", map[string]any{})
	ctx, cancel := context.WithCancel(t.Context())
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "slow", "Slow", func(ctx context.Context, in struct{}) (string, *crux.StateDelta, error) {
		cancel()
		return "done", nil, nil
	})
	agent := newMockAgent(t, mock, crux.OpenAIGPT5_4, crux.WithToolsRegistry([]string{"slow"}, reg))
	s := crux.MustSession(crux.NewSession(t.Context(), agent, crux.WithHTTPClient(mock.Client())))
	_, err := s.Run(ctx, "go")
	require.ErrorIs(t, err, context.Canceled)
	logs := s.Logs()
	last := logs[len(logs)-1]
	require.Equal(t, crux.KindRunFinished, last.Kind)
	require.Equal(t, crux.RunCancelled, last.Run.Outcome)
}

func toolResult(t *testing.T, logs []crux.Entry) *crux.ToolResult {
	t.Helper()
	for _, e := range logs {
		if e.Kind == crux.KindToolResult {
			return e.ToolResult
		}
	}
	t.Fatal("no tool result in the log")
	return nil
}

func TestSeveralEntryHandlers(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("agent_researcher", map[string]any{"task": "Paris"})
	mock.Expect().ReturnText("brief")
	mock.Expect().ReturnText("final")
	researcher := crux.Must(crux.New("researcher", crux.OpenAIGPT5_4))
	agent := newMockAgent(t, mock, crux.OpenAIGPT5_4, crux.WithSubAgent(researcher, "Researches"))

	var order []string
	var first, second recorder
	s := crux.MustSession(crux.NewSession(t.Context(), agent,
		crux.WithEntryHandler(func(ctx context.Context, s *crux.Session, e crux.Entry) {
			order = append(order, "first")
			e.Kind = crux.KindUser // a handler's copy is its own
			first.handle(ctx, s, e)
		}),
		crux.WithEntryHandler(func(ctx context.Context, s *crux.Session, e crux.Entry) {
			order = append(order, "second")
			second.handle(ctx, s, e)
		}),
		crux.WithHTTPClient(mock.Client()),
	))
	_, err := s.Run(t.Context(), "go")
	require.NoError(t, err)
	require.Equal(t, []string{"first", "second"}, order[:2])
	require.Len(t, first.entries, len(second.entries))
	require.Equal(t, s.Logs(), entriesOf(second, s), "both handlers see the parent")
	require.Greater(t, len(second.entries), len(s.Logs()), "and the subagent")
}
