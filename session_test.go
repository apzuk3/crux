package crux

import (
	"context"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestNewSessionPersistsNormalisedSeededLogs(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	user, err := NewUserEntry("hello")
	require.NoError(t, err)
	user.At = time.Time{}
	seeded := []Entry{
		user,
		{Seq: 0, Kind: KindAssistant, Content: []ContentPart{{Kind: ContentKindText, Text: "hi"}}},
		{Seq: 5, Kind: KindUser, At: time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC), Content: []ContentPart{{Kind: ContentKindText, Text: "more"}}},
		{Seq: 2, Kind: KindAssistant, Content: []ContentPart{{Kind: ContentKindText, Text: "ok"}}},
	}

	session, err := NewSession(ctx, Must(New("support", OpenAIGPT4)), WithStore(store), WithSessionLogs(seeded))
	require.NoError(t, err)

	stored, err := store.Get(ctx, session.id)
	require.NoError(t, err)
	require.Len(t, stored, 4)
	var seqs []uint64
	for _, e := range stored {
		seqs = append(seqs, e.Seq)
		require.False(t, e.At.IsZero(), "entry %d has no time", e.Seq)
	}
	require.Equal(t, []uint64{1, 2, 5, 6}, seqs, "sequence numbers are made increasing")
	require.Equal(t, seeded[2].At, stored[2].At, "a given time is kept")
	require.Equal(t, stored, session.logs)
	require.Zero(t, seeded[0].Seq, "the caller's entries are not changed")

	_, err = NewSession(ctx, session.agent, WithStore(store), WithSessionID(session.id), WithSessionLogs(seeded))
	require.ErrorContains(t, err, "already has history")
}

// toolCallLogs seeds a user message followed by one tool call entry per call,
// which is what executeUnexecutedToolCalls runs.
func toolCallLogs(calls ...ToolCall) []Entry {
	user, _ := NewUserEntry("go")
	logs := []Entry{user}
	for _, call := range calls {
		call := call
		logs = append(logs, Entry{Kind: KindToolCall, ToolCall: &call})
	}
	return logs
}

func resultByCall(t *testing.T, entries []Entry) map[string]ToolResult {
	t.Helper()
	results := make(map[string]ToolResult)
	for _, e := range entries {
		if e.Kind == KindToolResult {
			require.NotNil(t, e.ToolResult)
			results[e.ToolResult.CallID] = *e.ToolResult
		}
	}
	return results
}

func TestExecuteUnexecutedToolCallsRunsParallelLaneInCallOrder(t *testing.T) {
	reg := NewToolsRegistry()
	fastRan := make(chan struct{})
	RegisterToolWithRegistry(reg, "slow", "", func(ctx context.Context, in struct{}) (string, *StateDelta, error) {
		// Finishes after fast, so results in call order prove reordering.
		select {
		case <-fastRan:
		case <-ctx.Done():
			return "", nil, ctx.Err()
		}
		return "slow", nil, nil
	})
	RegisterToolWithRegistry(reg, "fast", "", func(ctx context.Context, in struct{}) (string, *StateDelta, error) {
		close(fastRan)
		return "fast", nil, nil
	})
	agent := Must(New("support", OpenAIGPT4, WithToolsRegistry([]string{"slow", "fast"}, reg)))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	session := MustSession(NewSession(ctx, agent, WithSessionLogs(toolCallLogs(
		ToolCall{ID: "c1", Name: "slow", Args: []byte(`{}`)},
		ToolCall{ID: "c2", Name: "fast", Args: []byte(`{}`)},
		ToolCall{ID: "c3", Name: "slow", Args: []byte(`{}`)},
	))))

	results, err := session.executeUnexecutedToolCalls(ctx)
	require.NoError(t, err)
	require.Len(t, results, 3)
	var order []string
	for _, e := range results {
		require.Equal(t, KindToolResult, e.Kind)
		order = append(order, e.ToolResult.CallID)
		require.Empty(t, e.ToolResult.Error)
	}
	require.Equal(t, []string{"c1", "c2", "c3"}, order)
	require.Equal(t, "slow", results[0].ToolResult.Output)
	require.Equal(t, "fast", results[1].ToolResult.Output)
	require.True(t, session.hasUnexecutedToolCalls(), "results are returned, not appended")
}

func TestExecuteUnexecutedToolCallsThreadsStateThroughSequentialLane(t *testing.T) {
	reg := NewToolsRegistry()
	var mu sync.Mutex
	var running, overlap int
	RegisterToolWithRegistry(reg, "step", "", func(ctx context.Context, in struct{}) (string, *StateDelta, error) {
		mu.Lock()
		running++
		overlap = max(overlap, running)
		mu.Unlock()
		defer func() { mu.Lock(); running--; mu.Unlock() }()
		time.Sleep(5 * time.Millisecond)
		state, _ := StateFromContext(ctx)
		count, _ := state["count"].(int)
		return strconv.Itoa(count + 1), &StateDelta{Set: map[string]any{"count": count + 1}}, nil
	}, WithSequential())
	RegisterToolWithRegistry(reg, "peek", "", func(ctx context.Context, in struct{}) (string, *StateDelta, error) {
		state, _ := StateFromContext(ctx)
		count, _ := state["count"].(int)
		return strconv.Itoa(count), nil, nil
	})
	agent := Must(New("support", OpenAIGPT4, WithToolsRegistry([]string{"step", "peek"}, reg)))
	ctx := context.Background()
	calls := toolCallLogs(
		ToolCall{ID: "s1", Name: "step", Args: []byte(`{}`)},
		ToolCall{ID: "p1", Name: "peek", Args: []byte(`{}`)},
		ToolCall{ID: "s2", Name: "step", Args: []byte(`{}`)},
		ToolCall{ID: "s3", Name: "step", Args: []byte(`{}`)},
	)
	// The delta precedes the calls, so every call starts from count 10.
	logs := []Entry{calls[0], {Kind: KindStateDelta, Delta: &StateDelta{By: "seed", Set: map[string]any{"count": 10}}}}
	logs = append(logs, calls[1:]...)
	session := MustSession(NewSession(ctx, agent, WithSessionLogs(logs)))

	results, err := session.executeUnexecutedToolCalls(ctx)
	require.NoError(t, err)
	outputs := resultByCall(t, results)
	require.Equal(t, "11", outputs["s1"].Output)
	require.Equal(t, "12", outputs["s2"].Output, "each sequential call sees the delta of the one before it")
	require.Equal(t, "13", outputs["s3"].Output)
	require.Equal(t, "10", outputs["p1"].Output, "a parallel call sees the snapshot taken before the turn")
	require.Equal(t, 1, overlap, "sequential calls never overlap")

	var kinds []Kind
	for _, e := range results {
		kinds = append(kinds, e.Kind)
	}
	require.Equal(t, []Kind{KindToolResult, KindStateDelta, KindToolResult, KindToolResult, KindStateDelta, KindToolResult, KindStateDelta}, kinds)
	require.Equal(t, "step", results[1].Delta.By)
}

func TestExecuteUnexecutedToolCallsAnswersRejectedCallsWithoutRunning(t *testing.T) {
	reg := NewToolsRegistry()
	var ran []string
	var mu sync.Mutex
	RegisterToolWithRegistry(reg, "refund", "", func(ctx context.Context, in struct{}) (string, *StateDelta, error) {
		mu.Lock()
		defer mu.Unlock()
		ran = append(ran, "refund")
		return "done", nil, nil
	}, WithApprovalNeeded(true))
	agent := Must(New("support", OpenAIGPT4, WithToolsRegistry([]string{"refund"}, reg)))
	ctx := context.Background()
	logs := toolCallLogs(
		ToolCall{ID: "r1", Name: "refund", Args: []byte(`{}`)},
		ToolCall{ID: "r2", Name: "refund", Args: []byte(`{}`)},
		ToolCall{ID: "r3", Name: "refund", Args: []byte(`{}`)},
	)
	logs = append(logs,
		Entry{Kind: KindApproval, Approval: &Approval{CallID: "r1", Approved: false, Reason: "not today"}},
		Entry{Kind: KindApproval, Approval: &Approval{CallID: "r2", Approved: false}},
		Entry{Kind: KindApproval, Approval: &Approval{CallID: "r3", Approved: true}},
	)
	session := MustSession(NewSession(ctx, agent, WithSessionLogs(logs)))
	require.Empty(t, session.PendingApprovals())

	results, err := session.executeUnexecutedToolCalls(ctx)
	require.NoError(t, err)
	byCall := resultByCall(t, results)
	require.Equal(t, ToolResult{CallID: "r1", Error: "not today", Denied: true}, byCall["r1"])
	require.Equal(t, ToolResult{CallID: "r2", Error: "tool execution declined by user", Denied: true}, byCall["r2"])
	require.Equal(t, ToolResult{CallID: "r3", Output: "done"}, byCall["r3"])
	require.Equal(t, []string{"refund"}, ran, "only the approved call runs")
	require.Equal(t, "r1", results[0].ToolResult.CallID, "results keep the model's order")
	require.False(t, results[0].At.IsZero(), "denied results are timestamped")
}

func TestExecuteUnexecutedToolCallsRecordsToolStartedBeforeRunning(t *testing.T) {
	reg := NewToolsRegistry()
	var session *Session
	var mu sync.Mutex
	startedWhenRun := make(map[string]bool)
	RegisterToolWithRegistry(reg, "lookup", "", func(ctx context.Context, in struct{ ID string }) (string, *StateDelta, error) {
		// The session is not written while the tools run, so the log can be read.
		for _, e := range session.logs {
			if e.Kind == KindToolStarted && e.ToolCall.ID == in.ID {
				mu.Lock()
				startedWhenRun[in.ID] = true
				mu.Unlock()
			}
		}
		return "ok", nil, nil
	})
	agent := Must(New("support", OpenAIGPT4, WithToolsRegistry([]string{"lookup"}, reg)))
	ctx := context.Background()
	var seen []Kind
	session = MustSession(NewSession(ctx, agent,
		WithSessionLogs(toolCallLogs(
			ToolCall{ID: "l1", Name: "lookup", Args: []byte(`{"ID":"l1"}`)},
			ToolCall{ID: "l2", Name: "lookup", Args: []byte(`{"ID":"l2"}`)},
		)),
		WithEntryHandler(func(_ context.Context, _ *Session, e Entry) { seen = append(seen, e.Kind) }),
	))

	results, err := session.executeUnexecutedToolCalls(ctx)
	require.NoError(t, err)
	require.Len(t, results, 2)
	require.Equal(t, map[string]bool{"l1": true, "l2": true}, startedWhenRun)
	require.Equal(t, []Kind{KindToolStarted, KindToolStarted}, seen, "starts are written in one batch, results are returned to the caller")

	var started []string
	for _, e := range session.logs {
		if e.Kind == KindToolStarted {
			started = append(started, e.ToolCall.ID+"/"+e.ToolCall.Name)
		}
	}
	require.Equal(t, []string{"l1/lookup", "l2/lookup"}, started)
	require.Equal(t, uint64(5), session.logs[len(session.logs)-1].Seq)
	require.True(t, session.hasUnexecutedToolCalls(), "the calls stay open until the caller appends the results")

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	results, err = session.executeUnexecutedToolCalls(cancelled)
	require.NoError(t, err)
	require.Empty(t, results, "nothing runs on a cancelled context")
	require.Len(t, session.logs, 5, "and nothing more is recorded")
}

func TestExecuteUnexecutedToolCallsDoesNothingWhileApprovalsPend(t *testing.T) {
	reg := NewToolsRegistry()
	RegisterToolWithRegistry(reg, "refund", "", func(ctx context.Context, in struct{}) (string, *StateDelta, error) {
		return "done", nil, nil
	}, WithApprovalNeeded(true))
	agent := Must(New("support", OpenAIGPT4, WithToolsRegistry([]string{"refund"}, reg)))
	ctx := context.Background()
	session := MustSession(NewSession(ctx, agent, WithSessionLogs(toolCallLogs(ToolCall{ID: "r1", Name: "refund", Args: []byte(`{}`)}))))
	require.Len(t, session.PendingApprovals(), 1)

	results, err := session.executeUnexecutedToolCalls(ctx)
	require.NoError(t, err)
	require.Nil(t, results)
	require.Len(t, session.logs, 2)
}
