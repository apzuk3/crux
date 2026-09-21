package crux

import (
	"context"
	"encoding/json"
	"testing"
	"time"
)

func TestAppendLogsPopulatesSeqAndAt(t *testing.T) {
	agent := &Agent{}

	e1 := Entry{Kind: KindUser, Content: []ContentPart{{Kind: ContentKindText, Text: "hi"}}}
	e2 := Entry{Kind: KindAssistant, Content: []ContentPart{{Kind: ContentKindText, Text: "hello"}}}

	if err := agent.appendLogs(e1, e2); err != nil {
		t.Fatalf("appendLogs failed: %v", err)
	}

	logs := agent.Logs()
	if len(logs) != 2 {
		t.Fatalf("expected 2 logs, got %d", len(logs))
	}

	if logs[0].Seq != 1 {
		t.Errorf("expected Seq 1, got %d", logs[0].Seq)
	}
	if logs[0].At.IsZero() {
		t.Errorf("expected non-zero At")
	}

	if logs[1].Seq != 2 {
		t.Errorf("expected Seq 2, got %d", logs[1].Seq)
	}
	if logs[1].At.IsZero() {
		t.Errorf("expected non-zero At")
	}

	// Appending more should continue the sequence
	e3 := Entry{Kind: KindToolCall, ToolCall: &ToolCall{ID: "c1", Name: "t1"}}
	if err := agent.appendLogs(e3); err != nil {
		t.Fatalf("appendLogs failed: %v", err)
	}

	logs = agent.Logs()
	if len(logs) != 3 {
		t.Fatalf("expected 3 logs, got %d", len(logs))
	}
	if logs[2].Seq != 3 {
		t.Errorf("expected Seq 3, got %d", logs[2].Seq)
	}
}

func TestToolDispatchRecordsDuration(t *testing.T) {
	sleepDuration := 20 * time.Millisecond
	agent := &Agent{
		Tools: []Tool{
			{
				name: "timed_tool",
				invoke: func(ctx context.Context, args json.RawMessage) (string, *StateDelta, error) {
					time.Sleep(sleepDuration)
					return "done", nil, nil
				},
			},
		},
	}

	call := &ToolCall{ID: "call_timed", Name: "timed_tool", Args: json.RawMessage(`{}`)}
	entries, _ := agent.dispatch(context.Background(), call, nil)

	if len(entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(entries))
	}

	res := entries[0]
	if res.Kind != KindToolResult {
		t.Fatalf("expected KindToolResult, got %v", res.Kind)
	}
	if res.ToolResult == nil || res.ToolResult.Output != "done" {
		t.Fatalf("unexpected tool result: %+v", res.ToolResult)
	}
	if res.Duration < sleepDuration {
		t.Errorf("expected Duration >= %v, got %v", sleepDuration, res.Duration)
	}
	if res.At.IsZero() {
		t.Errorf("expected non-zero At")
	}
}

func TestUsageAndDurationClonedOnFork(t *testing.T) {
	agent := &Agent{
		Provider: ProviderOpenAI,
		Model:    "gpt-4o",
		sessionLogs: []Entry{
			{
				Seq:      1,
				At:       time.Now().UTC(),
				Duration: 150 * time.Millisecond,
				Kind:     KindAssistant,
				Content:  []ContentPart{{Kind: ContentKindText, Text: "test"}},
				Usage: &Usage{
					InputTokens:      100,
					OutputTokens:     50,
					CacheReadTokens:  20,
					CacheWriteTokens: 10,
				},
			},
		},
	}

	forked, err := agent.Fork()
	if err != nil {
		t.Fatalf("Fork failed: %v", err)
	}

	forkLogs := forked.Logs()
	if len(forkLogs) != 1 {
		t.Fatalf("expected 1 log in fork, got %d", len(forkLogs))
	}

	fEntry := forkLogs[0]
	if fEntry.Duration != 150*time.Millisecond {
		t.Errorf("expected Duration 150ms, got %v", fEntry.Duration)
	}
	if fEntry.Usage == nil {
		t.Fatal("expected non-nil Usage")
	}
	if fEntry.Usage.InputTokens != 100 || fEntry.Usage.OutputTokens != 50 || fEntry.Usage.CacheReadTokens != 20 || fEntry.Usage.CacheWriteTokens != 10 {
		t.Errorf("unexpected Usage: %+v", fEntry.Usage)
	}

	// Ensure mutation of fork does not affect parent
	fEntry.Usage.InputTokens = 999
	if agent.sessionLogs[0].Usage.InputTokens == 999 {
		t.Errorf("mutating fork usage affected parent")
	}
}
