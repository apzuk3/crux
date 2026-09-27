package crux

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

func TestPendingApprovalsLifecycle(t *testing.T) {
	var executionOrder []string

	call1 := &ToolCall{ID: "call_1", Name: "get_weather", Args: json.RawMessage(`{"city": "Paris"}`)}
	call2 := &ToolCall{ID: "call_2", Name: "transfer_funds", Args: json.RawMessage(`{"amount": 100}`)}
	call3 := &ToolCall{ID: "call_3", Name: "delete_user", Args: json.RawMessage(`{"id": 42}`)}

	agent := &Agent{
		tools: []Tool{
			{
				name:           "get_weather",
				approvalNeeded: false,
				invoke: func(ctx context.Context, args json.RawMessage) (string, *StateDelta, error) {
					executionOrder = append(executionOrder, "get_weather")
					return "sunny", nil, nil
				},
			},
			{
				name:           "transfer_funds",
				approvalNeeded: true,
				invoke: func(ctx context.Context, args json.RawMessage) (string, *StateDelta, error) {
					executionOrder = append(executionOrder, "transfer_funds")
					return "funds transferred", nil, nil
				},
			},
			{
				name:           "delete_user",
				approvalNeeded: true,
				invoke: func(ctx context.Context, args json.RawMessage) (string, *StateDelta, error) {
					executionOrder = append(executionOrder, "delete_user")
					return "user deleted", nil, nil
				},
			},
		},
	}
	session, err := NewSession(t.Context(), agent, WithSessionLogs([]Entry{
		{Kind: KindUser, Content: []ContentPart{{Kind: ContentKindText, Text: "start"}}},
		{Kind: KindToolCall, ToolCall: call1},
		{Kind: KindToolCall, ToolCall: call2},
		{Kind: KindToolCall, ToolCall: call3},
	}))
	if err != nil {
		t.Fatalf("NewSession failed: %v", err)
	}

	// 1. Pending approvals should only include call2 and call3 (call1 doesn't need approval)
	pending := session.PendingApprovals()
	if len(pending) != 2 {
		t.Fatalf("expected 2 pending approvals, got %d", len(pending))
	}
	if pending[0].ID != "call_2" || pending[1].ID != "call_3" {
		t.Fatalf("unexpected pending calls: %+v", pending)
	}

	// 2. Attempting to resume before all approvals are decided should fail with ErrApprovalNeeded
	ctx := context.Background()
	if _, err := session.Resume(ctx); !errors.Is(err, ErrApprovalNeeded) {
		t.Fatalf("expected ErrApprovalNeeded, got %v", err)
	}
	if res := session.executeUnexecutedToolCalls(ctx); len(res) != 0 || len(executionOrder) != 0 {
		t.Fatalf("expected 0 tools executed before approvals, got res=%v, order=%v", res, executionOrder)
	}

	// 3. User approves call2
	if err := session.Approve(ctx, "call_2"); err != nil {
		t.Fatalf("Approve call_2 failed: %v", err)
	}

	// Verify approval entry is in logs and no execution occurred yet
	lastEntry := session.logs[len(session.logs)-1]
	if lastEntry.Kind != KindApproval || lastEntry.Approval == nil || !lastEntry.Approval.Approved {
		t.Fatalf("expected KindApproval with Approved=true, got %+v", lastEntry)
	}
	if len(executionOrder) != 0 {
		t.Fatalf("tool should not execute immediately upon Approve: %v", executionOrder)
	}

	// Still 1 pending approval (call3)
	pending = session.PendingApprovals()
	if len(pending) != 1 || pending[0].ID != "call_3" {
		t.Fatalf("expected call_3 pending, got %+v", pending)
	}

	// Still blocked because call3 is not decided
	if _, err := session.Resume(ctx); !errors.Is(err, ErrApprovalNeeded) {
		t.Fatalf("expected ErrApprovalNeeded while call_3 undecided, got %v", err)
	}
	if res := session.executeUnexecutedToolCalls(ctx); len(res) != 0 || len(executionOrder) != 0 {
		t.Fatalf("tool 1 must not execute ahead of undecided tool 3: res=%v, order=%v", res, executionOrder)
	}

	// 4. User rejects call3
	if err := session.Reject(ctx, "call_3", "permission denied by policy"); err != nil {
		t.Fatalf("Reject call_3 failed: %v", err)
	}

	// Verify rejection entry is in logs
	lastEntry = session.logs[len(session.logs)-1]
	if lastEntry.Kind != KindApproval || lastEntry.Approval == nil || lastEntry.Approval.Approved {
		t.Fatalf("expected KindApproval with Approved=false, got %+v", lastEntry)
	}

	// Now 0 pending approvals
	if len(session.PendingApprovals()) != 0 {
		t.Fatalf("expected 0 pending approvals after reject, got %d", len(session.PendingApprovals()))
	}

	// 5. Execute unexecuted tool calls: should execute in model order: call1, call2, call3
	toolResults := session.executeUnexecutedToolCalls(ctx)
	session.logs = append(session.logs, toolResults...)

	// Tool 1 and Tool 2 were executed, Tool 3 was rejected (invoke not called)
	if len(executionOrder) != 2 || executionOrder[0] != "get_weather" || executionOrder[1] != "transfer_funds" {
		t.Fatalf("unexpected execution order: %v", executionOrder)
	}

	// Check tool results in logs
	var results []ToolResult
	for _, e := range session.logs {
		if e.Kind == KindToolResult && e.ToolResult != nil {
			results = append(results, *e.ToolResult)
		}
	}

	if len(results) != 3 {
		t.Fatalf("expected 3 tool results in logs, got %d", len(results))
	}
	if results[0].CallID != "call_1" || results[0].Output != "sunny" {
		t.Fatalf("unexpected call_1 result: %+v", results[0])
	}
	if results[1].CallID != "call_2" || results[1].Output != "funds transferred" {
		t.Fatalf("unexpected call_2 result: %+v", results[1])
	}
	if results[2].CallID != "call_3" || results[2].Error != "permission denied by policy" {
		t.Fatalf("unexpected call_3 rejection result: %+v", results[2])
	}

	// No unexecuted calls remain
	if session.hasUnexecutedToolCalls() {
		t.Fatalf("expected all tool calls to be resolved")
	}
}

func TestProvidersIgnoreKindApproval(t *testing.T) {
	call := &ToolCall{ID: "call_abc", Name: "deploy", Args: json.RawMessage(`{}`)}
	logs := []Entry{
		{Kind: KindUser, Content: []ContentPart{{Kind: ContentKindText, Text: "deploy app"}}},
		{Kind: KindToolCall, ToolCall: call},
		{Kind: KindApproval, Approval: &Approval{CallID: "call_abc", Approved: true}},
		{Kind: KindToolResult, ToolResult: &ToolResult{CallID: "call_abc", Output: "deployed"}},
		{Kind: KindAssistant, Content: []ContentPart{{Kind: ContentKindText, Text: "All done!"}}},
	}

	// OpenAI
	openAIInput, err := toOpenAIResponseInput(logs)
	if err != nil {
		t.Fatalf("toOpenAIResponseInput failed with KindApproval in log: %v", err)
	}
	if len(openAIInput) != 4 { // User, ToolCall, ToolResult, Assistant (Approval skipped)
		t.Fatalf("expected 4 OpenAI input items, got %d", len(openAIInput))
	}

	// Anthropic
	anthropicMsgs, err := toAnthropicMessages(logs)
	if err != nil {
		t.Fatalf("toAnthropicMessages failed with KindApproval in log: %v", err)
	}
	if len(anthropicMsgs) == 0 {
		t.Fatalf("expected non-empty Anthropic messages")
	}

	// Gemini
	geminiContents, err := toGeminiContents(logs)
	if err != nil {
		t.Fatalf("toGeminiContents failed with KindApproval in log: %v", err)
	}
	if len(geminiContents) == 0 {
		t.Fatalf("expected non-empty Gemini contents")
	}
}

func TestForkRetainsKindApproval(t *testing.T) {
	call := &ToolCall{ID: "call_1", Name: "op", Args: json.RawMessage(`{"k":"v"}`)}
	agent := &Agent{
		Provider: ProviderOpenAI,
	}
	session, err := NewSession(t.Context(), agent, WithSessionLogs([]Entry{
		{Kind: KindUser, Content: []ContentPart{{Kind: ContentKindText, Text: "hi"}}},
		{Kind: KindToolCall, ToolCall: call},
		{Kind: KindApproval, Approval: &Approval{CallID: "call_1", Approved: true}},
		{Kind: KindToolResult, ToolResult: &ToolResult{CallID: "call_1", Output: "done"}},
	}))
	if err != nil {
		t.Fatalf("NewSession failed: %v", err)
	}

	fork, err := session.Fork(t.Context())
	if err != nil {
		t.Fatalf("session.Fork failed: %v", err)
	}
	if len(fork.logs) != 4 {
		t.Fatalf("expected 4 logs in fork, got %d", len(fork.logs))
	}
	if fork.logs[2].Kind != KindApproval {
		t.Fatalf("expected entry 2 to be KindApproval, got %v", fork.logs[2].Kind)
	}
	if fork.logs[2].Approval == nil || fork.logs[2].Approval.CallID != "call_1" || !fork.logs[2].Approval.Approved {
		t.Fatalf("unexpected Approval in forked approval entry: %+v", fork.logs[2].Approval)
	}
}

func TestApproveRejectHonorContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	call := &ToolCall{ID: "call_1", Name: "action", Args: json.RawMessage(`{}`)}
	agent := &Agent{
		tools: []Tool{{name: "action", approvalNeeded: true}},
	}
	session, err := NewSession(context.Background(), agent, WithSessionLogs([]Entry{
		{Kind: KindUser, Content: []ContentPart{{Kind: ContentKindText, Text: "hi"}}},
		{Kind: KindToolCall, ToolCall: call},
	}))
	if err != nil {
		t.Fatalf("NewSession failed: %v", err)
	}

	if err := session.Approve(ctx, "call_1"); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled from Approve, got %v", err)
	}

	if err := session.Reject(ctx, "call_1", "reason"); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled from Reject, got %v", err)
	}
}

func TestResumeIdempotentWhenAlreadyCompleted(t *testing.T) {
	agent := &Agent{}
	session, err := NewSession(t.Context(), agent, WithSessionLogs([]Entry{
		{Kind: KindUser, Content: []ContentPart{{Kind: ContentKindText, Text: "hello"}}},
		{Kind: KindAssistant, Content: []ContentPart{{Kind: ContentKindText, Text: "world"}}},
	}))
	if err != nil {
		t.Fatalf("NewSession failed: %v", err)
	}

	out, err := session.Resume(context.Background())
	if err != nil {
		t.Fatalf("expected Resume on completed agent to succeed, got %v", err)
	}
	if out != "world" {
		t.Fatalf("expected Resume to return %q, got %q", "world", out)
	}
}
