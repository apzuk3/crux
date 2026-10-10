package crux

import (
	"fmt"
	"testing"
)

func TestKindString(t *testing.T) {
	want := map[Kind]string{
		KindUser: "user", KindAssistant: "assistant", KindReasoning: "reasoning", KindToolCall: "tool_call",
		KindToolResult: "tool_result", KindStateDelta: "state_delta", KindCompaction: "compaction",
		KindProviderTool: "provider_tool", KindApproval: "approval", KindRunStarted: "run_started",
		KindRunFinished: "run_finished", KindTurnStarted: "turn_started", KindToolStarted: "tool_started",
	}
	for kind, name := range want {
		if got := kind.String(); got != name {
			t.Errorf("Kind(%d).String() = %q, want %q", uint8(kind), got, name)
		}
	}
	if got := Kind(0).String(); got != "Kind(0)" {
		t.Errorf("unknown kind = %q", got)
	}
	if got := fmt.Sprintf("%d", KindToolCall); got != "4" {
		t.Errorf("%%d must keep printing the number, got %q", got)
	}
}
