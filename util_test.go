package crux

import (
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

func TestRedactURLSecretsUserinfo(t *testing.T) {
	// "%41" decodes to "A", so the parsed and the raw userinfo differ.
	raw := "https://user:p%41ss@example.com/v1"
	err := errors.New("post https://user:pAss@example.com/v1: boom (https://user:p%41ss@example.com/v1)")

	redacted := redactURLSecrets(err, raw)
	require.Equal(t, "post https://redacted@example.com/v1: boom (https://redacted@example.com/v1)", redacted.Error())
	require.ErrorIs(t, redacted, err, "the original error stays reachable")
	var inner *redactedError
	require.ErrorAs(t, redacted, &inner)
	require.Same(t, err, inner.Unwrap())
}

func TestRedactURLSecretsQuery(t *testing.T) {
	raw := "https://example.com/v1?key=secret value&empty=&other=x"
	err := errors.New("get ?key=secret+value, ?key=secret value, ?empty=, other=x")

	redacted := redactURLSecrets(err, raw)
	require.Equal(t, "get ?key=redacted, ?key=redacted, ?empty=, other=redacted", redacted.Error())
	require.ErrorIs(t, redacted, err)
}

func TestRedactURLSecretsLeavesCleanErrors(t *testing.T) {
	err := errors.New("boom")
	require.Same(t, err, redactURLSecrets(err, "https://example.com/v1"), "nothing to redact")
	require.Same(t, err, redactURLSecrets(err, "https://user:pass@example.com/v1"), "no secret in the message")
	require.Same(t, err, redactURLSecrets(err, "://bad"), "unparsable base URL")
}

func TestCloneEntriesIsDeep(t *testing.T) {
	original := []Entry{
		{
			Seq: 1, At: time.Now(), Kind: KindUser,
			Content:   []ContentPart{{Kind: ContentKindText, Text: "hi"}, {Kind: ContentKindFile, Data: []byte("abc")}},
			Reasoning: &Reasoning{Summary: "thinking"},
			ToolCall:  &ToolCall{ID: "c1", Name: "lookup", Args: []byte(`{"a":1}`)},
			ToolResult: &ToolResult{
				CallID: "c1", Output: "ok",
			},
			Delta: &StateDelta{
				By:     "lookup",
				Set:    map[string]any{"nested": map[string]any{"list": []any{1, "x"}}, "raw": []byte("raw")},
				Delete: []string{"gone"},
			},
			Approval:   &Approval{CallID: "c1", Approved: true},
			Usage:      &Usage{InputTokens: 1},
			Run:        &RunStatus{Outcome: RunAnswered},
			Turn:       &TurnInfo{AgentID: uuid.New(), Model: "m"},
			Response:   &ResponseInfo{ID: "r1"},
			Compaction: &Compaction{Through: 1},
			Opaque:     map[string][]byte{"k": []byte("v")},
		},
		{Seq: 2, Kind: KindAssistant},
	}
	snapshot := cloneEntries(original)

	cloned := cloneEntries(original)
	cloned[0].Content[0].Text = "changed"
	cloned[0].Reasoning.Summary = "changed"
	cloned[0].ToolCall.Name = "changed"
	cloned[0].ToolCall.Args[2] = 'z'
	cloned[0].ToolResult.Output = "changed"
	cloned[0].Delta.By = "changed"
	cloned[0].Delta.Set["new"] = true
	cloned[0].Delta.Set["nested"].(map[string]any)["list"].([]any)[0] = 99
	cloned[0].Delta.Set["raw"].([]byte)[0] = 'z'
	cloned[0].Delta.Delete[0] = "changed"
	cloned[0].Approval.Approved = false
	cloned[0].Usage.InputTokens = 99
	cloned[0].Run.Outcome = RunFailed
	cloned[0].Turn.Model = "changed"
	cloned[0].Response.ID = "changed"
	cloned[0].Opaque["k"][0] = 'z'
	cloned[0].Opaque["new"] = []byte("x")
	cloned[1].Kind = KindUser

	require.Equal(t, snapshot, original, "changing the copy must not touch the original")
	require.Equal(t, "hi", original[0].Content[0].Text)
	require.Equal(t, json.RawMessage(`{"a":1}`), original[0].ToolCall.Args)
	require.Equal(t, 1, original[0].Delta.Set["nested"].(map[string]any)["list"].([]any)[0])
	require.Equal(t, []byte("raw"), original[0].Delta.Set["raw"])
	require.Equal(t, []byte("v"), original[0].Opaque["k"])
	require.NotContains(t, original[0].Opaque, "new")

	require.Nil(t, cloneEntries(nil))
	var empty Entry
	require.Equal(t, []Entry{empty}, cloneEntries([]Entry{empty}), "nil pointers stay nil")
}

func TestDecodeIntoDuration(t *testing.T) {
	var out struct {
		Took time.Duration `json:"took"`
	}
	if err := decodeInto("```json\n{\"took\":\"1m30s\"}\n```", &out); err != nil {
		t.Fatal(err)
	}
	if out.Took != 90*time.Second {
		t.Fatalf("Took = %v", out.Took)
	}
}
