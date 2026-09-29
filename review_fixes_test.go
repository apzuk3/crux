package crux

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/invopop/jsonschema"
	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"
)

func TestReusedToolCallIDRunsAgain(t *testing.T) {
	reg := NewToolsRegistry()
	var runs int
	RegisterToolWithRegistry(reg, "noop", "Does nothing", func(ctx context.Context, in struct{}) (string, *StateDelta, error) {
		runs++
		return "ok", nil, nil
	})
	RegisterToolWithRegistry(reg, "guarded", "Needs approval", func(ctx context.Context, in struct{}) (string, *StateDelta, error) {
		runs++
		return "ok", nil, nil
	}, WithApprovalNeeded(true))
	agent := Must(New("a", "m", WithProvider(ProviderOpenAI), WithAPIKey("k"), WithToolsRegistry([]string{"noop", "guarded"}, reg)))

	user, err := NewUserEntry("hi")
	if err != nil {
		t.Fatal(err)
	}
	call := func(name string) Entry {
		return Entry{Kind: KindToolCall, ToolCall: &ToolCall{ID: "call_0", Name: name, Args: json.RawMessage(`{}`)}}
	}
	result := Entry{Kind: KindToolResult, ToolResult: &ToolResult{CallID: "call_0", Output: "ok"}}

	// Some OpenAI-compatible servers number calls per turn, so a later turn
	// reuses an earlier ID. The later call still needs to run.
	session := MustSession(NewSession(t.Context(), agent, WithSessionLogs([]Entry{user, call("noop"), result, call("noop")})))
	if !session.hasUnexecutedToolCalls() {
		t.Fatal("reused call ID is treated as answered")
	}
	if got := session.executeUnexecutedToolCalls(t.Context()); len(got) != 1 || got[0].ToolResult.CallID != "call_0" {
		t.Fatalf("results = %+v", got)
	}
	if runs != 1 {
		t.Fatalf("tool ran %d times, want 1", runs)
	}

	// An approval given for an earlier call with the same ID does not
	// approve the new one.
	approved := Entry{Kind: KindApproval, Approval: &Approval{CallID: "call_0", Approved: true}}
	session = MustSession(NewSession(t.Context(), agent, WithSessionLogs([]Entry{user, call("guarded"), approved, result, call("guarded")})))
	if pending := session.PendingApprovals(); len(pending) != 1 || pending[0].ID != "call_0" {
		t.Fatalf("pending = %+v", pending)
	}
	if err := session.Approve(t.Context(), "call_0"); err != nil {
		t.Fatal(err)
	}
	if got := session.executeUnexecutedToolCalls(t.Context()); len(got) != 1 {
		t.Fatalf("results after approval = %+v", got)
	}
}

func TestToolStateDeltaIsCopied(t *testing.T) {
	reg := NewToolsRegistry()
	kept := map[string]any{"k": "v1"}
	RegisterToolWithRegistry(reg, "set", "Sets state", func(ctx context.Context, in struct{}) (string, *StateDelta, error) {
		return "ok", &StateDelta{Set: kept}, nil
	})
	agent := Must(New("a", "m", WithProvider(ProviderOpenAI), WithAPIKey("k"), WithToolsRegistry([]string{"set"}, reg)))
	user, _ := NewUserEntry("hi")
	call := Entry{Kind: KindToolCall, ToolCall: &ToolCall{ID: "c1", Name: "set", Args: json.RawMessage(`{}`)}}
	session := MustSession(NewSession(t.Context(), agent, WithSessionLogs([]Entry{user, call})))
	if err := session.appendLogs(t.Context(), session.executeUnexecutedToolCalls(t.Context())...); err != nil {
		t.Fatal(err)
	}
	kept["k"] = "v2"
	if got := session.StateSnapshot()["k"]; got != "v1" {
		t.Fatalf("state changed after the tool returned: %v", got)
	}
}

func TestDynamicMapOutputSchema(t *testing.T) {
	type Item struct {
		Name string `json:"name"`
	}
	type Out struct {
		Meta  map[string]any  `json:"meta"`
		Items map[string]Item `json:"items"`
	}
	agent := Must(New("a", "m", WithProvider(ProviderGoogle), WithOutputSchemaFrom[Out]()))

	// Closing {"type":"object"} would let the model return only {}.
	for _, provider := range []Provider{ProviderOpenAI, ProviderAnthropic} {
		if _, err := wireSchemaFor(agent.outputSchema, provider); err == nil || !strings.Contains(err.Error(), "dynamic map") {
			t.Fatalf("%s: err = %v, want a dynamic map error", provider, err)
		}
	}

	wire, err := wireSchemaFor(agent.outputSchema, ProviderGoogle)
	if err != nil {
		t.Fatal(err)
	}
	props := wire["properties"].(map[string]any)
	if _, closed := props["meta"].(map[string]any)["additionalProperties"]; closed {
		t.Fatalf("gemini meta = %v, want an open map", props["meta"])
	}
	values := props["items"].(map[string]any)["additionalProperties"].(map[string]any)
	if values["$ref"] == nil {
		// Inline map values must be closed too, not only $defs.
		if values["additionalProperties"] != false {
			t.Fatalf("gemini map values = %v, want closed objects", values)
		}
	} else if item := wire["$defs"].(map[string]any)["Item"].(map[string]any); item["additionalProperties"] != false {
		t.Fatalf("gemini map values = %v, want closed objects", item)
	}
	inline, err := wireSchemaFor(schemaFrom(t, `{"type":"object","properties":{"m":{"type":"object","additionalProperties":{"type":"object","properties":{"n":{"type":"string"}}}}}}`), ProviderGoogle)
	if err != nil {
		t.Fatal(err)
	}
	if v := inline["properties"].(map[string]any)["m"].(map[string]any)["additionalProperties"].(map[string]any); v["additionalProperties"] != false {
		t.Fatalf("inline map values = %v, want closed objects", v)
	}

	// An empty struct is a closed object, not a map.
	if _, err := wireSchemaFor(schemaFrom(t, `{"type":"object","properties":{"e":{"type":"object","properties":{}}}}`), ProviderOpenAI); err != nil {
		t.Fatalf("empty struct rejected: %v", err)
	}
}

func schemaFrom(t *testing.T, raw string) *jsonschema.Schema {
	t.Helper()
	var schema jsonschema.Schema
	if err := json.Unmarshal([]byte(raw), &schema); err != nil {
		t.Fatal(err)
	}
	return &schema
}

func TestProviderErrorsHideBaseURLCredentials(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":{"message":"bad request","type":"invalid_request_error"}}`)
	}))
	defer server.Close()

	base := strings.Replace(server.URL, "http://", "http://user:s3cr%2Ft-pw@", 1) + "/v1?token=q-secret"
	for _, provider := range []Provider{ProviderXAI, ProviderAnthropic} {
		t.Run(string(provider), func(t *testing.T) {
			agent := Must(New("a", "m", WithProvider(provider), WithBaseURL(base), WithAPIKey("k")))
			session := MustSession(NewSession(t.Context(), agent))
			_, err := session.Run(t.Context(), "hi")
			if err == nil {
				t.Fatal("expected an error")
			}
			for _, secret := range []string{"s3cr", "q-secret", "user:"} {
				if strings.Contains(err.Error(), secret) {
					t.Fatalf("error leaks %q: %v", secret, err)
				}
			}
		})
	}

	inner := errors.New(`POST "http://u:pw@host/v1/x?token=abc": 400`)
	err := redactURLSecrets(inner, "http://u:pw@host/v1?token=abc")
	if got := err.Error(); got != `POST "http://redacted@host/v1/x?token=redacted": 400` {
		t.Fatalf("redacted = %q", got)
	}
	if !errors.Is(err, inner) {
		t.Fatal("redacted error does not wrap the original")
	}
}

func TestGORMStoreAppendsLargeHistory(t *testing.T) {
	store := setupGORMTestDB(t)
	agent := Must(New("a", "m", WithProvider(ProviderOpenAI), WithAPIKey("k")))
	logs := make([]Entry, 9000)
	for i := range logs {
		entry, err := NewUserEntry(fmt.Sprintf("message %d", i))
		if err != nil {
			t.Fatal(err)
		}
		logs[i] = entry
	}
	id := uuid.New()
	if _, err := NewSession(t.Context(), agent, WithStore(store), WithSessionID(id), WithSessionLogs(logs)); err != nil {
		t.Fatal(err)
	}
	stored, err := store.Get(t.Context(), id)
	if err != nil {
		t.Fatal(err)
	}
	if len(stored) != len(logs) {
		t.Fatalf("stored %d entries, want %d", len(stored), len(logs))
	}
}

func TestFilesystemEditFileEdgeCases(t *testing.T) {
	registry, root := newFilesystemRegistry(t, map[string]string{
		"braces.go": "}\n}\n}\n",
		"mixed.txt": "a\r\nb\nc\n",
		"ro.txt":    "keep\n",
	})

	// Overlapping matches are two places, so the edit is ambiguous.
	wantFSError(t, registry, "edit_file", editFileInput{Path: "braces.go", Edits: []fileEdit{{OldText: "}\n}", NewText: "x"}}}, "appears 2 times")

	mustFSTool(t, registry, "edit_file", editFileInput{Path: "mixed.txt", Edits: []fileEdit{{OldText: "b\nc", NewText: "B\nC"}}})
	if data, _ := os.ReadFile(filepath.Join(root, "mixed.txt")); string(data) != "a\r\nB\nC\n" {
		t.Fatalf("mixed edit = %q", data)
	}

	if runtime.GOOS == "windows" {
		return
	}
	if err := os.Chmod(filepath.Join(root, "ro.txt"), 0o444); err != nil {
		t.Fatal(err)
	}
	wantFSError(t, registry, "write_file", writeFileInput{Path: "ro.txt", Content: "changed"}, "read-only")
	wantFSError(t, registry, "edit_file", editFileInput{Path: "ro.txt", Edits: []fileEdit{{OldText: "keep", NewText: "x"}}}, "read-only")
}

func TestGeminiStreamMergesText(t *testing.T) {
	text := func(s string, thought bool) *genai.Part { return &genai.Part{Text: s, Thought: thought} }
	signed := &genai.Part{Text: "c", ThoughtSignature: []byte("sig")}
	call := &genai.Part{FunctionCall: &genai.FunctionCall{Name: "f"}}

	cases := []struct {
		prev, next *genai.Part
		want       bool
	}{
		{text("a", false), text("b", false), true},
		{text("a", false), signed, true},
		{signed, text("d", false), false},
		{text("a", true), text("b", false), false},
		{text("a", false), call, false},
	}
	for i, c := range cases {
		if got := canMergeGeminiText(c.prev, c.next); got != c.want {
			t.Errorf("case %d: merge = %v, want %v", i, got, c.want)
		}
	}
}

func TestOpenAICommentaryIsNotFinalOutput(t *testing.T) {
	raw := `{"type":"message","id":"msg_1","role":"assistant","status":"completed","phase":"commentary","content":[{"type":"output_text","text":"Checking the weather.","annotations":[]}]}`
	var item responses.ResponseOutputItemUnion
	if err := json.Unmarshal([]byte(raw), &item); err != nil {
		t.Fatal(err)
	}
	entry, err := fromOpenAIResponseOutputItemUnion(item)
	if err != nil {
		t.Fatal(err)
	}
	if entry.Kind != KindReasoning || entry.Reasoning.Summary != "Checking the weather." {
		t.Fatalf("entry = %+v", entry)
	}
	// It is replayed as the original assistant message.
	param, err := toOpenAIResponseInputItemUnionParam(entry)
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := json.Marshal(param)
	if err != nil {
		t.Fatal(err)
	}
	if param.OfOutputMessage == nil || !strings.Contains(string(replayed), `"phase":"commentary"`) {
		t.Fatalf("replayed = %s", replayed)
	}
}
