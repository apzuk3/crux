package crux

import (
	"encoding/json"
	"testing"

	"google.golang.org/genai"
)

func TestWalkSchemasVisitsAllNestedSchemas(t *testing.T) {
	raw := `{
		"type": "object",
		"properties": {"a": {"title": "properties"}},
		"patternProperties": {"^x-": {"title": "patternProperties"}},
		"propertyNames": {"title": "propertyNames"},
		"dependentSchemas": {"a": {"title": "dependentSchemas"}},
		"dependencies": {"a": {"title": "dependencies"}, "b": ["a"]},
		"unevaluatedProperties": {"title": "unevaluatedProperties"},
		"additionalProperties": {"title": "additionalProperties"},
		"$defs": {"d": {"title": "$defs"}},
		"definitions": {"d": {"title": "definitions"}},
		"not": {"title": "not"},
		"if": {"title": "if"}, "then": {"title": "then"}, "else": {"title": "else"},
		"anyOf": [{"title": "anyOf"}], "allOf": [{"title": "allOf"}], "oneOf": [{"title": "oneOf"}],
		"items": {"title": "items"},
		"prefixItems": [{"title": "prefixItems"}],
		"additionalItems": {"title": "additionalItems"},
		"unevaluatedItems": {"title": "unevaluatedItems"},
		"contains": {"title": "contains"},
		"contentSchema": {"title": "contentSchema"}
	}`
	var root map[string]any
	if err := json.Unmarshal([]byte(raw), &root); err != nil {
		t.Fatal(err)
	}
	seen := make(map[string]bool)
	if err := walkSchemas(root, func(m map[string]any) error {
		if title, ok := m["title"].(string); ok {
			seen[title] = true
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{
		"properties", "patternProperties", "propertyNames", "dependentSchemas", "dependencies",
		"unevaluatedProperties", "additionalProperties", "$defs", "definitions", "not", "if", "then", "else",
		"anyOf", "allOf", "oneOf", "items", "prefixItems", "additionalItems", "unevaluatedItems", "contains", "contentSchema",
	} {
		if !seen[key] {
			t.Errorf("schema under %q was not visited", key)
		}
	}
}

func TestOpenAISchemaClosesPatternPropertyObjects(t *testing.T) {
	root := map[string]any{
		"type":       "object",
		"properties": map[string]any{"a": map[string]any{"type": "string"}},
		"patternProperties": map[string]any{
			"^x-": map[string]any{"type": "object", "properties": map[string]any{"b": map[string]any{"type": "string"}}},
		},
	}
	adapted, err := adaptOpenAI(root, ProviderOpenAI)
	if err != nil {
		t.Fatal(err)
	}
	nested := adapted["patternProperties"].(map[string]any)["^x-"].(map[string]any)
	if nested["additionalProperties"] != false {
		t.Fatalf("nested object was not adapted: %v", nested)
	}
}

func TestGeminiSignatureOnlyPartsFold(t *testing.T) {
	if !geminiPartEmpty(&genai.Part{Thought: true}) {
		t.Error("a bare thought flag should count as empty")
	}
	text := &genai.Part{Text: "Hello."}
	folded := foldGeminiSignatures([]*genai.Part{text, {}, {ThoughtSignature: []byte("sig")}})
	if len(folded) != 1 || string(folded[0].ThoughtSignature) != "sig" {
		t.Fatalf("folded = %+v", folded)
	}

	// A signature-only part recorded on its own is replayed on the part before it.
	entry := func(part *genai.Part, kind Kind) Entry {
		raw, err := json.Marshal(part)
		if err != nil {
			t.Fatal(err)
		}
		return Entry{Kind: kind, Opaque: map[string][]byte{geminiPartOpaqueKey: raw}}
	}
	user, err := NewUserEntry("Hi")
	if err != nil {
		t.Fatal(err)
	}
	contents, err := toGeminiContents([]Entry{
		user,
		entry(&genai.Part{Text: "Hello."}, KindAssistant),
		entry(&genai.Part{ThoughtSignature: []byte("sig")}, KindReasoning),
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(contents) != 2 || len(contents[1].Parts) != 1 || string(contents[1].Parts[0].ThoughtSignature) != "sig" {
		raw, _ := json.Marshal(contents)
		t.Fatalf("contents = %s", raw)
	}
}

func TestAnthropicStoredBlankTextIsSkipped(t *testing.T) {
	user, err := NewUserEntry("Hi")
	if err != nil {
		t.Fatal(err)
	}
	blank := Entry{Kind: KindAssistant, Content: []ContentPart{{Kind: ContentKindText, Text: "\n\n"}},
		Opaque: map[string][]byte{anthropicContentBlockOpaqueKey: []byte(`{"type":"text","text":"\n\n"}`)}}
	call := Entry{Kind: KindToolCall, ToolCall: &ToolCall{ID: "t1", Name: "f", Args: json.RawMessage(`{}`)},
		Opaque: map[string][]byte{anthropicContentBlockOpaqueKey: []byte(`{"type":"tool_use","id":"t1","name":"f","input":{}}`)}}
	messages, err := toAnthropicMessages([]Entry{user, blank, call})
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 2 || len(messages[1].Content) != 1 || messages[1].Content[0].OfToolUse == nil {
		raw, _ := json.Marshal(messages)
		t.Fatalf("messages = %s", raw)
	}
	// An answer that was only whitespace leaves no message behind.
	messages, err = toAnthropicMessages([]Entry{user, blank})
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 1 {
		raw, _ := json.Marshal(messages)
		t.Fatalf("messages = %s", raw)
	}
}
