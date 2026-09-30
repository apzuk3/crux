package crux

import (
	"context"
	"encoding/json"
	"testing"
)

type sampleInput struct {
	Query string `json:"query"`
	Count int    `json:"count"`
}

type sampleOutput struct {
	Answer string `json:"answer"`
	Score  int    `json:"score"`
}

type sampleStringer struct {
	msg string
}

func (s sampleStringer) String() string {
	return s.msg
}

func TestNewUserEntryTypes(t *testing.T) {
	// String
	e1, err := NewUserEntry("hello")
	if err != nil || e1.Text() != "hello" {
		t.Fatalf("expected 'hello', got %q, err=%v", e1.Text(), err)
	}

	// Bytes
	e2, err := NewUserEntry([]byte("bytes input"))
	if err != nil || e2.Text() != "bytes input" {
		t.Fatalf("expected 'bytes input', got %q, err=%v", e2.Text(), err)
	}

	// RawMessage
	e3, err := NewUserEntry(json.RawMessage(`{"raw":true}`))
	if err != nil || e3.Text() != `{"raw":true}` {
		t.Fatalf("expected raw JSON, got %q, err=%v", e3.Text(), err)
	}

	// Stringer
	e4, err := NewUserEntry(sampleStringer{msg: "from stringer"})
	if err != nil || e4.Text() != "from stringer" {
		t.Fatalf("expected 'from stringer', got %q, err=%v", e4.Text(), err)
	}

	// Struct
	e5, err := NewUserEntry(sampleInput{Query: "test", Count: 3})
	if err != nil {
		t.Fatalf("unexpected error for struct: %v", err)
	}
	var decoded sampleInput
	if err := json.Unmarshal([]byte(e5.Text()), &decoded); err != nil {
		t.Fatalf("failed to decode struct text: %v", err)
	}
	if decoded.Query != "test" || decoded.Count != 3 {
		t.Fatalf("unexpected struct content: %+v", decoded)
	}
}

func TestDecodeInto(t *testing.T) {
	t.Run("nil and non-pointer targets", func(t *testing.T) {
		// RunInto rejects them before it runs anything.
		var s Session
		if err := s.RunInto(t.Context(), "text", nil); err == nil {
			t.Fatal("expected error on nil target")
		}
		var val string
		if err := s.RunInto(t.Context(), "text", val); err == nil {
			t.Fatal("expected error on non-pointer target")
		}
		var nilPtr *string
		if err := s.RunInto(t.Context(), "text", nilPtr); err == nil {
			t.Fatal("expected error on nil pointer target")
		}
	})

	t.Run("string target", func(t *testing.T) {
		var out string
		if err := decodeInto("hello world", &out); err != nil || out != "hello world" {
			t.Fatalf("expected 'hello world', got %q, err=%v", out, err)
		}
	})

	t.Run("byte slice target", func(t *testing.T) {
		var out []byte
		if err := decodeInto("raw bytes", &out); err != nil || string(out) != "raw bytes" {
			t.Fatalf("expected 'raw bytes', got %q, err=%v", string(out), err)
		}
	})

	t.Run("json raw message target", func(t *testing.T) {
		var out json.RawMessage
		if err := decodeInto(`{"k":"v"}`, &out); err != nil || string(out) != `{"k":"v"}` {
			t.Fatalf("expected JSON, got %q, err=%v", string(out), err)
		}
	})

	t.Run("struct target plain json", func(t *testing.T) {
		var out sampleOutput
		input := `{"answer": "42", "score": 100}`
		if err := decodeInto(input, &out); err != nil {
			t.Fatalf("decode failed: %v", err)
		}
		if out.Answer != "42" || out.Score != 100 {
			t.Fatalf("unexpected output: %+v", out)
		}
	})

	t.Run("struct target markdown block with language tag", func(t *testing.T) {
		var out sampleOutput
		input := "```json\n{\"answer\": \"deep thought\", \"score\": 99}\n```"
		if err := decodeInto(input, &out); err != nil {
			t.Fatalf("decode failed: %v", err)
		}
		if out.Answer != "deep thought" || out.Score != 99 {
			t.Fatalf("unexpected output: %+v", out)
		}
	})

	t.Run("struct target single-line code block", func(t *testing.T) {
		var out sampleOutput
		input := "```json {\"answer\": \"quick\", \"score\": 10} ```"
		if err := decodeInto(input, &out); err != nil {
			t.Fatalf("decode failed: %v", err)
		}
		if out.Answer != "quick" || out.Score != 10 {
			t.Fatalf("unexpected output: %+v", out)
		}
	})
}

func TestSessionRunInto(t *testing.T) {
	agent := &Agent{}
	session, err := NewSession(t.Context(), agent, WithSessionLogs([]Entry{
		{Kind: KindUser, Content: []ContentPart{{Kind: ContentKindText, Text: "get answer"}}},
		{Kind: KindAssistant, Content: []ContentPart{{Kind: ContentKindText, Text: `{"answer": "yes", "score": 10}`}}},
	}))
	if err != nil {
		t.Fatalf("NewSession failed: %v", err)
	}

	var out sampleOutput
	if err := session.RunInto(context.Background(), nil, &out); err != nil {
		t.Fatalf("RunInto failed: %v", err)
	}
	if out.Answer != "yes" || out.Score != 10 {
		t.Fatalf("unexpected output: %+v", out)
	}

	var rawStr string
	if err := session.RunInto(context.Background(), nil, &rawStr); err != nil {
		t.Fatalf("RunInto string failed: %v", err)
	}
	if rawStr != `{"answer": "yes", "score": 10}` {
		t.Fatalf("unexpected string output: %q", rawStr)
	}
}
