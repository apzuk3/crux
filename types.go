package crux

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// UserLocation provides optional geographic context for provider searches.
// Providers use the fields they support; empty strings and nil coordinates are unset.
type UserLocation struct {
	Country   string   // ISO 3166-1 alpha-2 country code, e.g. "GB"
	City      string   // City name, e.g. "London"
	Region    string   // State, province, or region name
	Timezone  string   // IANA timezone, e.g. "Europe/London"
	Latitude  *float64 // Degrees in [-90, 90]; supply together with Longitude
	Longitude *float64 // Degrees in [-180, 180]; pointers preserve valid zero coordinates
}

// ---------- log entries ----------

type Kind uint8

const (
	KindUser Kind = iota + 1
	KindAssistant
	KindReasoning
	KindToolCall
	KindToolResult
	KindStateDelta
	_                // reserved for conversation compaction
	KindProviderTool // server-executed tool event retained in Opaque
	KindApproval
)

type ContentKind string

const (
	ContentKindText    ContentKind = "text"
	ContentKindRefusal ContentKind = "refusal"
)

type ContentPart struct {
	Kind ContentKind `json:"kind"`
	Text string      `json:"text"`
}

type Entry struct {
	Seq      uint64        `json:"seq"`
	At       time.Time     `json:"at"`
	Duration time.Duration `json:"duration,omitempty"`
	Kind     Kind          `json:"kind"`

	Content []ContentPart `json:"content,omitempty"` // portable user or assistant content

	Reasoning  *Reasoning  `json:"reasoning,omitempty"`
	ToolCall   *ToolCall   `json:"tool_call,omitempty"`
	ToolResult *ToolResult `json:"tool_result,omitempty"`
	Delta      *StateDelta `json:"delta,omitempty"`
	Approval   *Approval   `json:"approval,omitempty"`

	Opaque map[string][]byte `json:"opaque,omitempty"` // provider adornments; dropped on provider switch
	Usage  *Usage            `json:"usage,omitempty"`  // tokens; never affects replay
}

func NewUserEntry(input any) (Entry, error) {
	var text string
	switch v := any(input).(type) {
	case string:
		text = v
	case []byte:
		text = string(v)
	case json.RawMessage:
		text = string(v)
	case fmt.Stringer:
		text = v.String()
	default:
		raw, err := json.Marshal(input)
		if err != nil {
			return Entry{}, fmt.Errorf("unsupported user input type %T: %w", input, err)
		}
		text = string(raw)
	}
	return Entry{
		At:      time.Now().UTC(),
		Kind:    KindUser,
		Content: []ContentPart{{Kind: ContentKindText, Text: text}},
	}, nil
}

// Text returns the concatenated textual content, excluding refusals and other
// non-text parts.
func (e Entry) Text() string {
	var text strings.Builder
	for _, part := range e.Content {
		if part.Kind == ContentKindText {
			text.WriteString(part.Text)
		}
	}
	return text.String()
}

// HiddenFromModel reports whether the entry is an internal bookkeeping event
// (like state deltas or user approvals) that is not sent to LLM providers.
func (e Entry) HiddenFromModel() bool {
	return e.Kind == KindStateDelta || e.Kind == KindApproval
}

type Reasoning struct {
	Summary string `json:"summary,omitempty"` // plaintext, if the provider gives one
}

type ToolCall struct {
	ID   string          `json:"id"` // provider-issued call ID
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
	// Agent names the agent that made the call. It is set on the calls
	// PendingApprovals returns, which include calls made by subagents.
	Agent string `json:"agent,omitempty"`
}

// normalizeToolArgs keeps arguments that are not valid JSON as a JSON string,
// so the entry can still be stored and the tool reports the problem to the model.
func normalizeToolArgs(raw string) json.RawMessage {
	if raw == "" || json.Valid([]byte(raw)) {
		return json.RawMessage(raw)
	}
	quoted, _ := json.Marshal(raw)
	return quoted
}

// rawArgs returns the arguments as the model sent them, undoing normalizeToolArgs.
func (c *ToolCall) rawArgs() string {
	var raw string
	if len(c.Args) > 0 && c.Args[0] == '"' && json.Unmarshal(c.Args, &raw) == nil {
		return raw
	}
	return string(c.Args)
}

// objectArgs returns the arguments as a JSON object, for providers that
// reject anything else when history is replayed.
func (c *ToolCall) objectArgs() json.RawMessage {
	if trimmed := bytes.TrimSpace(c.Args); len(trimmed) > 0 && trimmed[0] == '{' {
		return c.Args
	}
	return json.RawMessage(`{}`)
}

type ToolResult struct {
	CallID string `json:"call_id"` // references ToolCall.ID
	Output string `json:"output,omitempty"`
	Error  string `json:"error,omitempty"` // set on failure or decline; still owed to the model
}

type StateDelta struct {
	By     string         `json:"by"` // tool that wrote it
	Set    map[string]any `json:"set,omitempty"`
	Delete []string       `json:"delete,omitempty"`
}

type Approval struct {
	CallID   string `json:"call_id"`
	Approved bool   `json:"approved"`
	Reason   string `json:"reason,omitempty"`
}

// Usage counts the tokens of one model response, the same way on every
// provider. InputTokens is the total prompt size, including tokens read from
// or written to the provider's cache; CacheReadTokens and CacheWriteTokens are
// the parts of it that were cache hits and cache writes. OutputTokens includes
// reasoning tokens.
//
// Provider documentation for the raw counts:
//   - Anthropic: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
//   - OpenAI: https://platform.openai.com/docs/guides/prompt-caching
//   - Gemini: https://ai.google.dev/api/generate-content#UsageMetadata
type Usage struct {
	InputTokens      int `json:"input_tokens"`
	OutputTokens     int `json:"output_tokens"`
	CacheReadTokens  int `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
}
