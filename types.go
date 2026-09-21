package crux

import (
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
	// KindCompaction represents a compacted conversation range.
	// NOTE: Runtime compaction is deferred and is unsupported in alpha.
	KindCompaction
	KindProviderTool // server-executed tool event retained in Opaque
	KindApproval
)

type ContentKind string

const (
	ContentKindText    ContentKind = "text"
	ContentKindRefusal ContentKind = "refusal"
)

const (
	EventAgentStarted = "agent_started"
	EventAgentEnded   = "agent_ended"
	EventTurnStarted  = "turn_started"
	EventTurnEnded    = "turn_ended"
)

type ContentPart struct {
	Kind ContentKind `json:"kind"`
	Text string      `json:"text"`
}

type Entry struct {
	Seq  uint64    `json:"seq"`
	At   time.Time `json:"at"`
	Kind Kind      `json:"kind"`

	Content []ContentPart `json:"content,omitempty"` // portable user or assistant content

	Reasoning  *Reasoning  `json:"reasoning,omitempty"`
	ToolCall   *ToolCall   `json:"tool_call,omitempty"`
	ToolResult *ToolResult `json:"tool_result,omitempty"`
	Delta      *StateDelta `json:"delta,omitempty"`
	Compaction *Compaction `json:"compaction,omitempty"`
	Approval   *Approval   `json:"approval,omitempty"`

	Opaque map[string][]byte `json:"opaque,omitempty"` // provider adornments; dropped on provider switch
	Usage  *Usage            `json:"usage,omitempty"`  // tokens, latency; never affects replay
}

func NewUserEntry(input any) (Entry, error) {
	var text string
	switch v := any(input).(type) {
	case string:
		text = v
	case fmt.Stringer:
		text = v.String()
	default:
		return Entry{}, fmt.Errorf("unsupported user input type %T", input)
	}
	return Entry{
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

// Compaction defines a superseded history range and its summary.
// NOTE: Runtime compaction execution is deferred and is currently
// unsupported in alpha. Compaction entries cannot yet be constructed by or replayed to providers.
type Compaction struct {
	From    uint64 `json:"from"` // superseded range start
	To      uint64 `json:"to"`   // superseded range end
	Summary string `json:"summary,omitempty"`
}

type Usage struct {
	InputTokens  int           `json:"input_tokens"`
	OutputTokens int           `json:"output_tokens"`
	Latency      time.Duration `json:"latency"`
}
