package crux

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
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

type Session struct {
	ID string

	mu  sync.Mutex
	log []Entry
}

func (c *Session) Append(expect uint64, e ...Entry) (uint64, error) { panic("not implemented") }
func (c *Session) Seq() uint64                                      { panic("not implemented") }
func (c *Session) Log() []Entry                                     { panic("not implemented") }
func (c *Session) Messages() []Entry                                { panic("not implemented") } // Kind User..Reasoning only
func (c *Session) State() map[string]any                            { panic("not implemented") } // fold of all deltas
func (c *Session) StateAt(seq uint64) map[string]any                { panic("not implemented") }
func (c *Session) Pending() []ToolCall                              { panic("not implemented") } // calls with no result
func (c *Session) Fork(atSeq uint64) *Session                       { panic("not implemented") }

// ---------- log entries ----------

type Kind uint8

const (
	KindUser Kind = iota + 1
	KindAssistant
	KindReasoning
	KindToolCall
	KindToolResult
	KindStateDelta
	KindCompaction
	KindProviderTool // server-executed tool event retained in Opaque
)

type ContentKind string

const (
	ContentKindText    ContentKind = "text"
	ContentKindRefusal ContentKind = "refusal"
)

type ContentPart struct {
	Kind ContentKind
	Text string
}

type Entry struct {
	Seq   uint64
	At    time.Time
	Kind  Kind
	Agent string // which agent produced it

	Content []ContentPart // portable user or assistant content

	Reasoning  *Reasoning
	ToolCall   *ToolCall
	ToolResult *ToolResult
	Delta      *StateDelta
	Compaction *Compaction

	Opaque map[string][]byte // provider adornments; dropped on provider switch
	Usage  *Usage            // tokens, latency; never affects replay
}

func NewUserEntry(input any) (Entry, error) {
	var text string
	switch v := any(input).(type) {
	case string:
		text = v
	case fmt.Stringer:
		text = v.String()
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

type Reasoning struct {
	Summary string // plaintext, if the provider gives one
}

type ToolCall struct {
	ID   string // provider-issued call ID
	Name string
	Args json.RawMessage
}

type ToolResult struct {
	CallID string // references ToolCall.ID
	Output string
	Error  string // set on failure or decline; still owed to the model
}

type StateDelta struct {
	By     string // tool that wrote it
	Set    map[string]any
	Delete []string
}

type Compaction struct {
	From, To uint64 // superseded range
	Summary  string
}

type Usage struct {
	InputTokens  int
	OutputTokens int
	Latency      time.Duration
}
