package crux

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/google/uuid"
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
	KindCompaction   // the model no longer sees earlier entries in full; Compaction says how
	KindProviderTool // server-executed tool event retained in Opaque
	KindApproval
	KindRunStarted  // a Run, Resume or Stream began work
	KindRunFinished // Run holds how it ended
	KindTurnStarted // a provider request is about to be sent; Turn describes it
	KindToolStarted // a tool is about to run; ToolCall holds the call's ID and name. Without a result, the tool may have run
)

type ContentKind string

const (
	ContentKindText    ContentKind = "text"
	ContentKindRefusal ContentKind = "refusal"
	ContentKindFile    ContentKind = "file" // an Attachment
)

type ContentPart struct {
	Kind ContentKind `json:"kind"`
	Text string      `json:"text"`

	// A file holds its bytes or, for URL attachments, its URL.
	Name string `json:"name,omitempty"`
	MIME string `json:"mime,omitempty"`
	Data []byte `json:"data,omitempty"`
	URL  string `json:"url,omitempty"`
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

	Run        *RunStatus    `json:"run,omitempty"`
	Turn       *TurnInfo     `json:"turn,omitempty"`
	Compaction *Compaction   `json:"compaction,omitempty"`
	Response   *ResponseInfo `json:"response,omitempty"` // on the last entry of a model response

	Opaque map[string][]byte `json:"opaque,omitempty"` // provider adornments; dropped on provider switch
	Usage  *Usage            `json:"usage,omitempty"`  // tokens; never affects replay
}

// NewUserEntry builds the user entry Run records for inputs, in order: strings
// and text, Attachments (read here), and other values sent as JSON. nil inputs
// are skipped.
func NewUserEntry(inputs ...any) (Entry, error) {
	var content []ContentPart
	for _, input := range inputs {
		var text string
		switch v := input.(type) {
		case nil:
			continue
		case Attachment:
			part, err := v.part()
			if err != nil {
				return Entry{}, err
			}
			content = append(content, part)
			continue
		case *Attachment:
			if v == nil {
				continue
			}
			part, err := v.part()
			if err != nil {
				return Entry{}, err
			}
			content = append(content, part)
			continue
		case string:
			text = v
		case []byte:
			if !utf8.Valid(v) {
				return Entry{}, errors.New("user input is binary; wrap it in crux.Data to send it as a file")
			}
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
		content = append(content, ContentPart{Kind: ContentKindText, Text: text})
	}
	return Entry{
		At:      time.Now().UTC(),
		Kind:    KindUser,
		Content: content,
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
// (like state deltas, user approvals or lifecycle records) that is not sent to
// LLM providers.
func (e Entry) HiddenFromModel() bool {
	return e.Kind == KindStateDelta || e.Kind == KindApproval || e.Kind == KindCompaction || e.Kind.lifecycle()
}

// lifecycle reports whether entries of this kind only record the progress of
// a run, so they neither end a model turn nor belong to one.
func (k Kind) lifecycle() bool {
	switch k {
	case KindRunStarted, KindRunFinished, KindTurnStarted, KindToolStarted:
		return true
	}
	return false
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

type ToolResult struct {
	CallID string `json:"call_id"` // references ToolCall.ID
	Output string `json:"output,omitempty"`
	Error  string `json:"error,omitempty"` // set on failure or decline; still owed to the model
	// Denied reports that the tool did not run because the user rejected the
	// call. Error holds the reason.
	Denied bool `json:"denied,omitempty"`
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

// RunOutcome says how a run ended.
type RunOutcome string

const (
	RunAnswered       RunOutcome = "answered"
	RunApprovalNeeded RunOutcome = "approval_needed"
	RunRefused        RunOutcome = "refused"
	RunMaxTurns       RunOutcome = "max_turns"
	RunCancelled      RunOutcome = "cancelled"
	RunFailed         RunOutcome = "failed"
)

// RunStatus is recorded on a KindRunFinished entry. Error is the text of the
// error the run returned, if any.
type RunStatus struct {
	Outcome RunOutcome `json:"outcome"`
	Error   string     `json:"error,omitempty"`
}

// TurnInfo is recorded on a KindTurnStarted entry. AgentID identifies the
// agent's configuration, including a hash of each tool's definition
// (GORMStore keeps it in crux_agents), so a request can be tied to the exact
// definitions it used.
type TurnInfo struct {
	AgentID  uuid.UUID `json:"agent_id"`
	Provider Provider  `json:"provider"`
	Model    string    `json:"model"`
}

// Compaction is recorded on a KindCompaction entry; the entries it covers
// stay in the log, but the model sees less of them. Every compaction omits
// large tool outputs and files up to Through. One with a Summary also
// replaces those entries with it. Usage counts the summary request.
type Compaction struct {
	Through uint64 `json:"through"`           // the Seq of the last entry it covers
	Summary string `json:"summary,omitempty"` // empty when only outputs were omitted
}

// ResponseInfo describes one model response. ID is the provider's response
// ID, for looking the request up with the provider. FirstTokenAfter is the
// time from sending the request to the first streamed delta; it is zero when
// the response was not streamed.
type ResponseInfo struct {
	ID              string        `json:"id,omitempty"`
	FirstTokenAfter time.Duration `json:"first_token_after,omitempty"`
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

var (
	// ErrToolNotFound is returned when an agent selects a tool that is not registered.
	ErrToolNotFound = errors.New("tool not found")
	// ErrApprovalNeeded is returned by Run when a tool call waits for Approve or Reject.
	ErrApprovalNeeded = errors.New("approval needed")
	// ErrOutputValidation wraps output that does not match the agent's output schema.
	ErrOutputValidation = errors.New("output validation failed")
	// ErrSessionNotFound is returned by a Store that has no entries for a session.
	ErrSessionNotFound = errors.New("session not found")
	// ErrSessionConflict is returned by a Store when another writer already
	// appended entries to the session. Load the session again with
	// WithSessionID and retry.
	ErrSessionConflict = errors.New("session was changed by another writer")
	// ErrContextTooLong wraps a provider error saying the request exceeds the
	// model's context window, when compacting the session could not help.
	ErrContextTooLong = errors.New("context window exceeded")
	// ErrMaxTurns is returned by Run when the agent used all its turns without a final answer.
	ErrMaxTurns = errors.New("max turns reached")
	// ErrRefused is returned by Run when the model refuses the request.
	ErrRefused = errors.New("model refused the request")
	// ErrRateLimited is matched by a *ProviderError saying too many requests
	// or tokens were sent. Retry after its RetryAfter.
	ErrRateLimited = errors.New("rate limited")
	// ErrOverloaded is matched by a *ProviderError saying the provider is
	// temporarily out of capacity or unavailable.
	ErrOverloaded = errors.New("provider overloaded")
	// ErrInsufficientCredits is matched by a *ProviderError saying the account
	// has no credit or quota left. Retrying won't help.
	ErrInsufficientCredits = errors.New("insufficient credits")
)

// ProviderError is a provider limit that crux recognised, the same on every
// provider. errors.Is matches its sentinel (ErrRateLimited, ErrOverloaded or
// ErrInsufficientCredits) and errors.As the provider's own error.
type ProviderError struct {
	Provider Provider
	// StatusCode is the HTTP status, or 0 when the error came after the
	// response started, as a stream event or a failed response.
	StatusCode int
	// RetryAfter is how long the provider asked to wait; 0 when it didn't say.
	RetryAfter time.Duration
	// Err is the provider's error.
	Err  error
	kind error
}

func (e *ProviderError) Error() string { return e.kind.Error() + ": " + e.Err.Error() }

func (e *ProviderError) Unwrap() []error { return []error{e.kind, e.Err} }
