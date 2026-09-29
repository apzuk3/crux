package cruxtest

import (
	"encoding/json"
	"fmt"
)

// ToolCall represents a mock tool invocation to be returned to the agent.
type ToolCall struct {
	ID   string
	Name string
	Args any
}

// TokenUsage defines the usage numbers reported by the mock response. It
// matches crux.Usage: InputTokens is the total input, including cache reads
// and writes, and each provider's wire format is derived from it.
type TokenUsage struct {
	InputTokens      int
	OutputTokens     int
	CacheReadTokens  int
	CacheWriteTokens int
}

// Turn represents an expected interaction turn from the model.
type Turn struct {
	Text       string
	ToolCalls  []ToolCall
	Refusal    string
	StatusCode int
	RawBody    []byte
	Usage      *TokenUsage
	empty      bool
}

// ReturnText configures the turn to respond with a plain text assistant message.
func (t *Turn) ReturnText(text string) *Turn {
	t.Text = text
	return t
}

// ReturnJSON marshals val to JSON and configures the turn to respond with that JSON text.
func (t *Turn) ReturnJSON(val any) *Turn {
	b, err := json.Marshal(val)
	if err != nil {
		panic(fmt.Sprintf("cruxtest: failed to marshal JSON in ReturnJSON: %v", err))
	}
	t.Text = string(b)
	return t
}

// ReturnToolCall configures the turn to invoke a single tool call.
func (t *Turn) ReturnToolCall(name string, args any) *Turn {
	t.ToolCalls = append(t.ToolCalls, ToolCall{
		Name: name,
		Args: args,
	})
	return t
}

// ReturnToolCalls configures the turn to invoke multiple tool calls.
func (t *Turn) ReturnToolCalls(calls ...ToolCall) *Turn {
	t.ToolCalls = append(t.ToolCalls, calls...)
	return t
}

// ReturnEmpty configures the turn to complete normally with no content at all.
func (t *Turn) ReturnEmpty() *Turn {
	t.empty = true
	return t
}

// ReturnRefusal configures the turn to refuse the request with a reason.
func (t *Turn) ReturnRefusal(reason string) *Turn {
	t.Refusal = reason
	return t
}

// ReturnError configures the turn to return an HTTP error status and body.
func (t *Turn) ReturnError(statusCode int, body string) *Turn {
	t.StatusCode = statusCode
	t.RawBody = []byte(body)
	return t
}

// ReturnRaw configures the turn with an exact raw HTTP status and byte payload.
func (t *Turn) ReturnRaw(statusCode int, body []byte) *Turn {
	t.StatusCode = statusCode
	t.RawBody = body
	return t
}

// WithUsage sets the reported token usage for this turn.
func (t *Turn) WithUsage(usage TokenUsage) *Turn {
	t.Usage = &usage
	return t
}

// NewToolCall creates a ToolCall with the specified name and arguments.
func NewToolCall(name string, args any) ToolCall {
	return ToolCall{
		Name: name,
		Args: args,
	}
}
