// Package provider holds the wire code of the three generation APIs crux
// speaks: OpenAI Responses (also used by xAI, DeepSeek, OpenRouter and
// Ollama), Anthropic Messages and Google GenAI. It knows nothing about crux:
// the crux package turns an agent and its log into a Request and the Items a
// Step returns back into log entries.
package provider

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"mime"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"
)

// Step sends one request and returns what the model produced, in order. The
// last item carries the response's Response and Usage. emit is nil when the
// response is not streamed.
type Step func(ctx context.Context, req *Request, emit Emit) ([]Item, error)

// Request is everything a Step needs.
type Request struct {
	Provider     string
	Model        string
	Instructions string
	APIKey       string
	BaseURL      string
	HTTPClient   *http.Client // nil uses the SDK's client or DefaultHTTPClient
	Tools        []Tool
	MaxTokens    int      // 0 uses the provider default
	Temperature  *float64 // nil uses the provider default
	Reasoning    string   // "" uses the provider default; see the Reasoning constants
	// ToolChoice is "" (the model decides) or one of the ToolChoice
	// constants; ToolChoiceTool forces the tool named ToolName.
	ToolChoice   string
	ToolName     string
	Parallel     *bool   // nil uses the provider default
	MaxRetries   *int    // nil retries twice
	Search       *Search // nil disables web search
	OutputSchema map[string]any
	// Log holds the entries the model sees, oldest first.
	Log []Item
	// Seq is the length of the whole session log, for IDs that must not
	// repeat across turns.
	Seq int
}

// Reasoning efforts, as crux.ReasoningEffort spells them.
const (
	ReasoningOff    = "off"
	ReasoningLow    = "low"
	ReasoningMedium = "medium"
	ReasoningHigh   = "high"
	ReasoningMax    = "max"
)

// Tool choices, as crux.ToolChoice spells them.
const (
	ToolChoiceAuto     = "auto"
	ToolChoiceRequired = "required"
	ToolChoiceNone     = "none"
	ToolChoiceTool     = "tool"
)

// retries returns req.MaxRetries, or the SDKs' default of two.
func (req *Request) retries() int {
	if req.MaxRetries != nil {
		return *req.MaxRetries
	}
	return 2
}

// Tool is a function the model may call.
type Tool struct {
	Name        string
	Description string
	Schema      map[string]any
}

// Search turns on the provider's web search.
type Search struct {
	Location *Location // nil means no location is supplied
}

// Location mirrors crux.UserLocation.
type Location struct {
	Country   string
	City      string
	Region    string
	Timezone  string
	Latitude  *float64
	Longitude *float64
}

// Kind is the kind of an Item: the kinds of crux log entries the model sees.
type Kind uint8

const (
	KindUser Kind = iota + 1
	KindAssistant
	KindReasoning
	KindToolCall
	KindToolResult
	KindProviderTool
)

type ContentKind string

const (
	ContentKindText    ContentKind = "text"
	ContentKindRefusal ContentKind = "refusal"
	ContentKindFile    ContentKind = "file"
)

// ContentPart is text, a refusal or a file. A file holds its bytes, or its
// URL when the provider downloads it.
type ContentPart struct {
	Kind ContentKind
	Text string
	Name string
	MIME string
	Data []byte
	URL  string
}

// IsImage reports whether the part is an image file.
func (p ContentPart) IsImage() bool {
	return p.Kind == ContentKindFile && strings.HasPrefix(p.MIME, "image/")
}

// InlineText returns a text file as text the model reads inline, wrapped in a
// tag that names it. Every provider accepts it this way, whatever the format.
func (p ContentPart) InlineText() (string, bool) {
	if p.Kind != ContentKindFile || p.URL != "" || !isTextMIME(p.MIME) {
		return "", false
	}
	if p.Name == "" {
		return fmt.Sprintf("<file type=%q>\n%s\n</file>", p.MIME, p.Data), true
	}
	return fmt.Sprintf("<file name=%q type=%q>\n%s\n</file>", p.Name, p.MIME, p.Data), true
}

// fileName returns the part's name, or "file" with an extension for its type.
func (p ContentPart) fileName() string {
	if p.Name != "" {
		return p.Name
	}
	if exts, _ := mime.ExtensionsByType(p.MIME); len(exts) > 0 {
		return "file" + exts[0]
	}
	return "file"
}

// DataURL returns the file's bytes as a base64 data URL.
func (p ContentPart) DataURL() string {
	return "data:" + p.MIME + ";base64," + base64.StdEncoding.EncodeToString(p.Data)
}

func isTextMIME(mimeType string) bool {
	switch mimeType {
	case "application/json", "application/xml", "application/yaml", "application/x-yaml":
		return true
	}
	return strings.HasPrefix(mimeType, "text/") || strings.HasSuffix(mimeType, "+json") || strings.HasSuffix(mimeType, "+xml")
}

// unsupportedFile is the error for a file a provider can't take.
func unsupportedFile(provider string, p ContentPart) error {
	how := "as bytes"
	if p.URL != "" {
		how = "by URL"
	}
	return fmt.Errorf("%s does not accept %s files %s", provider, p.MIME, how)
}

// Item mirrors the parts of a crux log entry that providers read and write.
type Item struct {
	At         time.Time
	Kind       Kind
	Content    []ContentPart
	Reasoning  *Reasoning
	ToolCall   *ToolCall
	ToolResult *ToolResult
	Response   *ResponseInfo
	Opaque     map[string][]byte
	Usage      *Usage
}

type Reasoning struct {
	Summary string
}

type ToolCall struct {
	ID   string
	Name string
	Args json.RawMessage
}

type ToolResult struct {
	CallID string
	Output string
	Error  string
}

type ResponseInfo struct {
	ID string
}

type Usage struct {
	InputTokens      int
	OutputTokens     int
	CacheReadTokens  int
	CacheWriteTokens int
}

// ChunkKind says what a streamed delta is.
type ChunkKind string

const (
	ChunkText      ChunkKind = "text"
	ChunkReasoning ChunkKind = "reasoning"
)

// Emit receives streamed deltas.
type Emit func(kind ChunkKind, delta string) error

func emitChunk(emit Emit, kind ChunkKind, delta string) error {
	if delta == "" {
		return nil
	}
	return emit(kind, delta)
}

// RefusedError reports that the model refused the request. The crux package
// turns it into an error wrapping crux.ErrRefused.
type RefusedError struct {
	Provider string
	Detail   string // empty when the provider gives none
}

func (e *RefusedError) Error() string {
	if e.Detail == "" {
		return e.Provider + ": model refused the request"
	}
	return e.Provider + ": model refused the request: " + e.Detail
}

// IsContextTooLong reports whether err is a provider saying the request is
// larger than the model accepts. Providers return these as plain 400s, so the
// message is all there is to go on.
func IsContextTooLong(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	for _, s := range []string{
		"context_length_exceeded",    // OpenAI
		"maximum context length",     // OpenAI, DeepSeek, OpenRouter
		"maximum prompt length",      // xAI
		"prompt is too long",         // Anthropic
		"request_too_large",          // Anthropic, request body over its size limit
		"exceeds the context window", // OpenAI-compatible servers
		"max_tokens_exceeded",        // TypeSafe Jev
	} {
		if strings.Contains(msg, s) {
			return true
		}
	}
	// Gemini: "The input token count (N) exceeds the maximum number of tokens allowed (M)."
	return strings.Contains(msg, "input token count") && strings.Contains(msg, "exceeds the maximum")
}

func refused(provider, detail string) error {
	return &RefusedError{Provider: provider, Detail: detail}
}

// DefaultHTTPClient is used when neither the agent nor the session sets a
// client and the SDK would not supply its own. Like the OpenAI SDK's default, it
// gives up on a server that accepts a request but never sends response headers;
// the body is not limited, so long streams are unaffected. It is built on first
// use, so a wrapped http.DefaultTransport (for tracing, say) is kept, though
// then without the timeout. It is a variable so tests can replace it.
var DefaultHTTPClient = sync.OnceValue(func() *http.Client {
	return NewHTTPClient(10 * time.Minute)
})

// NewHTTPClient returns a client on a clone of http.DefaultTransport that
// waits at most responseHeaderTimeout for response headers.
func NewHTTPClient(responseHeaderTimeout time.Duration) *http.Client {
	if transport, ok := http.DefaultTransport.(*http.Transport); ok {
		transport = transport.Clone()
		transport.ResponseHeaderTimeout = responseHeaderTimeout
		return &http.Client{Transport: transport}
	}
	return &http.Client{Transport: http.DefaultTransport}
}

// Text returns the concatenated textual content, excluding refusals.
func (e Item) Text() string {
	var text strings.Builder
	for _, part := range e.Content {
		if part.Kind == ContentKindText {
			text.WriteString(part.Text)
		}
	}
	return text.String()
}

// hasAnswerOrCall reports whether a model turn contains assistant content or a
// tool call. A turn with only reasoning or provider-tool events still needs an
// (empty) assistant entry, or it would never count as finished.
func hasAnswerOrCall(items []Item) bool {
	return slices.ContainsFunc(items, func(e Item) bool {
		return e.Kind == KindAssistant || e.Kind == KindToolCall
	})
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
