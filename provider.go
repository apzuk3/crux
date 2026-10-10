package crux

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"crux.foo/internal/provider"
	"crux.foo/internal/schema"
	"github.com/invopop/jsonschema"
	sjs "github.com/santhosh-tekuri/jsonschema/v5"
)

type Provider string

const (
	ProviderOpenAI     Provider = "openai"
	ProviderAnthropic  Provider = "anthropic"
	ProviderGoogle     Provider = "google"
	ProviderOpenrouter Provider = "openrouter"
	ProviderDeepSeek   Provider = "deepseek"
	ProviderXAI        Provider = "xai"
	ProviderOllama     Provider = "ollama"
	ProviderTypeSafe   Provider = "typesafe"
)

// providerSpec is everything crux knows about one provider.
type providerSpec struct {
	models  []string
	envVars []string // checked in order for an API key
	baseURL string   // "" uses the SDK default
	step    provider.Step
	schema  func(root map[string]any, provider string) (map[string]any, error)
	prepare func(a *Agent) error // optional; provider rules and defaults, run at the end of New
	// contextWindow returns a model's context window in tokens, or 0 when
	// unknown. Optional; WithContextWindow overrides it.
	contextWindow func(model string) int
	// decide answers typed questions for NewDecider; optional. isDecisionModel
	// reports which models it serves (nil: all of them). Other models decide
	// through step with a structured output schema.
	decide          provider.Decide
	isDecisionModel func(model string) bool
}

// decidesNatively reports whether model is served by the spec's decide.
func (spec providerSpec) decidesNatively(model string) bool {
	return spec.decide != nil && (spec.isDecisionModel == nil || spec.isDecisionModel(model))
}

var providerSpecs = map[Provider]providerSpec{}

// registerProvider adds a provider. It is called only from init functions.
func registerProvider(provider Provider, spec providerSpec) {
	if _, dup := providerSpecs[provider]; dup || (spec.step == nil && spec.decide == nil) || (spec.step != nil && spec.schema == nil) {
		panic(fmt.Sprintf("crux: invalid registration of provider %q", provider))
	}
	providerSpecs[provider] = spec
}

func inferProvider(modelName string) Provider {
	var matched Provider
	for provider, spec := range providerSpecs {
		for _, m := range spec.models {
			if m == modelName {
				if matched != "" && matched != provider {
					return "" // Ambiguous IDs require WithProvider.
				}
				matched = provider
			}
		}
	}
	return matched
}

// step sends one request to the provider. Credentials a base URL carries are
// removed from the error, because provider SDKs print the request URL.
// first reports the first request after new input, the only one WithToolChoice
// applies to.
func (a *Agent) step(ctx context.Context, client *http.Client, log []Entry, emit chunkSink, first bool) ([]Entry, error) {
	spec, ok := providerSpecs[a.provider]
	if !ok {
		return nil, fmt.Errorf("unsupported provider %q", a.provider)
	}
	req, err := a.wireRequest(log, first)
	if err != nil {
		return nil, redactURLSecrets(err, a.baseURL)
	}
	req.HTTPClient = client
	var sink provider.Emit
	emitted := false
	if emit != nil {
		sink = func(kind provider.ChunkKind, delta string) error {
			emitted = true
			return emit(Chunk{Kind: ChunkKind(kind), Delta: delta})
		}
	}
	items, err := stepWithRetries(ctx, spec, req, sink, &emitted)
	if err != nil {
		return nil, redactURLSecrets(a.stepError(err), a.baseURL)
	}
	entries := make([]Entry, len(items))
	for i, item := range items {
		entries[i] = fromItem(item)
	}
	return entries, nil
}

// stepWithRetries calls the provider. A limit the SDK didn't retry (one that
// came after the response started, or a Gemini HTTP limit) is retried here,
// waiting for the provider's delay, unless deltas were already emitted.
func stepWithRetries(ctx context.Context, spec providerSpec, req *provider.Request, sink provider.Emit, emitted *bool) ([]provider.Item, error) {
	for attempt := 0; ; attempt++ {
		items, err := spec.step(ctx, req, sink)
		var limit *provider.LimitError
		if err == nil || *emitted || attempt >= req.Retries() || !errors.As(err, &limit) || limit.Retried || !limit.Kind.Retryable() {
			return items, err
		}
		if provider.Sleep(ctx, provider.RetryDelay(limit.RetryAfter, attempt)) != nil {
			return items, err
		}
	}
}

// stepError maps a provider error onto crux's errors: refusals wrap
// ErrRefused, a context too long wraps ErrContextTooLong, and limits become
// a *ProviderError.
func (a *Agent) stepError(err error) error {
	var refusal *provider.RefusedError
	switch {
	case errors.As(err, &refusal):
		if refusal.Detail != "" {
			return fmt.Errorf("%s: %w: %s", refusal.Provider, ErrRefused, refusal.Detail)
		}
		return fmt.Errorf("%s: %w", refusal.Provider, ErrRefused)
	case provider.IsContextTooLong(err):
		return fmt.Errorf("%w: %w", ErrContextTooLong, err)
	default:
		return providerError(a.provider, err)
	}
}

// providerError returns err as a *ProviderError when the provider reported a
// limit, and err unchanged otherwise.
func providerError(p Provider, err error) error {
	var limit *provider.LimitError
	if !errors.As(err, &limit) {
		return err
	}
	kind := map[provider.LimitKind]error{
		provider.LimitRateLimited:         ErrRateLimited,
		provider.LimitOverloaded:          ErrOverloaded,
		provider.LimitInsufficientCredits: ErrInsufficientCredits,
	}[limit.Kind]
	return &ProviderError{Provider: p, StatusCode: limit.StatusCode, RetryAfter: limit.RetryAfter, Err: err, kind: kind}
}

// wireRequest describes the agent's next request: its settings and the
// entries of log the model sees. The tool choice is sent only on the first
// request after new input.
func (a *Agent) wireRequest(log []Entry, first bool) (*provider.Request, error) {
	req := &provider.Request{
		Provider:     string(a.provider),
		Model:        a.model,
		Instructions: a.instructions,
		APIKey:       a.apiKey,
		BaseURL:      a.baseURL,
		MaxTokens:    a.maxTokens,
		Temperature:  a.temperature,
		Reasoning:    string(a.reasoning),
		Parallel:     a.parallel,
		MaxRetries:   a.maxRetries,
		Seq:          len(log),
	}
	if first {
		a.wireToolChoice(req)
	}
	if err := a.wireTools(req); err != nil {
		return nil, err
	}
	req.Search = a.wireSearch()
	if a.outputSchema != nil {
		schema, err := wireSchemaFor(a.outputSchema, a.provider)
		if err != nil {
			return nil, err
		}
		req.OutputSchema = schema
	}
	req.Log = wireLog(log)
	return req, nil
}

func (a *Agent) wireToolChoice(req *provider.Request) {
	if name, ok := a.toolChoice.tool(); ok {
		req.ToolChoice, req.ToolName = provider.ToolChoiceTool, name
		return
	}
	req.ToolChoice = string(a.toolChoice)
}

// wireTools adds the tools to the request, and the instructions tools carry
// to its instructions, each set once.
func (a *Agent) wireTools(req *provider.Request) error {
	added := make(map[*toolInstructions]bool)
	for _, tool := range a.tools {
		req.Tools = append(req.Tools, provider.Tool{Name: tool.name, Description: tool.description, Schema: tool.schema})
		if tool.instructions == nil || added[tool.instructions] {
			continue
		}
		added[tool.instructions] = true
		text, err := tool.instructions.text()
		if err != nil {
			return fmt.Errorf("instructions of tool %q: %w", tool.name, err)
		}
		if text != "" && req.Instructions != "" {
			req.Instructions += "\n\n"
		}
		req.Instructions += text
	}
	return nil
}

func (a *Agent) wireSearch() *provider.Search {
	if a.searchOptions == nil {
		return nil
	}
	search := &provider.Search{}
	if l := a.searchOptions.UserLocation; l != nil {
		search.Location = &provider.Location{
			Country: l.Country, City: l.City, Region: l.Region, Timezone: l.Timezone,
			Latitude: l.Latitude, Longitude: l.Longitude,
		}
	}
	return search
}

// wireLog converts the entries of log the model sees.
func wireLog(log []Entry) []provider.Item {
	view := modelView(log)
	items := make([]provider.Item, 0, len(view))
	for _, e := range view {
		items = append(items, toItem(e))
	}
	return items
}

var itemKinds = map[Kind]provider.Kind{
	KindUser:         provider.KindUser,
	KindAssistant:    provider.KindAssistant,
	KindReasoning:    provider.KindReasoning,
	KindToolCall:     provider.KindToolCall,
	KindToolResult:   provider.KindToolResult,
	KindProviderTool: provider.KindProviderTool,
}

var entryKinds = func() map[provider.Kind]Kind {
	kinds := make(map[provider.Kind]Kind, len(itemKinds))
	for k, v := range itemKinds {
		kinds[v] = k
	}
	return kinds
}()

func toItem(e Entry) provider.Item {
	item := provider.Item{At: e.At, Kind: itemKinds[e.Kind], Opaque: e.Opaque}
	if e.Content != nil {
		item.Content = make([]provider.ContentPart, 0, len(e.Content))
	}
	for _, part := range e.Content {
		item.Content = append(item.Content, provider.ContentPart{
			Kind: provider.ContentKind(part.Kind), Text: part.Text,
			Name: part.Name, MIME: part.MIME, Data: part.Data, URL: part.URL,
		})
	}
	if e.Reasoning != nil {
		item.Reasoning = &provider.Reasoning{Summary: e.Reasoning.Summary}
	}
	if c := e.ToolCall; c != nil {
		item.ToolCall = &provider.ToolCall{ID: c.ID, Name: c.Name, Args: c.Args}
	}
	if r := e.ToolResult; r != nil {
		item.ToolResult = &provider.ToolResult{CallID: r.CallID, Output: r.Output, Error: r.Error}
	}
	return item
}

func fromItem(item provider.Item) Entry {
	e := Entry{At: item.At, Kind: entryKinds[item.Kind], Opaque: item.Opaque}
	if item.Content != nil {
		e.Content = make([]ContentPart, 0, len(item.Content))
	}
	for _, part := range item.Content {
		e.Content = append(e.Content, ContentPart{
			Kind: ContentKind(part.Kind), Text: part.Text,
			Name: part.Name, MIME: part.MIME, Data: part.Data, URL: part.URL,
		})
	}
	if item.Reasoning != nil {
		e.Reasoning = &Reasoning{Summary: item.Reasoning.Summary}
	}
	if c := item.ToolCall; c != nil {
		e.ToolCall = &ToolCall{ID: c.ID, Name: c.Name, Args: c.Args}
	}
	if r := item.ToolResult; r != nil {
		e.ToolResult = &ToolResult{CallID: r.CallID, Output: r.Output, Error: r.Error}
	}
	if item.Response != nil {
		e.Response = &ResponseInfo{ID: item.Response.ID}
	}
	if u := item.Usage; u != nil {
		e.Usage = &Usage{
			InputTokens:      u.InputTokens,
			OutputTokens:     u.OutputTokens,
			CacheReadTokens:  u.CacheReadTokens,
			CacheWriteTokens: u.CacheWriteTokens,
		}
	}
	return e
}

// wireSchemaFor adapts a jsonschema.Schema for a specific provider's wire format.
func wireSchemaFor(s *jsonschema.Schema, provider Provider) (map[string]any, error) {
	m, err := schema.Wire(s)
	if m == nil || err != nil {
		return m, err
	}
	spec, ok := providerSpecs[provider]
	if !ok || spec.schema == nil { // a decision-only provider adapts nothing
		return m, nil
	}
	m, err = spec.schema(m, string(provider))
	if err != nil {
		return nil, err
	}
	return plainNumbers(m).(map[string]any), nil
}

// plainNumbers replaces the json.Numbers in a decoded schema with int64 or
// float64 values. The OpenAI and Anthropic SDKs encode a json.Number as a
// string, which turns "maximum": 3 into "maximum": "3".
func plainNumbers(v any) any {
	switch v := v.(type) {
	case map[string]any:
		for k, item := range v {
			v[k] = plainNumbers(item)
		}
	case []any:
		for i, item := range v {
			v[i] = plainNumbers(item)
		}
	case json.Number:
		if n, err := v.Int64(); err == nil {
			return n
		}
		if f, err := v.Float64(); err == nil {
			return f
		}
	}
	return v
}

// validateOutput validates the output string against the pre-compiled validator.
func validateOutput(validator *sjs.Schema, text string) error {
	if err := schema.ValidateOutput(validator, text); err != nil {
		return fmt.Errorf("%w: %v", ErrOutputValidation, err)
	}
	return nil
}
