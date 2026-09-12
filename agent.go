package crux

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/invopop/jsonschema"
	"github.com/openai/openai-go/v3"
	"google.golang.org/genai"
)

type Agent struct {
	maxTurns     int32
	instructions string
	allowedTools []string

	provider     Provider
	model        string
	baseURL      string
	apikey       string
	outputSchema *jsonschema.Schema

	toolsRegistry *ToolsRegistry

	openai    *openai.Client
	anthropic *anthropic.Client
	gemini    *genai.Client
}

type AgentOption func(*Agent)

func WithAllowedTools(tools []string) AgentOption {
	return func(a *Agent) { a.allowedTools = tools }
}

func WithInstructions(instructions string) AgentOption {
	return func(a *Agent) { a.instructions = instructions }
}

func WithProvider(provider Provider) AgentOption {
	return func(a *Agent) { a.provider = provider }
}

func WithMaxTurns(turns int32) AgentOption {
	return func(a *Agent) { a.maxTurns = turns }
}

func WithBaseURL(url string) AgentOption {
	return func(a *Agent) { a.baseURL = url }
}

func WithAPIKey(apikey string) AgentOption {
	return func(a *Agent) { a.apikey = apikey }
}

func WithTools(registry *ToolsRegistry) AgentOption {
	return func(a *Agent) { a.toolsRegistry = registry }
}

// WithOutputSchema sets the response schema. For OpenAI, it must satisfy strict
// Structured Outputs requirements: an object root, all properties required, and
// additionalProperties: false on every object.
// For Anthropic, the schema must satisfy its supported JSON Schema subset;
// it is sent unchanged through output_config.format.
// For Gemini, the schema must satisfy its supported JSON Schema subset;
// it is sent unchanged through responseJsonSchema.
func WithOutputSchema(schema *jsonschema.Schema) AgentOption {
	return func(a *Agent) { a.outputSchema = schema }
}

// WithOutputSchemaFrom reflects T into a response schema, or disables structured
// output for string and any. T must satisfy WithOutputSchema's requirements;
// reflection does not normalize optional fields or maps for OpenAI strict mode.
func WithOutputSchemaFrom[T any]() AgentOption {
	return func(a *Agent) {
		var zero T
		switch any(&zero).(type) {
		case *string, *any:
			a.outputSchema = nil
		default:
			a.outputSchema = jsonschema.ReflectFromType(reflect.TypeFor[T]())
		}
	}
}

func NewAgent(model string, opts ...AgentOption) *Agent {
	agent := &Agent{
		model:         model,
		maxTurns:      10,
		toolsRegistry: defaultToolsRegistry,
	}
	for _, opt := range opts {
		opt(agent)
	}

	if agent.provider == "" {
		agent.provider = inferProvider(model)
	}

	if agent.apikey == "" {
		agent.apikey = discoverAPIKey(agent.provider)
	}

	if agent.provider == ProviderOpenrouter && agent.baseURL == "" {
		agent.baseURL = "https://api.openrouter.ai"
	}

	if agent.provider == ProviderOpenAI || agent.provider == ProviderOpenrouter {
		agent.openai = agent.newOpenAIClient()
	}

	if agent.provider == ProviderAnthropic {
		agent.anthropic = agent.newAnthropicClient()
	}

	return agent
}

func (a *Agent) Run(ctx context.Context, input any) (string, error) {
	// The user turn is part of the log, so every provider sees one shape and a
	// resumed session needs nothing but its history.
	entry, err := NewUserEntry(input)
	if err != nil {
		return "", err
	}
	log := []Entry{entry}

	// ---> Notify start
	for range a.maxTurns {
		var produced []Entry

		switch a.provider {
		case ProviderAnthropic:
			produced, err = a.anthropicStep(ctx, log)
			if err != nil {
				return "", err
			}
		case ProviderOpenAI, ProviderOpenrouter:
			produced, err = a.openAIstep(ctx, log)
			if err != nil {
				return "", err
			}
		case ProviderGoogle:
			produced, err = a.geminiStep(ctx, log)
			if err != nil {
				return "", err
			}
		default:
			return "", fmt.Errorf("unsupported provider %q", a.provider)
		}

		log = append(log, produced...)
		results := a.dispatchCalls(ctx, produced)
		log = append(log, results...)

		// Local tool results require another model turn.
		if len(results) == 0 {
			// ---> Notify end
			return finalText(produced), nil
		}
	}

	return "", errors.New("max turns reached")
}

// Run executes the agent and decodes its final text response into Out. Anything
// that is not text is read back as JSON.
func Run[Out any](ctx context.Context, a *Agent, input any) (Out, error) {
	text, err := a.Run(ctx, input)
	if err != nil {
		var zero Out
		return zero, err
	}
	return decodeOutput[Out](text)
}

// dispatch runs a tool call locally. A failure is reported to the model rather
// than returned, because the call is still owed an answer.
func (a *Agent) dispatch(ctx context.Context, call ToolCall) Entry {
	result := ToolResult{CallID: call.ID}

	switch tool, ok := a.toolsRegistry.lookup(call.Name); {
	case !ok:
		result.Error = fmt.Sprintf("unknown tool %q", call.Name)
	default:
		output, err := tool.invoke(ctx, call.Args)
		if err != nil {
			result.Error = err.Error()
		} else {
			result.Output = output
		}
	}

	return Entry{Kind: KindToolResult, ToolResult: &result}
}

// dispatchCalls executes local calls in model order. The caller appends the
// results after all model entries, as required by the provider protocols.
func (a *Agent) dispatchCalls(ctx context.Context, entries []Entry) []Entry {
	var results []Entry
	for _, e := range entries {
		if e.Kind == KindToolCall {
			results = append(results, a.dispatch(ctx, *e.ToolCall))
		}
	}
	return results
}

func finalText(entries []Entry) string {
	var text strings.Builder
	for _, entry := range entries {
		if entry.Kind == KindAssistant {
			text.WriteString(entry.Text())
		}
	}
	return text.String()
}

func (a *Agent) OutputSchema() *jsonschema.Schema {
	return a.outputSchema
}

// decodeOutput renders the model's final text as Out. Anything that is not
// text is read back as JSON.
func decodeOutput[Out any](text string) (Out, error) {
	var output Out

	switch target := any(&output).(type) {
	case *string:
		*target = text
		return output, nil
	case *any:
		*target = text
		return output, nil
	}

	if err := json.Unmarshal([]byte(text), &output); err != nil {
		return output, fmt.Errorf("decode agent output as %T: %w", output, err)
	}
	return output, nil
}
