package crux

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/invopop/jsonschema"
)

type Agent struct {
	name         string
	maxTurns     int32
	instructions string

	provider     Provider
	model        string
	baseURL      string
	apikey       string
	outputSchema *jsonschema.Schema

	searchOptions *SearchOptions // nil disables web search

	allowedTools  []string
	toolsRegistry *ToolsRegistry
	tools         []Tool // nil means tool options need binding; bound empty sets are non-nil

	logs []Entry
}

// SearchOptions configures provider-executed web search.
type SearchOptions struct {
	UserLocation *UserLocation // nil means no location is supplied
}

type SearchOption func(*SearchOptions)

func NewAgent(name, model string, opts ...AgentOption) *Agent {
	agent := &Agent{
		name:          name,
		model:         model,
		maxTurns:      10,
		toolsRegistry: defaultToolsRegistry,
	}
	for _, opt := range opts {
		opt(agent)
	}
	if agent.tools == nil {
		agent.tools = agent.toolsRegistry.selected(agent.allowedTools)
	}

	if agent.provider == "" {
		agent.provider = inferProvider(agent.model)
	}

	if agent.apikey == "" {
		agent.apikey = discoverAPIKey(agent.provider)
	}

	if agent.baseURL == "" {
		switch agent.provider {
		case ProviderOpenrouter:
			agent.baseURL = "https://api.openrouter.ai"
		case ProviderXAI:
			agent.baseURL = "https://api.x.ai/v1"
		case ProviderDeepSeek:
			agent.baseURL = "https://api.deepseek.com"
		case ProviderOllama:
			agent.baseURL = "http://localhost:11434/v1"
		}
	}

	if agent.provider == ProviderOllama && agent.apikey == "" {
		agent.apikey = "ollama" // Local Ollama ignores authentication.
	}

	return agent
}

// Run continues the retained conversation and returns its final text response.
// User inputs, model entries, and tool results are retained even if a later step
// fails. Run must not execute concurrently with other operations on the agent.
func (a *Agent) Run(ctx context.Context, input any) (string, error) {
	// The user turn is part of the log, so every provider sees one shape and a
	// resumed session needs nothing but its history.
	entry, err := NewUserEntry(input)
	if err != nil {
		return "", err
	}
	a.logs = append(a.logs, entry)
	log := a.logs

	// ---> Notify start
	for range a.maxTurns {
		var produced []Entry

		switch a.provider {
		case ProviderAnthropic:
			produced, err = a.anthropicStep(ctx, log)
			if err != nil {
				return "", err
			}
		case ProviderOpenAI:
			produced, err = a.openAIstep(ctx, log)
			if err != nil {
				return "", err
			}
		case ProviderOpenrouter:
			produced, err = a.openrouterStep(ctx, log)
			if err != nil {
				return "", err
			}
		case ProviderGoogle:
			produced, err = a.geminiStep(ctx, log)
			if err != nil {
				return "", err
			}
		case ProviderXAI:
			produced, err = a.xaiStep(ctx, log)
			if err != nil {
				return "", err
			}
		case ProviderDeepSeek:
			produced, err = a.deepseekStep(ctx, log)
			if err != nil {
				return "", err
			}
		case ProviderOllama:
			produced, err = a.ollamaStep(ctx, log)
			if err != nil {
				return "", err
			}
		default:
			return "", fmt.Errorf("unsupported provider %q", a.provider)
		}

		a.logs = append(a.logs, produced...)
		results := a.dispatchCalls(ctx, produced)
		a.logs = append(a.logs, results...)
		log = a.logs

		// Local tool results require another model step.
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
func (a *Agent) dispatch(ctx context.Context, call ToolCall) []Entry {
	result := ToolResult{CallID: call.ID}
	index := slices.IndexFunc(a.tools, func(tool Tool) bool { return tool.name == call.Name })
	if index < 0 {
		result.Error = fmt.Sprintf("tool %q is not allowed", call.Name)
		return []Entry{{Kind: KindToolResult, ToolResult: &result}}
	}

	var resp []Entry

	tool := a.tools[index]
	snapshot := a.StateSnapshot()

	output, delta, err := tool.invoke(ContextWithState(ctx, snapshot), call.Args)
	if err != nil {
		result.Error = err.Error()
	} else {
		result.Output = output
	}

	if delta != nil {
		resp = append(resp, Entry{Kind: KindStateDelta, ToolResult: &result, Delta: delta})
	}

	return append(resp, Entry{Kind: KindToolResult, ToolResult: &result})
}

// dispatchCalls executes local calls in model order. The caller appends the
// results after all model entries, as required by the provider protocols.
func (a *Agent) dispatchCalls(ctx context.Context, entries []Entry) []Entry {
	var results []Entry
	for _, e := range entries {
		if e.Kind == KindToolCall {
			if e.ToolCall == nil {
				continue
			}

			results = append(results, a.dispatch(ctx, *e.ToolCall)...)
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
