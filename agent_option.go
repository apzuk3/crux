package crux

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"slices"
	"strings"

	"github.com/google/uuid"
	"github.com/invopop/jsonschema"
)

// AgentOption configures an Agent in New.
type AgentOption func(*Agent) error

// WithAgentID overrides the agent's ID, which otherwise derives from its configuration.
func WithAgentID(id uuid.UUID) AgentOption {
	return func(a *Agent) error { a.id = id; return nil }
}

// SessionOption configures a Session in NewSession.
type SessionOption func(*Session) error

// WithSessionID sets the session ID. If the store already holds a session with
// this ID, NewSession loads its history and the conversation continues.
func WithSessionID(id uuid.UUID) SessionOption {
	return func(s *Session) error {
		s.id = id
		return nil
	}
}

// WithSessionLogs seeds a new session with existing history.
func WithSessionLogs(logs []Entry) SessionOption {
	return func(s *Session) error { s.logs = cloneEntries(logs); return nil }
}

// WithStore persists the session in store. The default is a MemoryStore.
func WithStore(store Store) SessionOption {
	return func(s *Session) error {
		if store == nil {
			return errors.New("store cannot be nil")
		}
		s.store = store
		return nil
	}
}

// WithSessionHTTPClient overrides the agent's HTTP client for this session.
func WithSessionHTTPClient(client *http.Client) SessionOption {
	return func(s *Session) error { s.httpClient = client; return nil }
}

// WithTools replaces existing registry tools with the selected ones, preserving subagents.
func WithTools(tools []string) AgentOption {
	return func(a *Agent) error {
		selected, err := defaultToolsRegistry.selected(tools)
		if err != nil {
			return err
		}

		a.tools = slices.DeleteFunc(a.tools, func(t Tool) bool {
			return t.kind == toolKindTool
		})
		a.tools = append(a.tools, selected...)

		return nil
	}
}

// WithoutTools removes all registry tools, preserving subagents.
func WithoutTools() AgentOption {
	return func(a *Agent) error {
		a.tools = slices.DeleteFunc(a.tools, func(t Tool) bool {
			return t.kind == toolKindTool
		})
		return nil
	}
}

// WithoutSubagents removes all subagents, preserving registry tools.
func WithoutSubagents() AgentOption {
	return func(a *Agent) error {
		a.tools = slices.DeleteFunc(a.tools, func(t Tool) bool {
			return t.kind == toolKindSubagent
		})
		return nil
	}
}

// WithToolsRegistry replaces existing registry tools using a custom registry, preserving subagents.
func WithToolsRegistry(tools []string, registry ToolsRegistry) AgentOption {
	return func(a *Agent) error {
		selected, err := registry.selected(tools)
		if err != nil {
			return err
		}

		a.tools = slices.DeleteFunc(a.tools, func(t Tool) bool {
			return t.kind == toolKindTool
		})
		a.tools = append(a.tools, selected...)

		return nil
	}
}

// WithToolsets adds every tool labelled with one of the toolset names (see
// WithToolset) from the default registry. WithTools replaces the agent's tool
// list, so put WithToolsets after it when using both.
func WithToolsets(names ...string) AgentOption {
	return WithToolsetsRegistry(defaultToolsRegistry, names...)
}

// WithToolsetsRegistry adds every tool labelled with one of the toolset names
// from a custom registry.
func WithToolsetsRegistry(registry ToolsRegistry, names ...string) AgentOption {
	return func(a *Agent) error {
		selected, err := registry.inToolsets(names)
		if err != nil {
			return err
		}

		for _, tool := range selected {
			if !slices.ContainsFunc(a.tools, func(t Tool) bool { return t.name == tool.name }) {
				a.tools = append(a.tools, tool)
			}
		}

		return nil
	}
}

func WithInstructions(instructions string) AgentOption {
	return func(a *Agent) error { a.instructions = instructions; return nil }
}

// WithUserLocation supplies geographic context using the fields supported by
// the provider. Gemini uses paired coordinates; OpenAI and Anthropic use the
// named location fields. Providers without location support ignore it.
func WithUserLocation(location UserLocation) SearchOption {
	return func(opts *SearchOptions) {
		value := location
		if location.Latitude != nil {
			latitude := *location.Latitude
			value.Latitude = &latitude
		}
		if location.Longitude != nil {
			longitude := *location.Longitude
			value.Longitude = &longitude
		}
		opts.UserLocation = &value
	}
}

// WithWebSearch enables provider-executed search on supported OpenAI,
// Anthropic, Gemini, and xAI models. The model decides when to search.
// Each call replaces the search configuration; no options means no location.
func WithWebSearch(opts ...SearchOption) AgentOption {
	return func(a *Agent) error {
		search := &SearchOptions{}
		for _, opt := range opts {
			opt(search)
		}
		a.searchOptions = search

		return nil
	}
}

// WithProvider sets the provider explicitly. It is required when the model
// is not one of the known models for a provider.
func WithProvider(provider Provider) AgentOption {
	return func(a *Agent) error { a.provider = provider; return nil }
}

// WithModel changes the model. When forking across providers, also set
// WithProvider and the destination's connection settings.
func WithModel(model string) AgentOption {
	return func(a *Agent) error { a.model = model; return nil }
}

// WithMaxTurns limits how many model requests one Run may make. The default is 10.
func WithMaxTurns(turns int32) AgentOption {
	return func(a *Agent) error {
		if turns < 1 {
			return fmt.Errorf("max turns must be at least 1, got %d", turns)
		}
		a.maxTurns = turns
		return nil
	}
}

func WithBaseURL(url string) AgentOption {
	return func(a *Agent) error { a.baseURL = url; return nil }
}

// WithHTTPClient configures a custom HTTP client for API requests across all providers.
func WithHTTPClient(client *http.Client) AgentOption {
	return func(a *Agent) error { a.httpClient = client; return nil }
}

func WithAPIKey(apikey string) AgentOption {
	return func(a *Agent) error { a.apiKey = apikey; return nil }
}

// WithOutputSchema sets the response schema. Crux automatically adapts the schema
// for each provider's wire requirements (strict object closure, property nullability,
// and constraint placement).
func WithOutputSchema(schema *jsonschema.Schema) AgentOption {
	return func(a *Agent) error { a.outputSchema = schema; return nil }
}

// WithMaxTokens caps the tokens the model may generate per request. Zero uses
// the provider default; Anthropic requires a cap and defaults to 16384.
func WithMaxTokens(tokens int) AgentOption {
	return func(a *Agent) error {
		if tokens < 0 {
			return fmt.Errorf("max tokens cannot be negative, got %d", tokens)
		}
		a.maxTokens = tokens
		return nil
	}
}

// WithTemperature sets the sampling temperature. Unset uses the provider
// default. Some reasoning models reject a temperature.
func WithTemperature(temperature float64) AgentOption {
	return func(a *Agent) error {
		if temperature < 0 {
			return fmt.Errorf("temperature cannot be negative, got %v", temperature)
		}
		a.temperature = &temperature
		return nil
	}
}

// WithMaxRepairs sets the number of attempts the agent will make
// to ask the model to repair its response if output validation fails.
func WithMaxRepairs(repairs int) AgentOption {
	return func(a *Agent) error {
		if repairs < 0 {
			return fmt.Errorf("max repairs cannot be negative, got %d", repairs)
		}
		a.maxRepairs = repairs
		return nil
	}
}

type subAgentInput struct {
	Task string `json:"task"`
}

var subAgentInputSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"task": map[string]any{
			"type":        "string",
			"description": "Everything the subagent needs to do the work: the task and any input it applies to.",
		},
	},
	"required": []string{"task"},
}

var outputReflector = &jsonschema.Reflector{
	Anonymous:      true,
	ExpandedStruct: true,
}

func makeOptionalNullable(schema *jsonschema.Schema) {
	if schema == nil {
		return
	}
	seen := make(map[*jsonschema.Schema]bool)
	var visit func(*jsonschema.Schema)
	visit = func(s *jsonschema.Schema) {
		if s == nil || seen[s] {
			return
		}
		seen[s] = true
		if s.Properties != nil {
			reqSet := make(map[string]bool, len(s.Required))
			for _, r := range s.Required {
				reqSet[r] = true
			}
			for pair := s.Properties.Oldest(); pair != nil; pair = pair.Next() {
				visit(pair.Value)
				if !reqSet[pair.Key] {
					s.Properties.Set(pair.Key, &jsonschema.Schema{
						AnyOf: []*jsonschema.Schema{pair.Value, {Type: "null"}},
					})
				}
			}
		}
		if s.Definitions != nil {
			for _, def := range s.Definitions {
				visit(def)
			}
		}
		if s.Items != nil {
			visit(s.Items)
		}
		for _, anyOf := range s.AnyOf {
			visit(anyOf)
		}
		for _, oneOf := range s.OneOf {
			visit(oneOf)
		}
		for _, allOf := range s.AllOf {
			visit(allOf)
		}
	}
	visit(schema)
}

// WithOutputSchemaFrom reflects T into a response schema, or disables structured
// output for string and any. Optional properties also permit null.
func WithOutputSchemaFrom[T any]() AgentOption {
	return func(a *Agent) error {
		var zero T
		switch any(&zero).(type) {
		case *string, *any:
			a.outputSchema = nil
		default:
			schema := outputReflector.ReflectFromType(reflect.TypeFor[T]())
			schema.Version = ""
			makeOptionalNullable(schema)
			a.outputSchema = schema
		}

		return nil
	}
}

// WithSubAgent exposes subAgent as a tool named "agent_<name>". The parent
// passes a task as text; each call runs in a fresh session of subAgent, and its
// final output, shaped by subAgent's output schema if it has one, becomes the
// tool result. description tells the parent model what the subagent does.
func WithSubAgent(subAgent *Agent, description string) AgentOption {
	return func(parent *Agent) error {
		if subAgent == nil {
			return errors.New("subagent cannot be nil")
		}
		name := "agent_" + subAgent.name
		if err := validateToolName(name); err != nil {
			return fmt.Errorf("subagent %q: %w", subAgent.name, err)
		}

		tool := Tool{
			name:        name,
			description: description,
			schema:      subAgentInputSchema,
			kind:        toolKindSubagent,
			invoke: func(ctx context.Context, args json.RawMessage) (string, *StateDelta, error) {
				input, err := decodeToolArgs[subAgentInput](name, args, subAgentArgsValidator())
				if err != nil {
					return "", nil, err
				}
				if strings.TrimSpace(input.Task) == "" {
					return "", nil, fmt.Errorf("subagent %q needs a non-empty task", subAgent.name)
				}

				sess, err := NewSession(ctx, subAgent)
				if err != nil {
					return "", nil, err
				}
				output, err := sess.Run(ctx, input.Task)
				if err != nil {
					return "", nil, err
				}

				return output, &StateDelta{Set: map[string]any{subAgent.name: output}}, nil
			},
		}

		parent.tools = append(parent.tools, tool)

		return nil
	}
}
