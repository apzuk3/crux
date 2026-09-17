package crux

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"

	"github.com/invopop/jsonschema"
)

type AgentConfig struct {
}

type AgentOption func(*Agent) error

// WithTools replaces existing registry tools with the selected ones, preserving subagents.
func WithTools(tools []string) AgentOption {
	return func(a *Agent) error {
		selected, err := defaultToolsRegistry.selected(tools)
		if err != nil {
			return err
		}

		a.tools = slices.DeleteFunc(a.tools, func(t Tool) bool {
			return t.kind == ToolKindTool
		})
		a.tools = append(a.tools, selected...)

		return nil
	}
}

// WithoutTools removes all registry tools, preserving subagents.
func WithoutTools() AgentOption {
	return func(a *Agent) error {
		a.tools = slices.DeleteFunc(a.tools, func(t Tool) bool {
			return t.kind == ToolKindTool
		})
		return nil
	}
}

// WithoutSubagents removes all subagents, preserving registry tools.
func WithoutSubagents() AgentOption {
	return func(a *Agent) error {
		a.tools = slices.DeleteFunc(a.tools, func(t Tool) bool {
			return t.kind == ToolKindSubagent
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
			return t.kind == ToolKindTool
		})
		a.tools = append(a.tools, selected...)

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

func WithProvider(provider Provider) AgentOption {
	return func(a *Agent) error { a.provider = provider; return nil }
}

// WithModel changes the model. When forking across providers, also set
// WithProvider and the destination's connection settings.
func WithModel(model string) AgentOption {
	return func(a *Agent) error { a.model = model; return nil }
}

func WithMaxTurns(turns int32) AgentOption {
	return func(a *Agent) error { a.maxTurns = turns; return nil }
}

func WithBaseURL(url string) AgentOption {
	return func(a *Agent) error { a.baseURL = url; return nil }
}

func WithAPIKey(apikey string) AgentOption {
	return func(a *Agent) error { a.apikey = apikey; return nil }
}

// WithOutputSchema sets the response schema. For OpenAI, it must satisfy strict
// Structured Outputs requirements: an object root, all properties required, and
// additionalProperties: false on every object.
// For Anthropic, the schema must satisfy its supported JSON Schema subset;
// it is sent unchanged through output_config.format.
// For Gemini, the schema must satisfy its supported JSON Schema subset;
// it is sent unchanged through responseJsonSchema.
// xAI, DeepSeek, and local Ollama use the OpenAI Responses schema format.
// Local Ollama support is verified in v0.34.0. Ollama Cloud does not currently
// support structured outputs.
func WithOutputSchema(schema *jsonschema.Schema) AgentOption {
	return func(a *Agent) error { a.outputSchema = schema; return nil }
}

// WithOutputSchemaFrom reflects T into a response schema, or disables structured
// output for string and any. T must satisfy WithOutputSchema's requirements;
// reflection does not normalize optional fields or maps for OpenAI strict mode.
func WithOutputSchemaFrom[T any]() AgentOption {
	return func(a *Agent) error {
		var zero T
		switch any(&zero).(type) {
		case *string, *any:
			a.outputSchema = nil
		default:
			a.outputSchema = jsonschema.ReflectFromType(reflect.TypeFor[T]())
		}

		return nil
	}
}

// WithSubAgent exposes subAgent as a tool, using its output schema for tool
// arguments as well. It panics if the schema cannot be represented as an object.
// description - one liner description of that agent does.
func WithSubAgent(subAgent *Agent, description string) AgentOption {
	return func(parent *Agent) error {
		schema := map[string]any{"type": "object", "properties": map[string]any{}}
		if subAgent.outputSchema != nil {
			schema = nil
			raw, err := json.Marshal(subAgent.outputSchema)
			if err != nil {
				panic(fmt.Errorf("encode schema for subagent %q: %w", subAgent.name, err))
			}
			if err := json.Unmarshal(raw, &schema); err != nil {
				panic(fmt.Errorf("decode schema for subagent %q: %w", subAgent.name, err))
			}
		}

		tool := Tool{
			name:        "agent_" + subAgent.name,
			description: description,
			schema:      schema,
			kind:        ToolKindSubagent,
			invoke: func(ctx context.Context, args json.RawMessage) (string, *StateDelta, error) {
				output, err := subAgent.Run(ctx, string(args))
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
