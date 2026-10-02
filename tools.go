package crux

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"

	"github.com/apzuk3/crux/internal/schema"
	sjs "github.com/santhosh-tekuri/jsonschema/v5"
)

type toolKind string

const (
	toolKindTool     toolKind = "fn"
	toolKindSubagent toolKind = "agent"
)

// Tool is a function the model can call. Tools are created by RegisterTool and
// its variants and selected by agents by name.
type Tool struct {
	name           string
	kind           toolKind
	description    string
	schema         map[string]any
	invoke         func(ctx context.Context, args json.RawMessage) (string, *StateDelta, error)
	approvalNeeded bool
	toolset        string
	instructions   *toolInstructions // added to the agent's instructions on each request
	schemaErr      error             // from WithInputSchema, reported at registration
	subAgent       *Agent            // set for WithSubAgent tools
}

type ToolOption func(*Tool)

func WithApprovalNeeded(approvalNeeded bool) ToolOption {
	return func(tool *Tool) {
		tool.approvalNeeded = approvalNeeded
	}
}

// WithToolset labels the tool as part of the named toolset, so agents can
// select every tool in it with WithToolsets.
func WithToolset(name string) ToolOption {
	return func(tool *Tool) {
		tool.toolset = name
	}
}

// toolInstructions produces text for the instructions of each request.
type toolInstructions struct {
	text func() (string, error)
}

// withInstructions makes the tool add fn's text to the instructions of every
// request from an agent that has the tool. fn runs before each request, so
// the text can change between turns; it is not part of the agent's ID. Give
// one option to several tools to add the text once for an agent with any of
// them.
func withInstructions(fn func() (string, error)) ToolOption {
	instructions := &toolInstructions{text: fn}
	return func(tool *Tool) {
		tool.instructions = instructions
	}
}

// WithInputSchema gives the tool a JSON Schema for its arguments in place of
// the one generated from its input type, for tools whose arguments are only
// known at run time, such as the tools of an MCP server. schema is anything
// that encodes to a JSON Schema object, such as json.RawMessage,
// map[string]any or *jsonschema.Schema. Arguments are validated against it
// before the tool runs. Take them as json.RawMessage or map[string]any to
// receive whatever the schema allows; a struct input still rejects keys that
// match none of its fields.
func WithInputSchema(schema any) ToolOption {
	return func(tool *Tool) {
		tool.schema, tool.schemaErr = inputSchemaFrom(schema)
	}
}

// inputSchemaFrom is schema.FromValue, which WithInputSchema's parameter hides.
var inputSchemaFrom = schema.FromValue

// ToolsRegistry holds registered tools. Create one with NewToolsRegistry; the
// zero value is not usable.
type ToolsRegistry struct {
	mu    *sync.Mutex
	tools map[string]Tool
}

// NewToolsRegistry returns an empty registry, for tools kept apart from the
// default one. Register tools in it with RegisterToolWithRegistry and give it
// to agents with WithToolsRegistry.
func NewToolsRegistry() ToolsRegistry {
	return ToolsRegistry{
		mu:    &sync.Mutex{},
		tools: make(map[string]Tool),
	}
}

var defaultToolsRegistry = NewToolsRegistry()

// RegisterTool registers fn as a tool in the default registry, so agents can
// select it by name with WithTools. Its arguments are described and validated
// by In; see RegisterToolWithRegistry for when it panics.
func RegisterTool[In, Out any](name string, description string, fn func(ctx context.Context, input In) (Out, error), opts ...ToolOption) {
	RegisterToolWithRegistry(defaultToolsRegistry, name, description, func(ctx context.Context, input In) (Out, *StateDelta, error) {
		output, err := fn(ctx, input)

		return output, nil, err
	}, opts...)
}

// RegisterToolStateMutate is like RegisterTool, but fn also returns a
// StateDelta that is applied to the session state after the call.
func RegisterToolStateMutate[In, Out any](name string, description string, fn func(ctx context.Context, input In) (Out, *StateDelta, error), opts ...ToolOption) {
	RegisterToolWithRegistry(defaultToolsRegistry, name, description, fn, opts...)
}

// RegisterToolWithRegistry registers a tool in registry. It panics if name is
// not a valid tool name (letters, digits, '_' and '-', at most 64 characters)
// or is already registered, like http.HandleFunc does for a repeated pattern,
// and if In is not a struct or map, since tool arguments are a JSON object. A
// struct decoded by UnmarshalText, or by a decoder it gets from an embedded
// field, is rejected too, because its fields do not describe its arguments.
// With WithInputSchema, In may also be json.RawMessage, and an invalid schema
// panics.
func RegisterToolWithRegistry[In, Out any](registry ToolsRegistry, name string, description string, fn func(ctx context.Context, input In) (Out, *StateDelta, error), opts ...ToolOption) {
	if err := schema.ValidateToolName(name); err != nil {
		panic(toolRegistrationError{err})
	}
	tool := Tool{
		name:        name,
		description: description,
		kind:        toolKindTool,
	}
	for _, opt := range opts {
		opt(&tool)
	}
	in := reflect.TypeFor[In]()
	switch {
	case tool.schemaErr != nil:
		panic(toolRegistrationError{fmt.Errorf("tool %q: %w", name, tool.schemaErr)})
	case tool.schema != nil:
		if !schema.DecodesObject(in) {
			panic(toolRegistrationError{fmt.Errorf("tool %q: input type %s cannot hold a JSON object; use a struct, a map or json.RawMessage", name, in)})
		}
	default:
		tool.schema = schema.Of[In]()
		if tool.schema["type"] != "object" {
			// Providers only accept tools whose arguments are a JSON object.
			if in.Kind() == reflect.Struct {
				panic(toolRegistrationError{fmt.Errorf("tool %q: input type %s must decode from a JSON object field by field; its UnmarshalText or embedded decoder does not", name, in)})
			}
			panic(toolRegistrationError{fmt.Errorf("tool %q: input type %s must be a struct or a map", name, in)})
		}
	}
	validator, err := schema.Compile(tool.schema)
	if err != nil {
		panic(toolRegistrationError{fmt.Errorf("tool %q: %w", name, err)})
	}
	tool.invoke = func(ctx context.Context, args json.RawMessage) (string, *StateDelta, error) {
		input, err := schema.DecodeArgs[In](name, args, validator)
		if err != nil {
			return "", nil, err
		}

		output, delta, err := fn(ctx, input)
		if err != nil {
			return "", nil, err
		}
		outputStr, err := schema.RenderOutput(output)

		return outputStr, delta, err
	}

	registry.mustBeInitialized()
	registry.mu.Lock()
	defer registry.mu.Unlock()

	if _, exists := registry.tools[name]; exists {
		panic(toolRegistrationError{fmt.Errorf("tool %q is already registered", name)})
	}
	registry.tools[name] = tool
}

// subAgentArgsValidator validates the arguments of a WithSubAgent tool.
var subAgentArgsValidator = sync.OnceValue(func() *sjs.Schema {
	validator, err := schema.Compile(subAgentInputSchema)
	if err != nil {
		panic(err)
	}
	return validator
})

func (r *ToolsRegistry) mustBeInitialized() {
	if r.mu == nil {
		panic("crux: ToolsRegistry must be created with NewToolsRegistry")
	}
}

// toolRegistrationError is the panic value for an invalid registration, so
// AddToolset can return it as an error.
type toolRegistrationError struct{ err error }

func (e toolRegistrationError) Error() string { return e.err.Error() }
func (e toolRegistrationError) Unwrap() error { return e.err }

// Toolset registers a group of related tools into a registry, so they can be
// added together with AddToolset instead of one by one.
type Toolset interface {
	Register(registry ToolsRegistry) error
}

// AddToolset registers the toolset's tools into the default registry.
func AddToolset(toolset Toolset) error {
	return AddToolsetWithRegistry(defaultToolsRegistry, toolset)
}

// AddToolsetWithRegistry registers the toolset's tools into the given registry.
// An invalid or duplicate tool name is returned as an error.
func AddToolsetWithRegistry(registry ToolsRegistry, toolset Toolset) (err error) {
	defer func() {
		if r := recover(); r != nil {
			regErr, ok := r.(toolRegistrationError)
			if !ok {
				panic(r)
			}
			err = fmt.Errorf("register toolset %T: %w", toolset, regErr.err)
		}
	}()
	if err := toolset.Register(registry); err != nil {
		return fmt.Errorf("register toolset %T: %w", toolset, err)
	}

	return nil
}

// selected returns registered tools in the order of the given names.
// Repeated names are included only once. If any name is not registered,
// selected returns an error wrapping ErrToolNotFound with the missing tool name.
func (r *ToolsRegistry) selected(names []string) ([]Tool, error) {
	r.mustBeInitialized()
	r.mu.Lock()
	defer r.mu.Unlock()

	tools := make([]Tool, 0, len(names))
	seen := make(map[string]bool, len(names))
	for _, name := range names {
		if seen[name] {
			continue
		}
		seen[name] = true
		tool, ok := r.tools[name]
		if !ok {
			return nil, fmt.Errorf("%w: %q", ErrToolNotFound, name)
		}
		tools = append(tools, tool)
	}

	return tools, nil
}

// inToolsets returns the tools labelled with any of the given toolset names,
// ordered by toolset and then tool name. If a toolset has no tools, it returns
// an error wrapping ErrToolNotFound with the toolset name.
func (r *ToolsRegistry) inToolsets(names []string) ([]Tool, error) {
	r.mustBeInitialized()
	r.mu.Lock()
	defer r.mu.Unlock()

	var tools []Tool
	seen := make(map[string]bool, len(names))
	for _, name := range names {
		if seen[name] {
			continue
		}
		seen[name] = true
		var members []Tool
		for _, tool := range r.tools {
			if tool.toolset == name {
				members = append(members, tool)
			}
		}
		if len(members) == 0 {
			return nil, fmt.Errorf("%w: toolset %q", ErrToolNotFound, name)
		}
		slices.SortFunc(members, func(a, b Tool) int { return strings.Compare(a.name, b.name) })
		tools = append(tools, members...)
	}

	return tools, nil
}
