package crux

import (
	"context"
	"encoding"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"
	"time"
)

type ToolKind string

const (
	ToolKindTool     ToolKind = "fn"
	ToolKindSubagent ToolKind = "agent"
)

type Tool struct {
	name           string
	kind           ToolKind
	description    string
	schema         map[string]any
	invoke         func(ctx context.Context, args json.RawMessage) (string, *StateDelta, error)
	approvalNeeded bool
	toolset        string
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

type ToolsRegistry struct {
	mu    *sync.Mutex
	tools map[string]Tool
}

func NewToolsRegistry() ToolsRegistry {
	return ToolsRegistry{
		mu:    &sync.Mutex{},
		tools: make(map[string]Tool),
	}
}

var defaultToolsRegistry = NewToolsRegistry()

func RegisterTool[In, Out any](name string, description string, fn func(ctx context.Context, input In) (Out, error), opts ...ToolOption) {
	RegisterToolWithRegistry(defaultToolsRegistry, name, description, func(ctx context.Context, input In) (Out, *StateDelta, error) {
		output, err := fn(ctx, input)

		return output, nil, err
	}, opts...)
}

func RegisterToolStateMutate[In, Out any](name string, description string, fn func(ctx context.Context, input In) (Out, *StateDelta, error), opts ...ToolOption) {
	RegisterToolWithRegistry(defaultToolsRegistry, name, description, fn, opts...)
}

func RegisterToolWithRegistry[In, Out any](registry ToolsRegistry, name string, description string, fn func(ctx context.Context, input In) (Out, *StateDelta, error), opts ...ToolOption) {
	tool := Tool{
		name:        name,
		description: description,
		schema:      jsonSchemaOf[In](),
		kind:        ToolKindTool,
		invoke: func(ctx context.Context, args json.RawMessage) (string, *StateDelta, error) {
			var input In
			if len(args) > 0 {
				if err := json.Unmarshal(args, &input); err != nil {
					var malformed string
					if json.Unmarshal(args, &malformed) == nil {
						return "", nil, fmt.Errorf("arguments for %q are not valid JSON: %s", name, malformed)
					}
					return "", nil, fmt.Errorf("decode arguments for %q: %w", name, err)
				}
			}

			output, delta, err := fn(ctx, input)
			if err != nil {
				return "", nil, err
			}
			outputStr, err := renderToolOutput(output)

			return outputStr, delta, err
		},
	}
	for _, opt := range opts {
		opt(&tool)
	}

	registry.mu.Lock()
	defer registry.mu.Unlock()

	registry.tools[name] = tool
}

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
func AddToolsetWithRegistry(registry ToolsRegistry, toolset Toolset) error {
	if err := toolset.Register(registry); err != nil {
		return fmt.Errorf("register toolset %T: %w", toolset, err)
	}

	return nil
}

// selected returns registered tools in the order of the given names.
// Repeated names are included only once. If any name is not registered,
// selected returns an error wrapping ErrToolNotFound with the missing tool name.
func (r *ToolsRegistry) selected(names []string) ([]Tool, error) {
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

func renderToolOutput(output any) (string, error) {
	switch v := output.(type) {
	case string:
		return v, nil
	case fmt.Stringer:
		return v.String(), nil
	default:
		raw, err := json.Marshal(output)
		if err != nil {
			return "", fmt.Errorf("render tool output %T: %w", output, err)
		}
		return string(raw), nil
	}
}

func jsonSchemaOf[In any]() map[string]any {
	var zero In
	return jsonSchema(reflect.TypeOf(&zero).Elem())
}

// jsonSchema describes a tool's input type, so the Go signature stays the only
// place a tool's parameters are declared. It follows encoding/json: field
// names come from the json tag, embedded structs are flattened, and a field is
// optional when it is a pointer or tagged omitempty. A `description` tag is
// passed on to the model.
func jsonSchema(t reflect.Type) map[string]any {
	return schemaOf(t, make(map[reflect.Type]bool))
}

var (
	timeType            = reflect.TypeFor[time.Time]()
	jsonUnmarshalerType = reflect.TypeFor[json.Unmarshaler]()
	textUnmarshalerType = reflect.TypeFor[encoding.TextUnmarshaler]()
)

// schemaOf builds the schema for t. visiting holds the struct types being
// expanded, so a recursive type is cut off instead of recursing forever.
func schemaOf(t reflect.Type, visiting map[reflect.Type]bool) map[string]any {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}

	switch t.Kind() {
	case reflect.Struct, reflect.Slice, reflect.Array, reflect.Map:
		switch {
		case t == timeType:
			return map[string]any{"type": "string", "format": "date-time"}
		case reflect.PointerTo(t).Implements(jsonUnmarshalerType):
			return map[string]any{} // decoded by its own UnmarshalJSON
		case reflect.PointerTo(t).Implements(textUnmarshalerType):
			return map[string]any{"type": "string"}
		}
	}

	switch t.Kind() {
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Slice:
		if t.Elem().Kind() == reflect.Uint8 {
			return map[string]any{"type": "string", "description": "Base64-encoded bytes"}
		}
		return map[string]any{"type": "array", "items": schemaOf(t.Elem(), visiting)}
	case reflect.Array:
		return map[string]any{"type": "array", "items": schemaOf(t.Elem(), visiting)}
	case reflect.Map:
		return map[string]any{"type": "object", "additionalProperties": schemaOf(t.Elem(), visiting)}
	case reflect.Struct:
		if visiting[t] {
			return map[string]any{"type": "object"}
		}
		visiting[t] = true
		defer delete(visiting, t)
		return structSchema(t, visiting)
	default:
		return map[string]any{}
	}
}

type schemaField struct {
	name     string
	depth    int
	tagged   bool
	required bool
	schema   map[string]any
}

func structSchema(t reflect.Type, visiting map[reflect.Type]bool) map[string]any {
	var fields []schemaField
	collectFields(t, 0, true, visiting, &fields)

	// As in encoding/json, the shallowest field wins a name; among equally
	// shallow fields a single tagged one wins, and otherwise the name is dropped.
	byName := make(map[string][]schemaField)
	var order []string
	for _, f := range fields {
		if _, ok := byName[f.name]; !ok {
			order = append(order, f.name)
		}
		byName[f.name] = append(byName[f.name], f)
	}

	properties := make(map[string]any)
	required := make([]string, 0, len(order))
	for _, name := range order {
		field, ok := dominantField(byName[name])
		if !ok {
			continue
		}
		properties[name] = field.schema
		if field.required {
			required = append(required, name)
		}
	}

	schema := map[string]any{"type": "object", "properties": properties}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema
}

func collectFields(t reflect.Type, depth int, required bool, visiting map[reflect.Type]bool, out *[]schemaField) {
	for i := range t.NumField() {
		field := t.Field(i)
		tag := field.Tag.Get("json")
		if tag == "-" {
			continue
		}
		name, options, _ := strings.Cut(tag, ",")

		if field.Anonymous {
			ft := field.Type
			if ft.Kind() == reflect.Pointer {
				ft = ft.Elem()
			}
			if name == "" && ft.Kind() == reflect.Struct {
				if !visiting[ft] {
					visiting[ft] = true
					collectFields(ft, depth+1, required && field.Type.Kind() != reflect.Pointer, visiting, out)
					delete(visiting, ft)
				}
				continue
			}
			if !field.IsExported() && ft.Kind() != reflect.Struct {
				continue
			}
		} else if !field.IsExported() {
			continue
		}

		tagged := name != ""
		if !tagged {
			name = field.Name
		}
		opts := strings.Split(options, ",")

		property := schemaOf(field.Type, visiting)
		if slices.Contains(opts, "string") && isStringEncodable(field.Type) {
			property = map[string]any{"type": "string"}
		}
		if description := field.Tag.Get("description"); description != "" {
			property["description"] = description
		}

		*out = append(*out, schemaField{
			name:     name,
			depth:    depth,
			tagged:   tagged,
			required: required && !slices.Contains(opts, "omitempty") && field.Type.Kind() != reflect.Pointer,
			schema:   property,
		})
	}
}

func dominantField(fields []schemaField) (schemaField, bool) {
	minDepth := fields[0].depth
	for _, f := range fields {
		minDepth = min(minDepth, f.depth)
	}
	var shallow []schemaField
	for _, f := range fields {
		if f.depth == minDepth {
			shallow = append(shallow, f)
		}
	}
	if len(shallow) == 1 {
		return shallow[0], true
	}
	var tagged []schemaField
	for _, f := range shallow {
		if f.tagged {
			tagged = append(tagged, f)
		}
	}
	if len(tagged) == 1 {
		return tagged[0], true
	}
	return schemaField{}, false
}

// isStringEncodable reports whether the json ",string" option applies to t.
func isStringEncodable(t reflect.Type) bool {
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Float32, reflect.Float64:
		return true
	}
	return false
}
