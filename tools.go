package crux

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"
)

type Tool struct {
	name        string
	description string
	schema      map[string]any
	invoke      func(ctx context.Context, args json.RawMessage) (string, error)
}

type ToolsRegistry struct {
	mu    sync.Mutex
	tools map[string]Tool
}

func NewToolsRegistry() *ToolsRegistry {
	return &ToolsRegistry{
		tools: make(map[string]Tool),
	}
}

var defaultToolsRegistry = NewToolsRegistry()

func RegisterTool[In, Out any](name string, description string, fn func(ctx context.Context, input In) (Out, error)) {
	RegisterToolWithRegistry(defaultToolsRegistry, name, description, fn)
}

func RegisterToolWithRegistry[In, Out any](registry *ToolsRegistry, name string, description string, fn func(ctx context.Context, input In) (Out, error)) {
	tool := Tool{
		name:        name,
		description: description,
		schema:      jsonSchemaOf[In](),
		invoke: func(ctx context.Context, args json.RawMessage) (string, error) {
			var input In
			if len(args) > 0 {
				if err := json.Unmarshal(args, &input); err != nil {
					return "", fmt.Errorf("decode arguments for %q: %w", name, err)
				}
			}

			output, err := fn(ctx, input)
			if err != nil {
				return "", err
			}
			return renderToolOutput(output)
		},
	}

	registry.mu.Lock()
	defer registry.mu.Unlock()

	registry.tools[name] = tool
}

func (r *ToolsRegistry) lookup(name string) (Tool, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	tool, ok := r.tools[name]
	return tool, ok
}

// selected returns registered tools in the order of the given names.
// Unknown names are skipped and repeated names are included only once.
func (r *ToolsRegistry) selected(names []string) []Tool {
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
			continue
		}
		tools = append(tools, tool)
	}

	return tools
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
// place a tool's parameters are declared. Field names come from the json tag
// and a field is optional when it is a pointer or tagged omitempty; a
// `description` tag is passed on to the model.
func jsonSchema(t reflect.Type) map[string]any {
	switch t.Kind() {
	case reflect.Pointer:
		return jsonSchema(t.Elem())
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Slice, reflect.Array:
		return map[string]any{"type": "array", "items": jsonSchema(t.Elem())}
	case reflect.Map:
		return map[string]any{"type": "object", "additionalProperties": jsonSchema(t.Elem())}
	case reflect.Struct:
		return structSchema(t)
	default:
		return map[string]any{}
	}
}

func structSchema(t reflect.Type) map[string]any {
	properties := make(map[string]any)
	required := make([]string, 0, t.NumField())

	for i := range t.NumField() {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}

		name, optional, ok := fieldName(field)
		if !ok {
			continue
		}

		property := jsonSchema(field.Type)
		if description := field.Tag.Get("description"); description != "" {
			property["description"] = description
		}
		properties[name] = property

		if !optional && field.Type.Kind() != reflect.Pointer {
			required = append(required, name)
		}
	}

	schema := map[string]any{"type": "object", "properties": properties}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema
}

func fieldName(field reflect.StructField) (name string, optional bool, ok bool) {
	tag := field.Tag.Get("json")
	if tag == "-" {
		return "", false, false
	}

	name, options, _ := strings.Cut(tag, ",")
	if name == "" {
		name = field.Name
	}
	return name, slices.Contains(strings.Split(options, ","), "omitempty"), true
}
