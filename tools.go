package crux

import (
	"bytes"
	"context"
	"encoding"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"regexp"
	"slices"
	"strings"
	"sync"
	"time"

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
	schemaErr      error  // from WithInputSchema, reported at registration
	subAgent       *Agent // set for WithSubAgent tools
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

func inputSchemaFrom(schema any) (map[string]any, error) {
	raw, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("encode input schema: %w", err)
	}
	var object map[string]any
	if err := json.Unmarshal(raw, &object); err != nil || object == nil {
		return nil, fmt.Errorf("input schema must be a JSON object, got %s", raw)
	}
	if object["type"] != "object" {
		// Providers only accept tools whose arguments are a JSON object.
		return nil, fmt.Errorf(`input schema must have "type": "object", got %v`, object["type"])
	}
	return object, nil
}

// decodesObject reports whether encoding/json can decode a JSON object into t.
func decodesObject(t reflect.Type) bool {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.Struct, reflect.Map, reflect.Interface:
		return true
	}
	return reflect.PointerTo(t).Implements(jsonUnmarshalerType)
}

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
	if err := validateToolName(name); err != nil {
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
		if !decodesObject(in) {
			panic(toolRegistrationError{fmt.Errorf("tool %q: input type %s cannot hold a JSON object; use a struct, a map or json.RawMessage", name, in)})
		}
	default:
		tool.schema = jsonSchemaOf[In]()
		if tool.schema["type"] != "object" {
			// Providers only accept tools whose arguments are a JSON object.
			if in.Kind() == reflect.Struct {
				panic(toolRegistrationError{fmt.Errorf("tool %q: input type %s must decode from a JSON object field by field; its UnmarshalText or embedded decoder does not", name, in)})
			}
			panic(toolRegistrationError{fmt.Errorf("tool %q: input type %s must be a struct or a map", name, in)})
		}
	}
	validator, err := compileToolSchema(tool.schema)
	if err != nil {
		panic(toolRegistrationError{fmt.Errorf("tool %q: %w", name, err)})
	}
	tool.invoke = func(ctx context.Context, args json.RawMessage) (string, *StateDelta, error) {
		input, err := decodeToolArgs[In](name, args, validator)
		if err != nil {
			return "", nil, err
		}

		output, delta, err := fn(ctx, input)
		if err != nil {
			return "", nil, err
		}
		outputStr, err := renderToolOutput(output)

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

// decodeToolArgs checks args with checkToolArgs and decodes them into In.
// Empty arguments are read as an empty object.
func decodeToolArgs[In any](name string, args json.RawMessage, validator *sjs.Schema) (In, error) {
	var input In
	if len(bytes.TrimSpace(args)) == 0 {
		args = json.RawMessage(`{}`) // still checked for required arguments
	}
	var malformed string
	if args[0] == '"' && json.Unmarshal(args, &malformed) == nil {
		return input, fmt.Errorf("arguments for %q are not valid JSON: %s", name, malformed)
	}
	if err := checkToolArgs(args, reflect.TypeFor[In](), validator); err != nil {
		return input, fmt.Errorf("invalid arguments for %q: %w", name, err)
	}
	if err := json.Unmarshal(args, &input); err != nil {
		return input, fmt.Errorf("decode arguments for %q: %w", name, err)
	}
	return input, nil
}

// subAgentArgsValidator validates the arguments of a WithSubAgent tool.
var subAgentArgsValidator = sync.OnceValue(func() *sjs.Schema {
	validator, err := compileToolSchema(subAgentInputSchema)
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

var toolNamePattern = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

func validateToolName(name string) error {
	if !toolNamePattern.MatchString(name) {
		return fmt.Errorf("invalid tool name %q: use 1-64 letters, digits, '_' or '-'", name)
	}
	return nil
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
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	// An input struct with its own UnmarshalJSON (often to fill in defaults)
	// still describes its arguments with its fields.
	if t.Kind() == reflect.Struct && t != timeType && !promotesUnmarshaler(t) &&
		(reflect.PointerTo(t).Implements(jsonUnmarshalerType) || !reflect.PointerTo(t).Implements(textUnmarshalerType)) {
		return structSchema(t, map[reflect.Type]bool{t: true})
	}
	return schemaOf(t, make(map[reflect.Type]bool))
}

// promotesUnmarshaler reports whether struct t gets UnmarshalJSON or
// UnmarshalText from an embedded field. Decoding then fills only that field,
// so t's own fields do not describe its arguments.
func promotesUnmarshaler(t reflect.Type) bool {
	for i := range t.NumField() {
		field := t.Field(i)
		if !field.Anonymous {
			continue
		}
		for _, iface := range []reflect.Type{jsonUnmarshalerType, textUnmarshalerType} {
			if field.Type.Implements(iface) || reflect.PointerTo(field.Type).Implements(iface) {
				return true
			}
		}
	}
	return false
}

// decodesItself reports whether values of t are decoded by their own
// UnmarshalJSON or UnmarshalText rather than field by field.
func decodesItself(t reflect.Type) bool {
	p := reflect.PointerTo(t)
	return p.Implements(jsonUnmarshalerType) || p.Implements(textUnmarshalerType)
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

	// A type with its own decoder, such as an int enum read from names or
	// slog.Level, is described by what that decoder accepts, not by its kind.
	switch {
	case t == timeType:
		return map[string]any{"type": "string", "format": "date-time"}
	case reflect.PointerTo(t).Implements(jsonUnmarshalerType):
		return map[string]any{} // decoded by its own UnmarshalJSON
	case reflect.PointerTo(t).Implements(textUnmarshalerType):
		return map[string]any{"type": "string"}
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
		// encoding/json drops extra elements and zero-fills missing ones.
		return map[string]any{"type": "array", "items": schemaOf(t.Elem(), visiting), "minItems": t.Len(), "maxItems": t.Len()}
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
	typ      reflect.Type
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
			typ:      field.Type,
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

func compileToolSchema(schema map[string]any) (*sjs.Schema, error) {
	raw, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("marshal input schema: %w", err)
	}
	compiler := sjs.NewCompiler()
	if err := compiler.AddResource("tool.json", bytes.NewReader(raw)); err != nil {
		return nil, fmt.Errorf("load input schema: %w", err)
	}
	validator, err := compiler.Compile("tool.json")
	if err != nil {
		return nil, fmt.Errorf("compile input schema: %w", err)
	}
	return validator, nil
}

// checkToolArgs rejects arguments that would decode differently from how they
// read: repeated keys (the last one wins in encoding/json), keys that only
// match a field when case is ignored, and keys of a struct that match no field
// at all (encoding/json ignores them). Every key of a struct argument therefore
// names the field it fills, so an approval shown from the raw arguments
// describes exactly what the tool receives. Maps, and nested types with their
// own UnmarshalJSON or UnmarshalText, accept any key. It then validates the
// arguments against the input schema, treating null like a missing value as
// encoding/json does.
func checkToolArgs(args json.RawMessage, input reflect.Type, validator *sjs.Schema) error {
	dec := json.NewDecoder(bytes.NewReader(args))
	dec.UseNumber()
	doc, err := decodeStrictJSON(dec)
	if err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		return errors.New("unexpected data after the arguments object")
	}
	doc, err = normalizeArgs(doc, input, "")
	if err != nil {
		return err
	}
	if err := validator.Validate(doc); err != nil {
		var verr *sjs.ValidationError
		if errors.As(err, &verr) {
			return errors.New(strings.Join(validationMessages(verr), "; "))
		}
		return err
	}
	return nil
}

// decodeStrictJSON decodes one JSON value, failing on a repeated object key.
func decodeStrictJSON(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	delim, ok := tok.(json.Delim)
	if !ok {
		return tok, nil
	}
	switch delim {
	case '{':
		object := make(map[string]any)
		for dec.More() {
			keyTok, err := dec.Token()
			if err != nil {
				return nil, err
			}
			key := keyTok.(string)
			if _, dup := object[key]; dup {
				return nil, fmt.Errorf("argument %q is given more than once", key)
			}
			if object[key], err = decodeStrictJSON(dec); err != nil {
				return nil, err
			}
		}
		_, err = dec.Token()
		return object, err
	case '[':
		var array []any
		for dec.More() {
			item, err := decodeStrictJSON(dec)
			if err != nil {
				return nil, err
			}
			array = append(array, item)
		}
		_, err = dec.Token()
		return array, err
	}
	return nil, fmt.Errorf("unexpected %v", delim)
}

// normalizeArgs drops null object members and rejects keys of a struct that
// match no field of the Go type t, including keys that differ only in case
// from a field, which encoding/json would decode into that field anyway. It
// follows the Go type rather than the schema, because the
// schema of a recursive type is cut off.
func normalizeArgs(value any, t reflect.Type, path string) (any, error) {
	for t != nil && t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t != nil && path != "" && decodesItself(t) {
		// A nested type with its own decoder may read any keys; the
		// top-level input's own UnmarshalJSON is checked by its fields.
		t = nil
	}
	switch v := value.(type) {
	case map[string]any:
		var fields map[string]reflect.Type
		var elem reflect.Type
		switch {
		case t == nil:
		case t.Kind() == reflect.Struct:
			fields = jsonFields(t)
		case t.Kind() == reflect.Map:
			elem = t.Elem()
		}
		for key, item := range v {
			sub, declared := fields[key]
			if !declared && fields != nil {
				for name := range fields {
					if strings.EqualFold(name, key) {
						return nil, fmt.Errorf("argument %q must be spelled %q", path+key, path+name)
					}
				}
				return nil, fmt.Errorf("unknown argument %q", path+key)
			}
			if item == nil {
				delete(v, key)
				continue
			}
			if !declared {
				sub = elem
			}
			normalized, err := normalizeArgs(item, sub, path+key+".")
			if err != nil {
				return nil, err
			}
			v[key] = normalized
		}
	case []any:
		var elem reflect.Type
		if t != nil && (t.Kind() == reflect.Slice || t.Kind() == reflect.Array) {
			elem = t.Elem()
		}
		for i, item := range v {
			normalized, err := normalizeArgs(item, elem, fmt.Sprintf("%s%d.", path, i))
			if err != nil {
				return nil, err
			}
			v[i] = normalized
		}
	}
	return value, nil
}

var jsonFieldsCache sync.Map // reflect.Type -> map[string]reflect.Type

// jsonFields returns the JSON names encoding/json decodes into for struct t,
// with the Go type of each field.
func jsonFields(t reflect.Type) map[string]reflect.Type {
	if cached, ok := jsonFieldsCache.Load(t); ok {
		return cached.(map[string]reflect.Type)
	}
	var all []schemaField
	collectFields(t, 0, true, map[reflect.Type]bool{t: true}, &all)
	byName := make(map[string][]schemaField)
	for _, f := range all {
		byName[f.name] = append(byName[f.name], f)
	}
	fields := make(map[string]reflect.Type, len(byName))
	for name, candidates := range byName {
		if f, ok := dominantField(candidates); ok {
			fields[name] = f.typ
		}
	}
	jsonFieldsCache.Store(t, fields)
	return fields
}

func validationMessages(err *sjs.ValidationError) []string {
	if len(err.Causes) == 0 {
		location := err.InstanceLocation
		if location == "" {
			location = "arguments"
		}
		return []string{location + ": " + err.Message}
	}
	var messages []string
	for _, cause := range err.Causes {
		messages = append(messages, validationMessages(cause)...)
	}
	return messages
}
