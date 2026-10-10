// Package schema derives JSON schemas for tool inputs from Go types, checks
// and decodes tool arguments strictly, and adapts output schemas to each
// provider's structured output rules.
package schema

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"regexp"
	"slices"
	"strings"
	"sync"

	sjs "github.com/santhosh-tekuri/jsonschema/v5"
)

// FromValue turns a schema given as any JSON-encodable value into a map,
// requiring a JSON object schema of "type": "object".
func FromValue(schema any) (map[string]any, error) {
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

// DecodesObject reports whether encoding/json can decode a JSON object into t.
func DecodesObject(t reflect.Type) bool {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.Struct, reflect.Map, reflect.Interface:
		return true
	}
	return reflect.PointerTo(t).Implements(jsonUnmarshalerType)
}

// DecodeArgs checks args with checkToolArgs and decodes them into In.
// Empty arguments are read as an empty object.
func DecodeArgs[In any](name string, args json.RawMessage, validator *sjs.Schema) (In, error) {
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

var toolNamePattern = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// ValidateToolName checks a tool name against the providers' rules.
func ValidateToolName(name string) error {
	if !toolNamePattern.MatchString(name) {
		return fmt.Errorf("invalid tool name %q: use 1-64 letters, digits, '_' or '-'", name)
	}
	return nil
}

// RenderOutput turns a tool's output into the text sent to the model.
func RenderOutput(output any) (string, error) {
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

// Of returns the input schema of a tool whose arguments decode into In.
func Of[In any]() map[string]any {
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
	if m, ok := customSchema(t); ok {
		return m
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

// schemaOf builds the schema for t. visiting holds the struct types being
// expanded, so a recursive type is cut off instead of recursing forever.
func schemaOf(t reflect.Type, visiting map[reflect.Type]bool) map[string]any {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}

	// A type with its own decoder, such as an int enum read from names or
	// slog.Level, is described by what that decoder accepts, not by its kind.
	if m, ok := customSchema(t); ok {
		return m
	}
	if r, ok := ruleFor(t); ok {
		return r.asMap()
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
		if skipField(field) {
			continue
		}
		name, options, _ := strings.Cut(field.Tag.Get("json"), ",")
		if embedded, ok := embeddedStruct(field, name); ok {
			collectEmbedded(embedded, depth+1, required && field.Type.Kind() != reflect.Pointer, visiting, out)
			continue
		}

		tagged := name != ""
		if !tagged {
			name = field.Name
		}
		opts := strings.Split(options, ",")
		*out = append(*out, schemaField{
			typ:      field.Type,
			name:     name,
			depth:    depth,
			tagged:   tagged,
			required: required && !slices.Contains(opts, "omitempty") && field.Type.Kind() != reflect.Pointer,
			schema:   fieldSchema(field, opts, visiting),
		})
	}
}

// skipField reports whether encoding/json ignores field: it is tagged "-", or
// unexported and not an embedded struct.
func skipField(field reflect.StructField) bool {
	if field.Tag.Get("json") == "-" {
		return true
	}
	if !field.Anonymous {
		return !field.IsExported()
	}
	ft := field.Type
	if ft.Kind() == reflect.Pointer {
		ft = ft.Elem()
	}
	return !field.IsExported() && ft.Kind() != reflect.Struct
}

// embeddedStruct returns the struct whose fields field promotes, if field is
// an untagged embedded struct or pointer to one.
func embeddedStruct(field reflect.StructField, name string) (reflect.Type, bool) {
	if !field.Anonymous || name != "" {
		return nil, false
	}
	ft := field.Type
	if ft.Kind() == reflect.Pointer {
		ft = ft.Elem()
	}
	if ft.Kind() != reflect.Struct {
		return nil, false
	}
	return ft, true
}

func collectEmbedded(t reflect.Type, depth int, required bool, visiting map[reflect.Type]bool, out *[]schemaField) {
	if visiting[t] {
		return
	}
	visiting[t] = true
	collectFields(t, depth, required, visiting, out)
	delete(visiting, t)
}

func fieldSchema(field reflect.StructField, opts []string, visiting map[reflect.Type]bool) map[string]any {
	property := schemaOf(field.Type, visiting)
	if slices.Contains(opts, "string") && isStringEncodable(field.Type) {
		property = map[string]any{"type": "string"}
	}
	if description := field.Tag.Get("description"); description != "" {
		// A duration's note about its format stays after the field's own text.
		if note, _ := property["description"].(string); note == durationHint {
			description += ". " + note
		}
		property["description"] = description
	}
	return property
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

// Compile compiles a tool input schema for DecodeArgs.
func Compile(schema map[string]any) (*sjs.Schema, error) {
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
	switch v := value.(type) {
	case map[string]any:
		return normalizeObject(v, argType(t, path), path)
	case []any:
		return normalizeArray(v, argType(t, path), path)
	}
	return value, nil
}

// argType is the Go type the value at path decodes into, or nil when any
// JSON is accepted there.
func argType(t reflect.Type, path string) reflect.Type {
	for t != nil && t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t != nil && path != "" && decodesItself(t) {
		// A nested type with its own decoder may read any keys; the
		// top-level input's own UnmarshalJSON is checked by its fields.
		return nil
	}
	return t
}

func normalizeObject(v map[string]any, t reflect.Type, path string) (any, error) {
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
		if !declared {
			if err := checkKnownKey(fields, key, path); err != nil {
				return nil, err
			}
			sub = elem
		}
		if item == nil {
			delete(v, key)
			continue
		}
		normalized, err := normalizeArgs(item, sub, path+key+".")
		if err != nil {
			return nil, err
		}
		v[key] = normalized
	}
	return v, nil
}

// checkKnownKey rejects a key that names no field of a struct, pointing out
// the field it matches when case is ignored. A nil fields map accepts any key.
func checkKnownKey(fields map[string]reflect.Type, key, path string) error {
	if fields == nil {
		return nil
	}
	for name := range fields {
		if strings.EqualFold(name, key) {
			return fmt.Errorf("argument %q must be spelled %q", path+key, path+name)
		}
	}
	return fmt.Errorf("unknown argument %q", path+key)
}

func normalizeArray(v []any, t reflect.Type, path string) (any, error) {
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
	return v, nil
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
