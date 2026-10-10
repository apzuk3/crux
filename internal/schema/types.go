package schema

import (
	"encoding"
	"encoding/json"
	"math/big"
	"reflect"
	"time"

	"github.com/invopop/jsonschema"
)

var (
	timeType            = reflect.TypeFor[time.Time]()
	durationType        = reflect.TypeFor[time.Duration]()
	bigIntType          = reflect.TypeFor[big.Int]()
	jsonUnmarshalerType = reflect.TypeFor[json.Unmarshaler]()
	textUnmarshalerType = reflect.TypeFor[encoding.TextUnmarshaler]()
	customSchemaType    = reflect.TypeFor[customSchemaer]()
)

// customSchemaer is implemented by a type that describes itself, as
// invopop/jsonschema honours it: the method needs a value receiver.
type customSchemaer interface{ JSONSchema() *jsonschema.Schema }

// durationHint tells the model what a time.Duration field takes.
const durationHint = `Duration such as "90s" or "1h30m"`

// typeRule describes a type by what its decoder accepts rather than by its
// kind. The zero rule means any JSON value.
type typeRule struct{ Type, Format, Description string }

// ruleFor returns the rule for the non-pointer type t, and false when t is
// described by its kind. The same rules serve tool inputs and output
// schemas, because both describe what json.Unmarshal into t accepts:
// time.Time is an RFC 3339 string, time.Duration a string ParseDuration
// reads, big.Int an integer (its UnmarshalJSON takes a number), a type with
// UnmarshalText a string, and a type with only UnmarshalJSON any JSON.
func ruleFor(t reflect.Type) (typeRule, bool) {
	switch {
	case t == timeType:
		return typeRule{Type: "string", Format: "date-time"}, true
	case t == durationType:
		return typeRule{Type: "string", Description: durationHint}, true
	case t == bigIntType:
		return typeRule{Type: "integer"}, true
	case reflect.PointerTo(t).Implements(textUnmarshalerType):
		return typeRule{Type: "string"}, true
	case reflect.PointerTo(t).Implements(jsonUnmarshalerType):
		return typeRule{}, true
	}
	return typeRule{}, false
}

// asMap renders the rule for a tool input schema, as a fresh map.
func (r typeRule) asMap() map[string]any {
	m := map[string]any{}
	if r.Type != "" {
		m["type"] = r.Type
	}
	if r.Format != "" {
		m["format"] = r.Format
	}
	if r.Description != "" {
		m["description"] = r.Description
	}
	return m
}

// asSchema renders the rule for an output schema; the zero rule is an empty
// schema, which marshals as true.
func (r typeRule) asSchema() *jsonschema.Schema {
	return &jsonschema.Schema{Type: r.Type, Format: r.Format, Description: r.Description}
}

// customSchema returns the schema a type declares with a JSONSchema method.
func customSchema(t reflect.Type) (map[string]any, bool) {
	if t.Kind() == reflect.Interface || !t.Implements(customSchemaType) {
		return nil, false
	}
	raw, err := json.Marshal(reflect.Zero(t).Interface().(customSchemaer).JSONSchema())
	if err != nil {
		return nil, false
	}
	var m map[string]any
	if json.Unmarshal(raw, &m) != nil {
		return map[string]any{}, true // a boolean schema: any JSON
	}
	cleanBaseSchema(m)
	return m, true
}

// ReflectOutput reflects an output schema from t with invopop/jsonschema,
// applying the type rules where invopop would expand a type by its fields
// (a decimal would become an empty object, a UUID an array of 16 integers).
// A type with its own JSONSchema method keeps invopop's handling of it.
func ReflectOutput(t reflect.Type) *jsonschema.Schema {
	// invopop resets a property's description from its jsonschema_description
	// tag after the mapper ran, so the rules' notes are put back afterwards,
	// where the field has no description of its own.
	noted := map[*jsonschema.Schema]string{}
	reflector := &jsonschema.Reflector{
		Anonymous:      true,
		ExpandedStruct: true,
		Mapper: func(t reflect.Type) *jsonschema.Schema {
			if t.Implements(customSchemaType) {
				return nil
			}
			r, ok := ruleFor(t)
			if !ok {
				return nil
			}
			s := r.asSchema()
			if r.Description != "" {
				noted[s] = r.Description
			}
			return s
		},
	}
	schema := reflector.ReflectFromType(t)
	for s, note := range noted {
		if s.Description == "" {
			s.Description = note
		}
	}
	return schema
}
