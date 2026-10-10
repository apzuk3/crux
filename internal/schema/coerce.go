package schema

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Duration fields are described as strings such as "90s", so what the
// model sends is converted to nanoseconds before encoding/json sees it.

var coercionCache sync.Map // reflect.Type -> bool

// needsCoercion reports whether a JSON value for t may hold a duration
// string somewhere, following the Go type like normalizeArgs does.
func needsCoercion(t reflect.Type) bool {
	if cached, ok := coercionCache.Load(t); ok {
		return cached.(bool)
	}
	needs := holdsDuration(t, map[reflect.Type]bool{}, true)
	coercionCache.Store(t, needs)
	return needs
}

func holdsDuration(t reflect.Type, visiting map[reflect.Type]bool, top bool) bool {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t == durationType {
		return true
	}
	// A nested type with its own decoder reads what it wants; the top-level
	// input's own UnmarshalJSON is still described by its fields.
	if !top && decodesItself(t) {
		return false
	}
	switch t.Kind() {
	case reflect.Slice, reflect.Array, reflect.Map:
		return holdsDuration(t.Elem(), visiting, false)
	case reflect.Struct:
		if visiting[t] {
			return false
		}
		visiting[t] = true
		defer delete(visiting, t)
		for _, ft := range jsonFields(t) {
			if holdsDuration(ft, visiting, false) {
				return true
			}
		}
	}
	return false
}

// coerce returns value with the duration strings under t replaced by their
// nanosecond counts, as json.Numbers. path locates errors as normalizeArgs
// does.
func coerce(value any, t reflect.Type, path string) (any, error) {
	for t != nil && t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t == nil || value == nil {
		return value, nil
	}
	if t == durationType {
		s, ok := value.(string)
		if !ok {
			return value, nil
		}
		d, err := time.ParseDuration(s)
		if err != nil {
			return nil, fmt.Errorf("%s: invalid duration %q, use a value like \"90s\" or \"1h30m\"", location(path), s)
		}
		return json.Number(strconv.FormatInt(d.Nanoseconds(), 10)), nil
	}
	if path != "" && decodesItself(t) {
		return value, nil
	}
	switch v := value.(type) {
	case map[string]any:
		var fields map[string]reflect.Type
		var elem reflect.Type
		switch t.Kind() {
		case reflect.Struct:
			fields = jsonFields(t)
		case reflect.Map:
			elem = t.Elem()
		default:
			return value, nil
		}
		for key, item := range v {
			sub, ok := fields[key]
			if !ok {
				sub = elem
			}
			coerced, err := coerce(item, sub, path+key+".")
			if err != nil {
				return nil, err
			}
			v[key] = coerced
		}
	case []any:
		if t.Kind() != reflect.Slice && t.Kind() != reflect.Array {
			return value, nil
		}
		for i, item := range v {
			coerced, err := coerce(item, t.Elem(), fmt.Sprintf("%s%d.", path, i))
			if err != nil {
				return nil, err
			}
			v[i] = coerced
		}
	}
	return value, nil
}

func location(path string) string {
	if path == "" {
		return "value"
	}
	return strings.TrimSuffix(path, ".")
}

// Unmarshal is json.Unmarshal into target with the duration strings the
// target's type holds converted first; other targets decode as they are.
func Unmarshal(data []byte, target any) error {
	t := reflect.TypeOf(target)
	if t == nil || t.Kind() != reflect.Pointer || !needsCoercion(t.Elem()) {
		return json.Unmarshal(data, target)
	}
	dec := json.NewDecoder(strings.NewReader(string(data)))
	dec.UseNumber()
	var doc any
	if err := dec.Decode(&doc); err != nil {
		return err
	}
	doc, err := coerce(doc, t.Elem(), "")
	if err != nil {
		return err
	}
	data, err = json.Marshal(doc)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, target)
}
