package schema

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"
)

type retryPolicy struct {
	Timeout time.Duration `json:"timeout"`
}

type opaqueWait struct{ d time.Duration }

func (o *opaqueWait) UnmarshalJSON(data []byte) error { return nil }

type coerceInput struct {
	For     time.Duration            `json:"for"`
	Grace   *time.Duration           `json:"grace,omitempty"`
	All     []time.Duration          `json:"all"`
	ByName  map[string]time.Duration `json:"by_name"`
	Items   []retryPolicy            `json:"items"`
	Opaque  opaqueWait               `json:"opaque"`
	Comment string                   `json:"comment"`
}

func TestCoerce(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		want    string
		wantErr string
	}{
		{"field", `{"for":"90s"}`, `{"for":90000000000}`, ""},
		{"pointer and slice", `{"grace":"1m","all":["1s","2ms"]}`, `{"all":[1000000000,2000000],"grace":60000000000}`, ""},
		{"map values", `{"by_name":{"a":"1h"}}`, `{"by_name":{"a":3600000000000}}`, ""},
		{"nested struct", `{"items":[{"timeout":"5s"}]}`, `{"items":[{"timeout":5000000000}]}`, ""},
		{"number passes through", `{"for":42}`, `{"for":42}`, ""},
		{"opaque type untouched", `{"opaque":"never"}`, `{"opaque":"never"}`, ""},
		{"other strings untouched", `{"comment":"90s"}`, `{"comment":"90s"}`, ""},
		{"invalid", `{"items":[{"timeout":"5 seconds"}]}`, "", `items.0.timeout: invalid duration "5 seconds", use a value like "90s" or "1h30m"`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dec := json.NewDecoder(strings.NewReader(tc.in))
			dec.UseNumber()
			var doc any
			if err := dec.Decode(&doc); err != nil {
				t.Fatal(err)
			}
			got, err := coerce(doc, reflect.TypeFor[coerceInput](), "")
			if tc.wantErr != "" {
				if err == nil || err.Error() != tc.wantErr {
					t.Fatalf("err = %v, want %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			raw, _ := json.Marshal(got)
			if string(raw) != tc.want {
				t.Fatalf("got %s, want %s", raw, tc.want)
			}
		})
	}
}

func TestNeedsCoercion(t *testing.T) {
	if needsCoercion(reflect.TypeFor[normalizeInput]()) {
		t.Fatal("a type without durations needs no coercion")
	}
	for _, typ := range []reflect.Type{reflect.TypeFor[coerceInput](), reflect.TypeFor[*retryPolicy](), reflect.TypeFor[[]retryPolicy](), reflect.TypeFor[time.Duration]()} {
		if !needsCoercion(typ) {
			t.Fatalf("%v holds a duration", typ)
		}
	}
	if needsCoercion(reflect.TypeFor[opaqueWait]()) {
		t.Fatal("a type with its own decoder is not looked into")
	}
}

func TestDecodeArgsDuration(t *testing.T) {
	validator, err := Compile(Of[retryPolicy]())
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeArgs[retryPolicy]("wait", json.RawMessage(`{"timeout":"90s"}`), validator)
	if err != nil || got.Timeout != 90*time.Second {
		t.Fatalf("DecodeArgs = %v, %v", got, err)
	}
	_, err = DecodeArgs[retryPolicy]("wait", json.RawMessage(`{"timeout":90}`), validator)
	if err == nil || !strings.Contains(err.Error(), "string") {
		t.Fatalf("a number should fail validation as the schema says string, got %v", err)
	}
	_, err = DecodeArgs[retryPolicy]("wait", json.RawMessage(`{"timeout":"90 seconds"}`), validator)
	if err == nil || !strings.Contains(err.Error(), `invalid duration "90 seconds"`) {
		t.Fatalf("an unparsable duration should be reported, got %v", err)
	}
}

func TestUnmarshalDurations(t *testing.T) {
	var p retryPolicy
	if err := Unmarshal([]byte(`{"timeout":"1m30s"}`), &p); err != nil || p.Timeout != 90*time.Second {
		t.Fatalf("Unmarshal = %v, %v", p, err)
	}
	var plain struct{ N int }
	if err := Unmarshal([]byte(`{"N":1}`), &plain); err != nil || plain.N != 1 {
		t.Fatalf("plain Unmarshal = %v, %v", plain, err)
	}
}
