package schema

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/invopop/jsonschema"
)

func decodeSchema(t *testing.T, raw string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := decodeJSONNumber(strings.NewReader(raw), &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestAdaptOpenAIGolden(t *testing.T) {
	tests := []struct {
		name     string
		provider string
		in       string
		want     string
		wantErr  string
	}{
		{
			name:     "optional properties become nullable and required",
			provider: "openai",
			in:       `{"properties":{"a":{"type":"integer"},"b":{"type":"string"},"c":{"anyOf":[{"type":"string"},{"type":"null"}]},"d":{"properties":{"x":{"type":["string","null"]}},"type":"object"},"e":{"items":{"properties":{"y":{"type":"boolean"}}},"type":"array"}},"required":["b"],"type":"object"}`,
			want:     `{"additionalProperties":false,"properties":{"a":{"anyOf":[{"type":"integer"},{"type":"null"}]},"b":{"type":"string"},"c":{"anyOf":[{"type":"string"},{"type":"null"}]},"d":{"anyOf":[{"additionalProperties":false,"properties":{"x":{"type":["string","null"]}},"required":["x"],"type":"object"},{"type":"null"}]},"e":{"anyOf":[{"items":{"additionalProperties":false,"properties":{"y":{"anyOf":[{"type":"boolean"},{"type":"null"}]}},"required":["y"]},"type":"array"},{"type":"null"}]}},"required":["b","a","c","d","e"],"type":"object"}`,
		},
		{
			name:     "empty object",
			provider: "openai",
			in:       `{"$schema":"x","properties":{},"type":"object"}`,
			want:     `{"additionalProperties":false,"properties":{},"type":"object"}`,
		},
		{
			name:     "non-object root",
			provider: "xai",
			in:       `{"type":"array","items":{"type":"string"}}`,
			wantErr:  "xai structured outputs require an object at the root of the output schema, got type array; wrap the value in a struct field",
		},
		{
			name:     "ollama accepts any root",
			provider: "ollama",
			in:       `{"type":"array","items":{"type":"string"}}`,
			want:     `{"items":{"type":"string"},"type":"array"}`,
		},
		{
			name:     "dynamic map",
			provider: "openai",
			in:       `{"type":"object","properties":{"m":{"type":"object","additionalProperties":{"type":"string"}}}}`,
			wantErr:  "openai does not support dynamic map schemas in structured outputs",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := AdaptOpenAI(decodeSchema(t, tc.in), tc.provider)
			if tc.wantErr != "" {
				if err == nil || err.Error() != tc.wantErr {
					t.Fatalf("err = %v, want %s", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			raw, _ := json.Marshal(got)
			if string(raw) != tc.want {
				t.Fatalf("adapted\n got %s\nwant %s", raw, tc.want)
			}
		})
	}
}

func TestAllowsNull(t *testing.T) {
	tests := map[string]bool{
		`{"type":"null"}`:                                           true,
		`{"type":"string"}`:                                         false,
		`{"type":["string","null"]}`:                                true,
		`{"type":["string","integer"]}`:                             false,
		`{"type":[1]}`:                                              false,
		`{"anyOf":[{"type":"string"},{"type":"null"}]}`:             true,
		`{"oneOf":[{"type":"string"},{"type":["null"]}]}`:           true,
		`{"anyOf":[{"type":"string"},{"oneOf":[{"type":"null"}]}]}`: true,
		`{"anyOf":[{"type":"string"},"null"]}`:                      false,
		`{"allOf":[{"type":"null"}]}`:                               false,
		`{}`:                                                        false,
	}
	for raw, want := range tests {
		if got := allowsNull(decodeSchema(t, raw)); got != want {
			t.Errorf("allowsNull(%s) = %v, want %v", raw, got, want)
		}
	}
}

func TestCompileOutput(t *testing.T) {
	validator, err := CompileOutput(nil)
	if err != nil || validator != nil {
		t.Fatalf("nil schema: %v, %v", validator, err)
	}

	schema := &jsonschema.Schema{Type: "object", Required: []string{"name"}}
	schema.Properties = jsonschema.NewProperties()
	schema.Properties.Set("name", &jsonschema.Schema{Type: "string"})
	schema.Properties.Set("note", &jsonschema.Schema{Type: "string"})
	schema.Properties.Set("flag", &jsonschema.Schema{AnyOf: []*jsonschema.Schema{{Type: "boolean"}, {Type: "null"}}})
	nested := &jsonschema.Schema{Type: "object", Required: []string{"id"}}
	nested.Properties = jsonschema.NewProperties()
	nested.Properties.Set("id", &jsonschema.Schema{Type: "integer"})
	nested.Properties.Set("tag", &jsonschema.Schema{Type: "string"})
	schema.Properties.Set("inner", nested)
	validator, err = CompileOutput(schema)
	if err != nil {
		t.Fatal(err)
	}

	tests := map[string]bool{
		`{"name":"a"}`: true,
		`{"name":"a","note":null,"flag":null,"inner":null}`: true,
		`{"name":"a","inner":{"id":1,"tag":null}}`:          true,
		`{"name":null}`:                    false,
		`{"name":"a","note":1}`:            false,
		`{"name":"a","inner":{"id":null}}`: false,
		`{"name":"a","inner":{"tag":"t"}}`: false,
	}
	for raw, want := range tests {
		err := ValidateOutput(validator, raw)
		if (err == nil) != want {
			t.Errorf("validate %s: err = %v, want ok=%v", raw, err, want)
		}
	}
}
