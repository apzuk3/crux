package crux

import (
	"encoding/json"
	"testing"

	"github.com/invopop/jsonschema"
)

func TestAnthropicSchemaMovesConstraintsEverywhere(t *testing.T) {
	raw := `{
		"type": "object",
		"minProperties": 1,
		"properties": {
			"name": {"type": "string", "minLength": 2, "maxLength": 5, "pattern": "^[a-z]+$"},
			"tags": {"type": "array", "minItems": 2, "maxItems": 4, "items": {"type": "string", "maxLength": 9}},
			"one": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/Item"}},
			"choice": {"anyOf": [{"type": "integer", "minimum": 3, "multipleOf": 3}, {"type": "null"}]}
		},
		"$defs": {
			"Item": {"type": "object", "description": "An item", "properties": {"n": {"type": "number", "exclusiveMaximum": 10}}}
		}
	}`
	var schema jsonschema.Schema
	if err := json.Unmarshal([]byte(raw), &schema); err != nil {
		t.Fatal(err)
	}
	wire, err := wireSchemaFor(&schema, ProviderAnthropic)
	if err != nil {
		t.Fatal(err)
	}
	props := wire["properties"].(map[string]any)
	name := props["name"].(map[string]any)
	tags := props["tags"].(map[string]any)
	one := props["one"].(map[string]any)
	choice := props["choice"].(map[string]any)["anyOf"].([]any)[0].(map[string]any)
	item := wire["$defs"].(map[string]any)["Item"].(map[string]any)
	n := item["properties"].(map[string]any)["n"].(map[string]any)

	checks := []struct {
		node        map[string]any
		gone        []string
		description string
	}{
		{wire, []string{"minProperties"}, "minProperties=1"},
		{name, []string{"minLength", "maxLength", "pattern"}, "minLength=2, maxLength=5, pattern=^[a-z]+$"},
		{tags, []string{"minItems", "maxItems"}, "maxItems=4, minItems=2"},
		{tags["items"].(map[string]any), []string{"maxLength"}, "maxLength=9"},
		{choice, []string{"minimum", "multipleOf"}, "minimum=3, multipleOf=3"},
		{n, []string{"exclusiveMaximum"}, "exclusiveMaximum=10"},
	}
	for _, check := range checks {
		for _, keyword := range check.gone {
			if _, ok := check.node[keyword]; ok {
				t.Errorf("%s kept in %v", keyword, check.node)
			}
		}
		if got := check.node["description"]; got != check.description {
			t.Errorf("description = %q, want %q", got, check.description)
		}
	}
	if one["minItems"] != json.Number("1") {
		t.Errorf("minItems 1 should be kept, got %v", one)
	}
	if item["description"] != "An item" || item["additionalProperties"] != false {
		t.Errorf("unexpected $defs item %v", item)
	}
}
