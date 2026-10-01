package schema

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/invopop/jsonschema"
	sjs "github.com/santhosh-tekuri/jsonschema/v5"
)

func decodeJSONNumber(r io.Reader, target any) error {
	dec := json.NewDecoder(r)
	dec.UseNumber()
	if err := dec.Decode(target); err != nil {
		return err
	}
	// Trailing data would make validation accept output that decoding rejects.
	if _, err := dec.Token(); err != io.EOF {
		return errors.New("unexpected data after the JSON value")
	}
	return nil
}

// Wire turns an output schema into the map the provider adapters work on.
// A nil schema gives nil.
func Wire(schema *jsonschema.Schema) (map[string]any, error) {
	if schema == nil {
		return nil, nil
	}
	raw, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("marshal output schema: %w", err)
	}
	var m map[string]any
	if err := decodeJSONNumber(strings.NewReader(string(raw)), &m); err != nil {
		return nil, fmt.Errorf("decode output schema: %w", err)
	}
	cleanBaseSchema(m)
	return m, nil
}

func cleanBaseSchema(m map[string]any) {
	delete(m, "$schema")
	delete(m, "$id")
}

// AdaptOpenAI adapts schemas for OpenAI and OpenAI-compatible endpoints using strict structured outputs.
// Official documentation:
//   - OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
//   - xAI API Reference: https://docs.x.ai/api
//   - DeepSeek API (OpenAI compatible): https://api-docs.deepseek.com/guides/json_mode
//   - OpenRouter Structured Outputs: https://openrouter.ai/docs/structured-outputs
//   - Ollama Structured Outputs: https://ollama.com/blog/structured-outputs
//
// Requirements:
//   - All properties in an object must be listed in "required".
//   - Optional properties are placed in "required" and marked nullable ("anyOf": [prop, {"type": "null"}]).
//   - "additionalProperties": false on every object.
//   - Dynamic map/dictionary schemas (arbitrary keys) are rejected in strict mode.
//   - The root must be an object (Ollama excepted):
//     https://platform.openai.com/docs/guides/structured-outputs#supported-schemas
func AdaptOpenAI(root map[string]any, provider string) (map[string]any, error) {
	// Ollama does not enforce strict mode and accepts any root.
	if provider != "ollama" {
		if typ, ok := root["type"]; ok && typ != "object" {
			return nil, fmt.Errorf("%s structured outputs require an object at the root of the output schema, got type %v; wrap the value in a struct field", provider, typ)
		}
	}
	err := walkSchemas(root, func(m map[string]any) error {
		isObject := m["type"] == "object" || m["properties"] != nil
		if !isObject {
			return nil
		}

		if isDynamicMap(m) {
			return fmt.Errorf("%s does not support dynamic map schemas in structured outputs", provider)
		}
		m["additionalProperties"] = false

		props, ok := m["properties"].(map[string]any)
		if !ok || len(props) == 0 {
			return nil
		}

		reqSet := make(map[string]bool)
		var reqList []string
		if existing, ok := m["required"].([]any); ok {
			for _, item := range existing {
				if s, ok := item.(string); ok {
					reqSet[s] = true
					reqList = append(reqList, s)
				}
			}
		}

		var missing []string
		for k, propVal := range props {
			if !reqSet[k] {
				missing = append(missing, k)
				if prop, ok := propVal.(map[string]any); ok && !allowsNull(prop) {
					props[k] = map[string]any{
						"anyOf": []any{
							prop,
							map[string]any{"type": "null"},
						},
					}
				}
			}
		}
		sort.Strings(missing)
		reqList = append(reqList, missing...)
		m["required"] = reqList
		return nil
	})
	return root, err
}

// AdaptAnthropic adapts schemas for Anthropic Claude structured outputs.
// Official documentation:
//   - Anthropic Structured Outputs: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
//
// Requirements:
//   - "additionalProperties": false on all objects.
//   - Dynamic map/dictionary schemas are not supported.
//   - Numeric, string length, pattern and most array constraints are not
//     supported as schema keywords ("JSON Schema limitations" in the page
//     above), so they move into the description of the schema that had them.
//     Output validation still enforces them.
func AdaptAnthropic(root map[string]any, _ string) (map[string]any, error) {
	err := walkSchemas(root, func(m map[string]any) error {
		moveToDescription(m, anthropicUnsupportedKeywords)
		if minItems, ok := m["minItems"]; ok && !isZeroOrOne(minItems) {
			moveToDescription(m, []string{"minItems"})
		}

		isObject := m["type"] == "object" || m["properties"] != nil
		if !isObject {
			return nil
		}
		if isDynamicMap(m) {
			return fmt.Errorf("anthropic does not support dynamic map schemas in structured outputs")
		}
		m["additionalProperties"] = false
		return nil
	})
	return root, err
}

// anthropicUnsupportedKeywords are constraints Anthropic structured outputs
// reject. minItems is supported only as 0 or 1 and is handled separately.
var anthropicUnsupportedKeywords = []string{
	"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
	"minLength", "maxLength", "pattern",
	"maxItems", "uniqueItems", "minContains", "maxContains",
	"minProperties", "maxProperties",
}

// moveToDescription removes the given keywords from a schema and records them
// in its description, so the model still sees them.
func moveToDescription(m map[string]any, keywords []string) {
	var notes []string
	for _, keyword := range keywords {
		value, exists := m[keyword]
		if !exists {
			continue
		}
		delete(m, keyword)
		notes = append(notes, fmt.Sprintf("%s=%v", keyword, value))
	}
	if len(notes) == 0 {
		return
	}
	note := strings.Join(notes, ", ")
	if d, ok := m["description"].(string); ok && strings.TrimSpace(d) != "" {
		m["description"] = strings.TrimSpace(d) + ", " + note
	} else {
		m["description"] = note
	}
}

func isZeroOrOne(value any) bool {
	switch v := value.(type) {
	case json.Number:
		return v == "0" || v == "1"
	case float64:
		return v == 0 || v == 1
	case int:
		return v == 0 || v == 1
	}
	return false
}

// AdaptPermissive adapts schemas for providers supporting dynamic dictionaries (Google Gemini).
// Official documentation:
//   - Gemini Structured Outputs: https://ai.google.dev/gemini-api/docs/structured-output
//
// Requirements:
//   - Supports dynamic map schemas via "additionalProperties".
//   - For objects without specified additionalProperties, locks them with false.
func AdaptPermissive(root map[string]any, _ string) (map[string]any, error) {
	err := walkSchemas(root, func(m map[string]any) error {
		isObject := m["type"] == "object" || m["properties"] != nil
		if !isObject {
			return nil
		}
		// Close objects with declared properties; a dynamic map stays open.
		if !isDynamicMap(m) {
			m["additionalProperties"] = false
		}
		return nil
	})
	return root, err
}

// isDynamicMap reports whether an object schema accepts arbitrary keys: it
// allows additional properties, or, like the schema reflected from
// map[string]any, declares neither properties nor additionalProperties.
// Closing such a schema would let the model return only an empty object.
func isDynamicMap(m map[string]any) bool {
	additional, exists := m["additionalProperties"]
	if exists {
		return additional != false
	}
	_, hasProps := m["properties"]
	return !hasProps
}

// walkSchemas recursively visits every JSON schema object in the tree.
func walkSchemas(node any, visit func(m map[string]any) error) error {
	m, ok := node.(map[string]any)
	if !ok {
		if list, ok := node.([]any); ok {
			for _, item := range list {
				if err := walkSchemas(item, visit); err != nil {
					return err
				}
			}
		}
		return nil
	}

	cleanBaseSchema(m)

	if err := visit(m); err != nil {
		return err
	}

	// Keywords whose value is a schema, or a list of schemas.
	for _, key := range []string{
		"items", "additionalItems", "prefixItems", "contains", "unevaluatedItems",
		"additionalProperties", "propertyNames", "unevaluatedProperties",
		"not", "if", "then", "else", "anyOf", "allOf", "oneOf", "contentSchema",
	} {
		if sub, ok := m[key]; ok {
			if err := walkSchemas(sub, visit); err != nil {
				return err
			}
		}
	}
	// Keywords whose value maps names to schemas. The draft-7 "dependencies"
	// may also map to property lists, which hold no schemas to visit.
	for _, key := range []string{"properties", "patternProperties", "dependentSchemas", "dependencies", "$defs", "definitions"} {
		if named, ok := m[key].(map[string]any); ok {
			for _, v := range named {
				if err := walkSchemas(v, visit); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func allowsNull(prop map[string]any) bool {
	if typ, ok := prop["type"].(string); ok && typ == "null" {
		return true
	}
	if types, ok := prop["type"].([]any); ok {
		for _, t := range types {
			if s, ok := t.(string); ok && s == "null" {
				return true
			}
		}
	}
	for _, key := range []string{"anyOf", "oneOf"} {
		if branches, ok := prop[key].([]any); ok {
			for _, b := range branches {
				if bm, ok := b.(map[string]any); ok && allowsNull(bm) {
					return true
				}
			}
		}
	}
	return false
}

// CompileOutput compiles a JSON Schema validator using Draft 2020-12.
// Optional properties also accept null, because that is how strict providers
// are told to leave them out (see AdaptOpenAI).
func CompileOutput(schema *jsonschema.Schema) (*sjs.Schema, error) {
	if schema == nil {
		return nil, nil
	}

	raw, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("marshal schema for compilation: %w", err)
	}
	var m map[string]any
	if err := decodeJSONNumber(strings.NewReader(string(raw)), &m); err != nil {
		return nil, fmt.Errorf("decode schema for compilation: %w", err)
	}
	_ = walkSchemas(m, func(m map[string]any) error {
		props, ok := m["properties"].(map[string]any)
		if !ok {
			return nil
		}
		required := make(map[string]bool)
		if list, ok := m["required"].([]any); ok {
			for _, item := range list {
				if name, ok := item.(string); ok {
					required[name] = true
				}
			}
		}
		for name, value := range props {
			if prop, ok := value.(map[string]any); ok && !required[name] && !allowsNull(prop) {
				props[name] = map[string]any{"anyOf": []any{prop, map[string]any{"type": "null"}}}
			}
		}
		return nil
	})
	if raw, err = json.Marshal(m); err != nil {
		return nil, fmt.Errorf("marshal schema for compilation: %w", err)
	}

	compiler := sjs.NewCompiler()
	compiler.Draft = sjs.Draft2020
	if err := compiler.AddResource("output.json", strings.NewReader(string(raw))); err != nil {
		return nil, fmt.Errorf("add schema resource: %w", err)
	}

	return compiler.Compile("output.json")
}

// ValidateOutput validates the output string against the pre-compiled validator.
// A nil validator accepts anything.
func ValidateOutput(validator *sjs.Schema, text string) error {
	if validator == nil {
		return nil
	}
	for _, candidate := range JSONCandidates(text) {
		var doc any
		if decodeJSONNumber(strings.NewReader(candidate), &doc) != nil {
			continue
		}
		if err := validator.Validate(doc); err != nil {
			return err
		}
		return nil
	}
	return errors.New("invalid json: output could not be parsed as JSON")
}

// JSONCandidates returns the forms a model's JSON answer may take: the whole
// text, trimmed, and the content of a fenced code block (```json ... ```) in
// it, if there is one.
func JSONCandidates(text string) []string {
	clean := strings.TrimSpace(text)
	candidates := []string{clean}
	start := strings.Index(clean, "```")
	if start == -1 {
		return candidates
	}
	rest := clean[start+3:]
	end := strings.LastIndex(rest, "```")
	if end == -1 {
		return candidates
	}
	block := rest[:end]
	if nl := strings.Index(block, "\n"); nl != -1 {
		block = block[nl+1:]
	} else if strings.HasPrefix(strings.ToLower(block), "json") {
		block = block[4:]
	}
	return append(candidates, strings.TrimSpace(block))
}
