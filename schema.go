package crux

import (
	"encoding/json"
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
	return dec.Decode(target)
}

// wireSchemaFor adapts a jsonschema.Schema for a specific provider's wire format.
func wireSchemaFor(schema *jsonschema.Schema, provider Provider) (map[string]any, error) {
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

	switch provider {
	case ProviderOpenAI, ProviderXAI, ProviderDeepSeek, ProviderOpenrouter, ProviderOllama:
		return adaptOpenAI(m, provider)
	case ProviderAnthropic:
		return adaptAnthropic(m)
	case ProviderGoogle:
		return adaptPermissive(m)
	default:
		return m, nil
	}
}

func cleanBaseSchema(m map[string]any) {
	delete(m, "$schema")
	delete(m, "$id")
}

// adaptOpenAI adapts schemas for OpenAI and OpenAI-compatible endpoints using strict structured outputs.
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
func adaptOpenAI(root map[string]any, provider Provider) (map[string]any, error) {
	err := walkSchemas(root, func(m map[string]any) error {
		isObject := m["type"] == "object" || m["properties"] != nil
		if !isObject {
			return nil
		}

		if additional, exists := m["additionalProperties"]; exists && additional != false {
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

// adaptAnthropic adapts schemas for Anthropic Claude structured outputs.
// Official documentation:
//   - Anthropic Structured Outputs: https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs
//
// Requirements:
//   - "additionalProperties": false on all objects.
//   - Dynamic map/dictionary schemas are not supported.
//   - Numeric validation constraints ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")
//     are not supported as schema keywords and must be transformed into descriptions.
func adaptAnthropic(root map[string]any) (map[string]any, error) {
	err := walkSchemas(root, func(m map[string]any) error {
		isObject := m["type"] == "object" || m["properties"] != nil
		if !isObject {
			return nil
		}

		if additional, exists := m["additionalProperties"]; exists && additional != false {
			return fmt.Errorf("anthropic does not support dynamic map schemas in structured outputs")
		}
		m["additionalProperties"] = false

		props, ok := m["properties"].(map[string]any)
		if !ok {
			return nil
		}

		for _, propVal := range props {
			prop, ok := propVal.(map[string]any)
			if !ok {
				continue
			}
			for _, constraint := range []string{"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"} {
				val, exists := prop[constraint]
				if !exists {
					continue
				}
				delete(prop, constraint)
				descStr := fmt.Sprintf("%s=%v", constraint, val)
				if d, ok := prop["description"].(string); ok && strings.TrimSpace(d) != "" {
					prop["description"] = strings.TrimSpace(d) + ", " + descStr
				} else {
					prop["description"] = descStr
				}
			}
		}
		return nil
	})
	return root, err
}

// adaptPermissive adapts schemas for providers supporting dynamic dictionaries (Google Gemini).
// Official documentation:
//   - Gemini Structured Outputs: https://ai.google.dev/gemini-api/docs/structured-output
//
// Requirements:
//   - Supports dynamic map schemas via "additionalProperties".
//   - For objects without specified additionalProperties, locks them with false.
func adaptPermissive(root map[string]any) (map[string]any, error) {
	err := walkSchemas(root, func(m map[string]any) error {
		isObject := m["type"] == "object" || m["properties"] != nil
		if !isObject {
			return nil
		}
		// If additionalProperties is not set as a dynamic schema, lock it to false.
		if _, exists := m["additionalProperties"]; !exists {
			m["additionalProperties"] = false
		}
		return nil
	})
	return root, err
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

	if props, ok := m["properties"].(map[string]any); ok {
		for _, v := range props {
			if err := walkSchemas(v, visit); err != nil {
				return err
			}
		}
	}
	if items, ok := m["items"]; ok {
		if err := walkSchemas(items, visit); err != nil {
			return err
		}
	}
	if defs, ok := m["$defs"].(map[string]any); ok {
		for _, v := range defs {
			if err := walkSchemas(v, visit); err != nil {
				return err
			}
		}
	}
	if defs, ok := m["definitions"].(map[string]any); ok {
		for _, v := range defs {
			if err := walkSchemas(v, visit); err != nil {
				return err
			}
		}
	}
	for _, key := range []string{"anyOf", "allOf", "oneOf"} {
		if list, ok := m[key].([]any); ok {
			for _, item := range list {
				if err := walkSchemas(item, visit); err != nil {
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

// compileValidator compiles a JSON Schema validator using Draft 2020-12.
// Optional properties also accept null, because that is how strict providers
// are told to leave them out (see adaptOpenAI).
func compileValidator(schema *jsonschema.Schema) (*sjs.Schema, error) {
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

// validateOutput validates the output string against the pre-compiled validator.
func validateOutput(validator *sjs.Schema, text string) error {
	if validator == nil {
		return nil
	}

	clean := strings.TrimSpace(text)
	var doc any
	if err := decodeJSONNumber(strings.NewReader(clean), &doc); err == nil {
		if err := validator.Validate(doc); err != nil {
			return fmt.Errorf("%w: %v", ErrOutputValidation, err)
		}
		return nil
	}

	// Try extracting from markdown fence ```json ... ```
	if start := strings.Index(clean, "```"); start != -1 {
		rest := clean[start+3:]
		if end := strings.LastIndex(rest, "```"); end != -1 {
			block := rest[:end]
			if nl := strings.Index(block, "\n"); nl != -1 {
				block = block[nl+1:]
			} else if strings.HasPrefix(strings.ToLower(block), "json") {
				block = block[4:]
			}
			block = strings.TrimSpace(block)
			if err := decodeJSONNumber(strings.NewReader(block), &doc); err == nil {
				if err := validator.Validate(doc); err != nil {
					return fmt.Errorf("%w: %v", ErrOutputValidation, err)
				}
				return nil
			}
		}
	}

	return fmt.Errorf("%w: invalid json: output could not be parsed as JSON", ErrOutputValidation)
}
