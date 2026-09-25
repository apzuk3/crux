package crux

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/invopop/jsonschema"
	sjs "github.com/santhosh-tekuri/jsonschema/v5"
)

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
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("decode output schema: %w", err)
	}

	normalizeNode(m, provider)
	return m, nil
}

func normalizeNode(node any, provider Provider) {
	m, ok := node.(map[string]any)
	if !ok {
		if s, ok := node.([]any); ok {
			for _, item := range s {
				normalizeNode(item, provider)
			}
		}
		return
	}

	delete(m, "$schema")
	delete(m, "$id")

	isObject := m["type"] == "object" || m["properties"] != nil
	if isObject {
		m["additionalProperties"] = false

		if provider == ProviderAnthropic {
			if props, ok := m["properties"].(map[string]any); ok {
				for _, propVal := range props {
					if prop, ok := propVal.(map[string]any); ok {
						for _, constraint := range []string{"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"} {
							if val, exists := prop[constraint]; exists {
								delete(prop, constraint)
								descStr := fmt.Sprintf("%s=%v", constraint, val)
								if d, ok := prop["description"].(string); ok && strings.TrimSpace(d) != "" {
									prop["description"] = strings.TrimSpace(d) + ", " + descStr
								} else {
									prop["description"] = descStr
								}
							}
						}
					}
				}
			}
		} else if provider != ProviderGoogle {
			// OpenAI and OpenAI-compatible providers: all properties must be in required.
			if props, ok := m["properties"].(map[string]any); ok && len(props) > 0 {
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
				for k := range props {
					if !reqSet[k] {
						missing = append(missing, k)
					}
				}
				sort.Strings(missing)
				reqList = append(reqList, missing...)
				m["required"] = reqList
			}
		}
	}

	if props, ok := m["properties"].(map[string]any); ok {
		for _, v := range props {
			normalizeNode(v, provider)
		}
	}
	if items, ok := m["items"]; ok {
		normalizeNode(items, provider)
	}
	if defs, ok := m["$defs"].(map[string]any); ok {
		for _, v := range defs {
			normalizeNode(v, provider)
		}
	}
	if defs, ok := m["definitions"].(map[string]any); ok {
		for _, v := range defs {
			normalizeNode(v, provider)
		}
	}
	for _, key := range []string{"anyOf", "allOf", "oneOf"} {
		if list, ok := m[key].([]any); ok {
			for _, item := range list {
				normalizeNode(item, provider)
			}
		}
	}
}

// compileValidator prepares and compiles a JSON Schema validator using Draft 2020-12.
func compileValidator(schema *jsonschema.Schema) (*sjs.Schema, error) {
	if schema == nil {
		return nil, nil
	}

	raw, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("marshal schema for compilation: %w", err)
	}

	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("decode schema for compilation: %w", err)
	}

	prepareValidationSchema(m)

	processed, err := json.Marshal(m)
	if err != nil {
		return nil, fmt.Errorf("marshal prepared schema: %w", err)
	}

	compiler := sjs.NewCompiler()
	compiler.Draft = sjs.Draft2020
	if err := compiler.AddResource("output.json", strings.NewReader(string(processed))); err != nil {
		return nil, fmt.Errorf("add schema resource: %w", err)
	}

	return compiler.Compile("output.json")
}

// prepareValidationSchema ensures optional fields (not in required) permit null.
func prepareValidationSchema(node any) {
	m, ok := node.(map[string]any)
	if !ok {
		if s, ok := node.([]any); ok {
			for _, item := range s {
				prepareValidationSchema(item)
			}
		}
		return
	}

	if props, ok := m["properties"].(map[string]any); ok {
		reqSet := make(map[string]bool)
		if req, ok := m["required"].([]any); ok {
			for _, item := range req {
				if s, ok := item.(string); ok {
					reqSet[s] = true
				}
			}
		}

		for propName, propVal := range props {
			if prop, ok := propVal.(map[string]any); ok {
				if !reqSet[propName] {
					// Optional property: allow null in addition to its declared type.
					if typ, ok := prop["type"].(string); ok && typ != "null" {
						prop["type"] = []any{typ, "null"}
					} else if types, ok := prop["type"].([]any); ok {
						hasNull := false
						for _, t := range types {
							if t == "null" {
								hasNull = true
								break
							}
						}
						if !hasNull {
							prop["type"] = append(types, "null")
						}
					}
				}
				prepareValidationSchema(prop)
			}
		}
	}

	if items, ok := m["items"]; ok {
		prepareValidationSchema(items)
	}
	if defs, ok := m["$defs"].(map[string]any); ok {
		for _, v := range defs {
			prepareValidationSchema(v)
		}
	}
	if defs, ok := m["definitions"].(map[string]any); ok {
		for _, v := range defs {
			prepareValidationSchema(v)
		}
	}
	for _, key := range []string{"anyOf", "allOf", "oneOf"} {
		if list, ok := m[key].([]any); ok {
			for _, item := range list {
				prepareValidationSchema(item)
			}
		}
	}
}

// validateOutput validates the output string against the provided schema.
func validateOutput(schema *jsonschema.Schema, text string) error {
	if schema == nil {
		return nil
	}

	validator, err := compileValidator(schema)
	if err != nil {
		return fmt.Errorf("compile output schema validator: %w", err)
	}

	clean := strings.TrimSpace(text)
	var doc any
	if err := json.Unmarshal([]byte(clean), &doc); err == nil {
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
			if err := json.Unmarshal([]byte(block), &doc); err == nil {
				if err := validator.Validate(doc); err != nil {
					return fmt.Errorf("%w: %v", ErrOutputValidation, err)
				}
				return nil
			}
		}
	}

	return fmt.Errorf("%w: invalid json: output could not be parsed as JSON", ErrOutputValidation)
}
