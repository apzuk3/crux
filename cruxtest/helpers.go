package cruxtest

import (
	"encoding/json"
	"fmt"
)

// normalizeToolArgs converts any supported tool args representation into:
// 1. A JSON object (map[string]any) for Anthropic and Gemini
// 2. A JSON string (string) for OpenAI
func normalizeToolArgs(args any) (map[string]any, string, error) {
	if args == nil {
		return map[string]any{}, "{}", nil
	}

	switch v := args.(type) {
	case string:
		if v == "" {
			return map[string]any{}, "{}", nil
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(v), &m); err != nil {
			// If not a JSON object, still allow the string for OpenAI
			return map[string]any{"raw": v}, v, nil
		}
		return m, v, nil
	case []byte:
		if len(v) == 0 {
			return map[string]any{}, "{}", nil
		}
		var m map[string]any
		if err := json.Unmarshal(v, &m); err != nil {
			return map[string]any{"raw": string(v)}, string(v), nil
		}
		return m, string(v), nil
	case map[string]any:
		b, err := json.Marshal(v)
		if err != nil {
			return nil, "", fmt.Errorf("marshal tool args map: %w", err)
		}
		return v, string(b), nil
	default:
		// Struct or any other JSON-serializable type
		b, err := json.Marshal(v)
		if err != nil {
			return nil, "", fmt.Errorf("marshal tool args: %w", err)
		}
		var m map[string]any
		if err := json.Unmarshal(b, &m); err != nil {
			return nil, "", fmt.Errorf("unmarshal tool args into map: %w", err)
		}
		return m, string(b), nil
	}
}
