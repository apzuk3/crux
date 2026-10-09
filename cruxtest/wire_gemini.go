package cruxtest

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
)

func buildGeminiResponse(turn *Turn, callIndex int) ([]byte, error) {
	if body, done, err := rawOrErrorBody(turn, geminiError); done {
		return body, err
	}

	parts, finishReason, err := geminiParts(turn, callIndex)
	if err != nil {
		return nil, err
	}

	resp := map[string]any{
		"responseId": fmt.Sprintf("gemini_resp_%d", callIndex),
		"candidates": []any{
			map[string]any{
				"content": map[string]any{
					"role":  "model",
					"parts": parts,
				},
				"finishReason": finishReason,
			},
		},
		"usageMetadata": geminiUsage(turn.Usage),
	}

	return json.Marshal(resp)
}

func geminiError(status int) any {
	return map[string]any{
		"error": map[string]any{
			"code":    status,
			"message": fmt.Sprintf("HTTP %d error", status),
			"status":  "UNAVAILABLE",
		},
	}
}

// geminiParts returns the parts of a turn and its finish reason.
func geminiParts(turn *Turn, callIndex int) ([]any, string, error) {
	switch {
	case turn.empty:
		return []any{}, "STOP", nil
	case turn.Refusal != "":
		return []any{map[string]any{"text": turn.Refusal}}, "SAFETY", nil
	case len(turn.ToolCalls) > 0:
		parts, err := geminiFunctionCallParts(turn, callIndex)
		return parts, "STOP", err
	default:
		part := withThoughtSignature(map[string]any{"text": turn.Text}, turn.thoughtSignature)
		return []any{part}, "STOP", nil
	}
}

// geminiFunctionCallParts returns the functionCall parts of a turn, preceded
// by its text part when it has one.
func geminiFunctionCallParts(turn *Turn, callIndex int) ([]any, error) {
	parts := []any{}
	if turn.Text != "" {
		parts = append(parts, map[string]any{"text": turn.Text})
	}
	for i, tc := range turn.ToolCalls {
		part, err := geminiFunctionCallPart(turn, tc, callIndex, i)
		if err != nil {
			return nil, err
		}
		parts = append(parts, part)
	}
	return parts, nil
}

func geminiFunctionCallPart(turn *Turn, tc ToolCall, callIndex, i int) (map[string]any, error) {
	argsMap, _, err := normalizeToolArgs(tc.Args)
	if err != nil {
		return nil, err
	}
	call := map[string]any{"name": tc.Name, "args": argsMap}
	if !turn.omitCallIDs {
		toolID := tc.ID
		if toolID == "" {
			toolID = fmt.Sprintf("call_%d_%d", callIndex, i+1)
		}
		call["id"] = toolID
	}
	part := map[string]any{"functionCall": call}
	if i == 0 {
		return withThoughtSignature(part, turn.thoughtSignature), nil
	}
	return part, nil
}

func withThoughtSignature(part map[string]any, signature string) map[string]any {
	if signature != "" {
		part["thoughtSignature"] = base64.StdEncoding.EncodeToString([]byte(signature))
	}
	return part
}

func geminiUsage(usage *TokenUsage) map[string]any {
	promptTokens := 10
	candidatesTokens := 5
	cachedTokens := 0
	if usage != nil {
		promptTokens = usage.InputTokens
		candidatesTokens = usage.OutputTokens
		cachedTokens = usage.CacheReadTokens
	}
	return map[string]any{
		"promptTokenCount":        promptTokens,
		"candidatesTokenCount":    candidatesTokens,
		"cachedContentTokenCount": cachedTokens,
	}
}
