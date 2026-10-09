package cruxtest

import (
	"encoding/json"
	"fmt"
)

func buildAnthropicResponse(turn *Turn, callIndex int) ([]byte, error) {
	if body, done, err := rawOrErrorBody(turn, anthropicError); done {
		return body, err
	}

	content, stopReason, err := anthropicContent(turn, callIndex)
	if err != nil {
		return nil, err
	}

	resp := map[string]any{
		"id":            fmt.Sprintf("msg_%d", callIndex),
		"type":          "message",
		"role":          "assistant",
		"content":       content,
		"stop_reason":   stopReason,
		"stop_sequence": nil,
		"usage":         anthropicUsage(turn.Usage),
	}

	return json.Marshal(resp)
}

func anthropicError(status int) any {
	return map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    "api_error",
			"message": fmt.Sprintf("HTTP %d error", status),
		},
	}
}

// anthropicContent returns the content blocks of a turn and its stop reason.
func anthropicContent(turn *Turn, callIndex int) ([]any, string, error) {
	switch {
	case turn.empty:
		return []any{}, "end_turn", nil
	case turn.Refusal != "":
		return []any{anthropicText(turn.Refusal)}, "refusal", nil
	case len(turn.ToolCalls) > 0:
		content, err := anthropicToolUseBlocks(turn, callIndex)
		return content, "tool_use", err
	default:
		return []any{anthropicText(turn.Text)}, "end_turn", nil
	}
}

func anthropicText(text string) map[string]any {
	return map[string]any{
		"type": "text",
		"text": text,
	}
}

// anthropicToolUseBlocks returns the tool_use blocks of a turn, preceded by
// its text block when it has one.
func anthropicToolUseBlocks(turn *Turn, callIndex int) ([]any, error) {
	content := []any{}
	for i, tc := range turn.ToolCalls {
		toolID := tc.ID
		if toolID == "" {
			toolID = fmt.Sprintf("toolu_%d_%d", callIndex, i+1)
		}
		argsMap, _, err := normalizeToolArgs(tc.Args)
		if err != nil {
			return nil, err
		}
		content = append(content, map[string]any{
			"type":  "tool_use",
			"id":    toolID,
			"name":  tc.Name,
			"input": argsMap,
		})
	}
	text := turn.Text
	if text == "" && turn.blankText {
		text = "\n\n"
	}
	if text != "" {
		content = append([]any{anthropicText(text)}, content...)
	}
	return content, nil
}

func anthropicUsage(usage *TokenUsage) map[string]any {
	inputTokens := 10
	outputTokens := 5
	cacheRead := 0
	cacheWrite := 0
	if usage != nil {
		inputTokens = usage.InputTokens
		outputTokens = usage.OutputTokens
		cacheRead = usage.CacheReadTokens
		cacheWrite = usage.CacheWriteTokens
		// Anthropic reports uncached input apart from cache reads and writes.
		inputTokens = max(0, inputTokens-cacheRead-cacheWrite)
	}
	return map[string]any{
		"input_tokens":                inputTokens,
		"output_tokens":               outputTokens,
		"cache_read_input_tokens":     cacheRead,
		"cache_creation_input_tokens": cacheWrite,
	}
}
