package cruxtest

import (
	"encoding/json"
	"fmt"
)

func buildAnthropicResponse(turn *Turn, callIndex int) ([]byte, error) {
	if turn.StatusCode != 0 && turn.StatusCode != 200 {
		if len(turn.RawBody) > 0 {
			return turn.RawBody, nil
		}
		errPayload := map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    "api_error",
				"message": fmt.Sprintf("HTTP %d error", turn.StatusCode),
			},
		}
		return json.Marshal(errPayload)
	}

	if len(turn.RawBody) > 0 {
		return turn.RawBody, nil
	}

	content := []any{}
	stopReason := "end_turn"

	if turn.empty {
		// No content blocks.
	} else if turn.Refusal != "" {
		stopReason = "refusal"
		content = append(content, map[string]any{
			"type": "text",
			"text": turn.Refusal,
		})
	} else if len(turn.ToolCalls) > 0 {
		stopReason = "tool_use"
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
		if turn.Text != "" {
			content = append([]any{
				map[string]any{
					"type": "text",
					"text": turn.Text,
				},
			}, content...)
		}
	} else {
		content = append(content, map[string]any{
			"type": "text",
			"text": turn.Text,
		})
	}

	inputTokens := 10
	outputTokens := 5
	cacheRead := 0
	cacheWrite := 0
	if turn.Usage != nil {
		inputTokens = turn.Usage.InputTokens
		outputTokens = turn.Usage.OutputTokens
		cacheRead = turn.Usage.CacheReadTokens
		cacheWrite = turn.Usage.CacheWriteTokens
		// Anthropic reports uncached input apart from cache reads and writes.
		inputTokens = max(0, inputTokens-cacheRead-cacheWrite)
	}

	resp := map[string]any{
		"id":            fmt.Sprintf("msg_%d", callIndex),
		"type":          "message",
		"role":          "assistant",
		"content":       content,
		"stop_reason":   stopReason,
		"stop_sequence": nil,
		"usage": map[string]any{
			"input_tokens":                inputTokens,
			"output_tokens":               outputTokens,
			"cache_read_input_tokens":     cacheRead,
			"cache_creation_input_tokens": cacheWrite,
		},
	}

	return json.Marshal(resp)
}
