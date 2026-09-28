package cruxtest

import (
	"encoding/json"
	"fmt"
)

func buildOpenAIResponse(turn *Turn, callIndex int) ([]byte, error) {
	if turn.StatusCode != 0 && turn.StatusCode != 200 {
		if len(turn.RawBody) > 0 {
			return turn.RawBody, nil
		}
		errPayload := map[string]any{
			"error": map[string]any{
				"message": fmt.Sprintf("HTTP %d error", turn.StatusCode),
				"type":    "api_error",
				"code":    fmt.Sprintf("http_%d", turn.StatusCode),
			},
		}
		return json.Marshal(errPayload)
	}

	if len(turn.RawBody) > 0 {
		return turn.RawBody, nil
	}

	output := []any{}

	// 1. Tool calls
	for i, tc := range turn.ToolCalls {
		callID := tc.ID
		if callID == "" {
			callID = fmt.Sprintf("call_%d_%d", callIndex, i+1)
		}
		_, argsStr, err := normalizeToolArgs(tc.Args)
		if err != nil {
			return nil, err
		}
		output = append(output, map[string]any{
			"id":        fmt.Sprintf("fc_%d_%d", callIndex, i+1),
			"type":      "function_call",
			"call_id":   callID,
			"name":      tc.Name,
			"arguments": argsStr,
			"status":    "completed",
		})
	}

	// 2. Refusal
	if turn.Refusal != "" {
		output = append(output, map[string]any{
			"id":     fmt.Sprintf("msg_%d_refusal", callIndex),
			"type":   "message",
			"role":   "assistant",
			"status": "completed",
			"content": []any{
				map[string]any{
					"type":    "refusal",
					"refusal": turn.Refusal,
				},
			},
		})
	} else if !turn.empty && (turn.Text != "" || len(turn.ToolCalls) == 0) {
		// 3. Text output
		output = append(output, map[string]any{
			"id":     fmt.Sprintf("msg_%d", callIndex),
			"type":   "message",
			"role":   "assistant",
			"status": "completed",
			"content": []any{
				map[string]any{
					"type":        "output_text",
					"text":        turn.Text,
					"annotations": []any{},
				},
			},
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
	}

	resp := map[string]any{
		"id":     fmt.Sprintf("resp_%d", callIndex),
		"status": "completed",
		"output": output,
		"usage": map[string]any{
			"input_tokens":  inputTokens,
			"output_tokens": outputTokens,
			"input_tokens_details": map[string]any{
				"cached_tokens":      cacheRead,
				"cache_write_tokens": cacheWrite,
			},
		},
	}

	return json.Marshal(resp)
}
