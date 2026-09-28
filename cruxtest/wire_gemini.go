package cruxtest

import (
	"encoding/json"
	"fmt"
)

func buildGeminiResponse(turn *Turn, callIndex int) ([]byte, error) {
	if turn.StatusCode != 0 && turn.StatusCode != 200 {
		if len(turn.RawBody) > 0 {
			return turn.RawBody, nil
		}
		errPayload := map[string]any{
			"error": map[string]any{
				"code":    turn.StatusCode,
				"message": fmt.Sprintf("HTTP %d error", turn.StatusCode),
				"status":  "UNAVAILABLE",
			},
		}
		return json.Marshal(errPayload)
	}

	if len(turn.RawBody) > 0 {
		return turn.RawBody, nil
	}

	parts := []any{}
	finishReason := "STOP"

	if turn.empty {
		// No parts.
	} else if turn.Refusal != "" {
		finishReason = "SAFETY"
		parts = append(parts, map[string]any{
			"text": turn.Refusal,
		})
	} else if len(turn.ToolCalls) > 0 {
		if turn.Text != "" {
			parts = append(parts, map[string]any{
				"text": turn.Text,
			})
		}
		for i, tc := range turn.ToolCalls {
			toolID := tc.ID
			if toolID == "" {
				toolID = fmt.Sprintf("call_%d_%d", callIndex, i+1)
			}
			argsMap, _, err := normalizeToolArgs(tc.Args)
			if err != nil {
				return nil, err
			}
			parts = append(parts, map[string]any{
				"functionCall": map[string]any{
					"id":   toolID,
					"name": tc.Name,
					"args": argsMap,
				},
			})
		}
	} else {
		parts = append(parts, map[string]any{
			"text": turn.Text,
		})
	}

	promptTokens := 10
	candidatesTokens := 5
	cachedTokens := 0
	if turn.Usage != nil {
		promptTokens = turn.Usage.InputTokens
		candidatesTokens = turn.Usage.OutputTokens
		cachedTokens = turn.Usage.CacheReadTokens
	}

	resp := map[string]any{
		"candidates": []any{
			map[string]any{
				"content": map[string]any{
					"role":  "model",
					"parts": parts,
				},
				"finishReason": finishReason,
			},
		},
		"usageMetadata": map[string]any{
			"promptTokenCount":        promptTokens,
			"candidatesTokenCount":    candidatesTokens,
			"cachedContentTokenCount": cachedTokens,
		},
	}

	return json.Marshal(resp)
}
