package cruxtest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"

	"github.com/apzuk3/crux"
)

// buildStreamResponse encodes ordinary mock turns using each provider's SSE
// protocol. Text and argument deltas are fragmented to exercise accumulation.
func buildStreamResponse(provider crux.Provider, raw []byte) ([]byte, error) {
	var response map[string]any
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	write := func(event map[string]any) {
		data, _ := json.Marshal(event)
		if kind, ok := event["type"].(string); ok {
			fmt.Fprintf(&buf, "event: %s\n", kind)
		}
		fmt.Fprintf(&buf, "data: %s\n\n", data)
	}
	split := func(text string, emit func(string)) {
		runes := []rune(text)
		for len(runes) > 0 {
			n := min(4, len(runes))
			emit(string(runes[:n]))
			runes = runes[n:]
		}
	}
	switch provider {
	case crux.ProviderAnthropic:
		start := maps.Clone(response)
		start["content"] = []any{}
		start["stop_reason"] = nil
		write(map[string]any{"type": "message_start", "message": start})
		for i, value := range response["content"].([]any) {
			block := value.(map[string]any)
			initial := maps.Clone(block)
			var text, kind, field string
			switch block["type"] {
			case "text":
				text, _ = block["text"].(string)
				initial["text"] = ""
				kind, field = "text_delta", "text"
			case "tool_use":
				args, _ := json.Marshal(block["input"])
				text = string(args)
				initial["input"] = map[string]any{}
				kind, field = "input_json_delta", "partial_json"
			}
			write(map[string]any{"type": "content_block_start", "index": i, "content_block": initial})
			split(text, func(delta string) {
				write(map[string]any{"type": "content_block_delta", "index": i, "delta": map[string]any{"type": kind, field: delta}})
			})
			write(map[string]any{"type": "content_block_stop", "index": i})
		}
		write(map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": response["stop_reason"], "stop_sequence": nil}, "usage": response["usage"]})
		write(map[string]any{"type": "message_stop"})
	case crux.ProviderGoogle:
		candidate := response["candidates"].([]any)[0].(map[string]any)
		content := candidate["content"].(map[string]any)
		for _, value := range content["parts"].([]any) {
			part := value.(map[string]any)
			emit := func(p map[string]any) {
				write(map[string]any{"candidates": []any{map[string]any{"content": map[string]any{"role": "model", "parts": []any{p}}}}})
			}
			if text, ok := part["text"].(string); ok {
				split(text, func(delta string) { emit(map[string]any{"text": delta}) })
			} else {
				emit(part)
			}
		}
		write(map[string]any{"candidates": []any{map[string]any{"finishReason": candidate["finishReason"]}}, "usageMetadata": response["usageMetadata"]})
	default:
		for i, value := range response["output"].([]any) {
			item := value.(map[string]any)
			if item["type"] == "message" {
				for j, value := range item["content"].([]any) {
					part := value.(map[string]any)
					text, _ := part["text"].(string)
					split(text, func(delta string) {
						write(map[string]any{"type": "response.output_text.delta", "output_index": i, "content_index": j, "delta": delta})
					})
				}
			}
		}
		write(map[string]any{"type": "response.completed", "response": response})
	}
	return buf.Bytes(), nil
}
