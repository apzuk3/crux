package cruxtest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"

	"crux.foo"
)

// buildStreamResponse encodes ordinary mock turns using each provider's SSE
// protocol. Text and argument deltas are fragmented to exercise accumulation.
func buildStreamResponse(provider crux.Provider, raw []byte) ([]byte, error) {
	var response map[string]any
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, err
	}
	var w sseWriter
	switch provider {
	case crux.ProviderAnthropic:
		w.streamAnthropic(response)
	case crux.ProviderGoogle:
		w.streamGemini(response)
	default:
		w.streamOpenAI(response)
	}
	return w.buf.Bytes(), nil
}

// sseWriter encodes events as server-sent events, one per provider chunk.
type sseWriter struct {
	buf bytes.Buffer
}

func (w *sseWriter) write(event map[string]any) {
	data, _ := json.Marshal(event)
	if kind, ok := event["type"].(string); ok {
		fmt.Fprintf(&w.buf, "event: %s\n", kind)
	}
	fmt.Fprintf(&w.buf, "data: %s\n\n", data)
}

// textDeltas splits text into runs of four runes.
func textDeltas(text string) []string {
	runes := []rune(text)
	var deltas []string
	for len(runes) > 0 {
		n := min(4, len(runes))
		deltas = append(deltas, string(runes[:n]))
		runes = runes[n:]
	}
	return deltas
}

func (w *sseWriter) streamAnthropic(response map[string]any) {
	start := maps.Clone(response)
	start["content"] = []any{}
	start["stop_reason"] = nil
	w.write(map[string]any{"type": "message_start", "message": start})
	for i, value := range response["content"].([]any) {
		w.streamAnthropicBlock(i, value.(map[string]any))
	}
	w.write(map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": response["stop_reason"], "stop_sequence": nil}, "usage": response["usage"]})
	w.write(map[string]any{"type": "message_stop"})
}

func (w *sseWriter) streamAnthropicBlock(i int, block map[string]any) {
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
	w.write(map[string]any{"type": "content_block_start", "index": i, "content_block": initial})
	for _, delta := range textDeltas(text) {
		w.write(map[string]any{"type": "content_block_delta", "index": i, "delta": map[string]any{"type": kind, field: delta}})
	}
	w.write(map[string]any{"type": "content_block_stop", "index": i})
}

func (w *sseWriter) streamGemini(response map[string]any) {
	candidate := response["candidates"].([]any)[0].(map[string]any)
	content := candidate["content"].(map[string]any)
	for _, value := range content["parts"].([]any) {
		w.streamGeminiPart(value.(map[string]any))
	}
	w.write(map[string]any{"candidates": []any{map[string]any{"finishReason": candidate["finishReason"]}}, "usageMetadata": response["usageMetadata"], "responseId": response["responseId"]})
}

func (w *sseWriter) streamGeminiPart(part map[string]any) {
	text, ok := part["text"].(string)
	if !ok {
		w.write(geminiPartEvent(part))
		return
	}
	for _, delta := range textDeltas(text) {
		w.write(geminiPartEvent(map[string]any{"text": delta}))
	}
	if signature, ok := part["thoughtSignature"]; ok {
		w.write(geminiPartEvent(map[string]any{"text": "", "thoughtSignature": signature}))
	}
}

func geminiPartEvent(part map[string]any) map[string]any {
	return map[string]any{"candidates": []any{map[string]any{"content": map[string]any{"role": "model", "parts": []any{part}}}}}
}

func (w *sseWriter) streamOpenAI(response map[string]any) {
	for i, value := range response["output"].([]any) {
		item := value.(map[string]any)
		if item["type"] != "message" {
			continue
		}
		for j, value := range item["content"].([]any) {
			part := value.(map[string]any)
			text, _ := part["text"].(string)
			for _, delta := range textDeltas(text) {
				w.write(map[string]any{"type": "response.output_text.delta", "output_index": i, "content_index": j, "delta": delta})
			}
		}
	}
	w.write(map[string]any{"type": "response.completed", "response": response})
}
