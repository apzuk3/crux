package cruxtest

import (
	"testing"

	"crux.foo"
)

// The wire builders must produce byte-identical responses: the whole mock
// suite and the root tests depend on the exact JSON and SSE they emit. These
// goldens were captured from the builders and pin them across refactors.

type goldenTurn struct {
	name string
	turn *Turn
}

func goldenTurns() []goldenTurn {
	return []goldenTurn{
		{"text", (&Turn{}).ReturnText("Hello, world")},
		{"empty", (&Turn{}).ReturnEmpty()},
		{"tool_call", (&Turn{}).ReturnToolCall("get_weather", map[string]any{"city": "Paris"})},
		{"tool_calls_with_text", (&Turn{}).ReturnText("Checking").ReturnToolCalls(
			ToolCall{ID: "call_a", Name: "lookup", Args: `{"q":"x"}`},
			NewToolCall("noop", nil),
		)},
		{"blank_text_tool_call", (&Turn{}).ReturnToolCall("get_weather", nil).WithBlankText()},
		{"no_ids_signature", (&Turn{}).ReturnToolCall("get_weather", map[string]any{"city": "Paris"}).WithoutCallIDs().WithThoughtSignature("sig")},
		{"text_signature", (&Turn{}).ReturnText("Hi").WithThoughtSignature("sig")},
		{"refusal", (&Turn{}).ReturnRefusal("I cannot help with that")},
		{"usage", (&Turn{}).ReturnText("ok").WithUsage(TokenUsage{InputTokens: 100, OutputTokens: 20, CacheReadTokens: 30, CacheWriteTokens: 10})},
		{"raw_body", (&Turn{}).ReturnRaw(200, []byte(`{"custom":true}`))},
		{"status_error", (&Turn{}).ReturnRaw(503, nil)},
		{"status_error_body", (&Turn{}).ReturnError(429, `{"error":"slow down"}`)},
	}
}

type goldenBuilder struct {
	provider crux.Provider
	build    func(*Turn, int) ([]byte, error)
}

var goldenBuilders = []goldenBuilder{
	{crux.ProviderAnthropic, buildAnthropicResponse},
	{crux.ProviderGoogle, buildGeminiResponse},
	{crux.ProviderOpenAI, buildOpenAIResponse},
}

const goldenCallIndex = 7

func TestWireBuildersMatchGoldens(t *testing.T) {
	seen := make(map[string]bool)
	for _, tc := range goldenTurns() {
		for _, b := range goldenBuilders {
			checkTurnGoldens(t, seen, tc, b)
		}
	}
	for key := range wireGoldens {
		if !seen[key] {
			t.Errorf("%s: golden has no case", key)
		}
	}
}

// checkTurnGoldens checks one builder's response for a turn and, when the
// mock would stream it, the streaming variant too.
func checkTurnGoldens(t *testing.T, seen map[string]bool, tc goldenTurn, b goldenBuilder) {
	t.Helper()
	key := tc.name + "/" + string(b.provider)
	raw, err := b.build(tc.turn, goldenCallIndex)
	checkGolden(t, seen, key, raw, err)
	if !streamsGolden(tc.turn) {
		return
	}
	stream, err := buildStreamResponse(b.provider, raw)
	checkGolden(t, seen, key+"/stream", stream, err)
}

// streamsGolden mirrors the mock: only a regular response with no raw body
// is encoded as a stream.
func streamsGolden(turn *Turn) bool {
	return (turn.StatusCode == 0 || turn.StatusCode == 200) && len(turn.RawBody) == 0
}

func checkGolden(t *testing.T, seen map[string]bool, key string, got []byte, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("%s: build: %v", key, err)
	}
	seen[key] = true
	want, ok := wireGoldens[key]
	if !ok {
		t.Errorf("%s: no golden", key)
		return
	}
	if string(got) != want {
		t.Errorf("%s: output changed\n got: %s\nwant: %s", key, got, want)
	}
}

func TestValidateAnthropicRequest(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{"not json", `{`, "decode anthropic request: unexpected end of JSON input"},
		{"no messages", `{"messages":[]}`, "anthropic: messages must not be empty"},
		{"too many cache controls", `{"messages":[{"role":"user","content":"hi"}],"system":[{"cache_control":{}},{"cache_control":{}},{"cache_control":{}},{"cache_control":{}},{"cache_control":{}}]}`,
			"anthropic: a maximum of 4 blocks with cache_control may be provided, found 5"},
		{"bad content", `{"messages":[{"role":"user","content":42}]}`,
			"anthropic: messages.0: decode content: json: cannot unmarshal number into Go value of type []cruxtest.anthropicBlock"},
		{"empty content", `{"messages":[{"role":"user","content":[]}]}`, "anthropic: messages.0: content must not be empty"},
		{"whitespace text", `{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"text","text":" \n"}]}]}`,
			"anthropic: messages.1.content.0: text content blocks must contain non-whitespace text"},
		{"tool_result in assistant message", `{"messages":[{"role":"assistant","content":[{"type":"tool_result","tool_use_id":"t1"}]}]}`,
			"anthropic: messages.0.content.0: tool_result blocks belong in user messages"},
		{"tool_result after text", `{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"f","input":{}}]},{"role":"user","content":[{"type":"text","text":"x"},{"type":"tool_result","tool_use_id":"t1"}]}]}`,
			"anthropic: messages.2.content.1: tool_result blocks must come first in a user message"},
		{"tool_result without tool_use", `{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"f","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"t2"}]}]}`,
			"anthropic: messages.2: tool_result for \"t2\" has no tool_use in the previous message"},
		{"tool_use without tool_result", `{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"f","input":{}},{"type":"tool_use","id":"t2","name":"f","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1"}]}]}`,
			"anthropic: messages.2: tool_use \"t2\" has no tool_result immediately after it"},
		{"tool_use at the end", `{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"f","input":{}}]}]}`, ""},
		{"paired tool calls", `{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"text","text":"ok"},{"type":"tool_use","id":"t1","name":"f","input":{}},{"type":"tool_use","id":"t2","name":"f","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"t2"},{"type":"tool_result","tool_use_id":"t1"},{"type":"text","text":"next"}]}]}`, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			checkValidation(t, validateAnthropicRequest([]byte(tc.body)), tc.want)
		})
	}
}

func TestValidateGeminiRequest(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{"not json", `{`, "decode gemini request: unexpected end of JSON input"},
		{"no contents", `{"contents":[]}`, "gemini: contents must not be empty"},
		{"no parts", `{"contents":[{"role":"user","parts":[]}]}`, "gemini: contents.0: parts must not be empty"},
		{"metadata only", `{"contents":[{"role":"user","parts":[{"text":"hi"},{"thoughtSignature":"sig"}]}]}`,
			"gemini: contents.0.parts.1: part must carry data (such as text or functionCall), not only metadata"},
		{"null text", `{"contents":[{"role":"user","parts":[{"text":null}]}]}`,
			"gemini: contents.0.parts.0: part must carry data (such as text or functionCall), not only metadata"},
		{"empty text", `{"contents":[{"role":"model","parts":[{"text":""}]}]}`,
			"gemini: contents.0.parts.0: part must carry data (such as text or functionCall), not only metadata"},
		{"empty text with signature", `{"contents":[{"role":"model","parts":[{"text":"","thoughtSignature":"sig"}]}]}`,
			"gemini: contents.0.parts.0: part must carry data (such as text or functionCall), not only metadata"},
		{"function call", `{"contents":[{"role":"user","parts":[{"text":"hi"}]},{"role":"model","parts":[{"functionCall":{"name":"f","args":{}},"thoughtSignature":"sig"}]},{"role":"user","parts":[{"functionResponse":{"name":"f","response":{}}}]}]}`, ""},
		{"later field", `{"contents":[{"role":"user","parts":[{"text":null,"inlineData":{"mimeType":"image/png","data":"AA=="}}]}]}`, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			checkValidation(t, validateGeminiRequest([]byte(tc.body)), tc.want)
		})
	}
}

func checkValidation(t *testing.T, err error, want string) {
	t.Helper()
	if want == "" {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		return
	}
	if err == nil {
		t.Fatalf("expected error %q, got nil", want)
	}
	if got := err.Error(); got != want {
		t.Fatalf("error mismatch\n got: %s\nwant: %s", got, want)
	}
}

var wireGoldens = map[string]string{
	"text/anthropic": `{"content":[{"text":"Hello, world","type":"text"}],"id":"msg_7","role":"assistant","stop_reason":"end_turn","stop_sequence":null,"type":"message","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}}`,
	"text/anthropic/stream": `event: message_start
data: {"message":{"content":[],"id":"msg_7","role":"assistant","stop_reason":null,"stop_sequence":null,"type":"message","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}},"type":"message_start"}

event: content_block_start
data: {"content_block":{"text":"","type":"text"},"index":0,"type":"content_block_start"}

event: content_block_delta
data: {"delta":{"text":"Hell","type":"text_delta"},"index":0,"type":"content_block_delta"}

event: content_block_delta
data: {"delta":{"text":"o, w","type":"text_delta"},"index":0,"type":"content_block_delta"}

event: content_block_delta
data: {"delta":{"text":"orld","type":"text_delta"},"index":0,"type":"content_block_delta"}

event: content_block_stop
data: {"index":0,"type":"content_block_stop"}

event: message_delta
data: {"delta":{"stop_reason":"end_turn","stop_sequence":null},"type":"message_delta","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

`,
	"text/google": `{"candidates":[{"content":{"parts":[{"text":"Hello, world"}],"role":"model"},"finishReason":"STOP"}],"responseId":"gemini_resp_7","usageMetadata":{"cachedContentTokenCount":0,"candidatesTokenCount":5,"promptTokenCount":10}}`,
	"text/google/stream": `data: {"candidates":[{"content":{"parts":[{"text":"Hell"}],"role":"model"}}]}

data: {"candidates":[{"content":{"parts":[{"text":"o, w"}],"role":"model"}}]}

data: {"candidates":[{"content":{"parts":[{"text":"orld"}],"role":"model"}}]}

data: {"candidates":[{"finishReason":"STOP"}],"responseId":"gemini_resp_7","usageMetadata":{"cachedContentTokenCount":0,"candidatesTokenCount":5,"promptTokenCount":10}}

`,
	"text/openai": `{"id":"resp_7","output":[{"content":[{"annotations":[],"text":"Hello, world","type":"output_text"}],"id":"msg_7","role":"assistant","status":"completed","type":"message"}],"status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cache_write_tokens":0,"cached_tokens":0},"output_tokens":5}}`,
	"text/openai/stream": `event: response.output_text.delta
data: {"content_index":0,"delta":"Hell","output_index":0,"type":"response.output_text.delta"}

event: response.output_text.delta
data: {"content_index":0,"delta":"o, w","output_index":0,"type":"response.output_text.delta"}

event: response.output_text.delta
data: {"content_index":0,"delta":"orld","output_index":0,"type":"response.output_text.delta"}

event: response.completed
data: {"response":{"id":"resp_7","output":[{"content":[{"annotations":[],"text":"Hello, world","type":"output_text"}],"id":"msg_7","role":"assistant","status":"completed","type":"message"}],"status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cache_write_tokens":0,"cached_tokens":0},"output_tokens":5}},"type":"response.completed"}

`,
	"empty/anthropic": `{"content":[],"id":"msg_7","role":"assistant","stop_reason":"end_turn","stop_sequence":null,"type":"message","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}}`,
	"empty/anthropic/stream": `event: message_start
data: {"message":{"content":[],"id":"msg_7","role":"assistant","stop_reason":null,"stop_sequence":null,"type":"message","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}},"type":"message_start"}

event: message_delta
data: {"delta":{"stop_reason":"end_turn","stop_sequence":null},"type":"message_delta","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

`,
	"empty/google": `{"candidates":[{"content":{"parts":[],"role":"model"},"finishReason":"STOP"}],"responseId":"gemini_resp_7","usageMetadata":{"cachedContentTokenCount":0,"candidatesTokenCount":5,"promptTokenCount":10}}`,
	"empty/google/stream": `data: {"candidates":[{"finishReason":"STOP"}],"responseId":"gemini_resp_7","usageMetadata":{"cachedContentTokenCount":0,"candidatesTokenCount":5,"promptTokenCount":10}}

`,
	"empty/openai": `{"id":"resp_7","output":[],"status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cache_write_tokens":0,"cached_tokens":0},"output_tokens":5}}`,
	"empty/openai/stream": `event: response.completed
data: {"response":{"id":"resp_7","output":[],"status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cache_write_tokens":0,"cached_tokens":0},"output_tokens":5}},"type":"response.completed"}

`,
	"tool_call/anthropic": `{"content":[{"id":"toolu_7_1","input":{"city":"Paris"},"name":"get_weather","type":"tool_use"}],"id":"msg_7","role":"assistant","stop_reason":"tool_use","stop_sequence":null,"type":"message","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}}`,
	"tool_call/anthropic/stream": `event: message_start
data: {"message":{"content":[],"id":"msg_7","role":"assistant","stop_reason":null,"stop_sequence":null,"type":"message","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}},"type":"message_start"}

event: content_block_start
data: {"content_block":{"id":"toolu_7_1","input":{},"name":"get_weather","type":"tool_use"},"index":0,"type":"content_block_start"}

event: content_block_delta
data: {"delta":{"partial_json":"{\"ci","type":"input_json_delta"},"index":0,"type":"content_block_delta"}

event: content_block_delta
data: {"delta":{"partial_json":"ty\":","type":"input_json_delta"},"index":0,"type":"content_block_delta"}

event: content_block_delta
data: {"delta":{"partial_json":"\"Par","type":"input_json_delta"},"index":0,"type":"content_block_delta"}

event: content_block_delta
data: {"delta":{"partial_json":"is\"}","type":"input_json_delta"},"index":0,"type":"content_block_delta"}

event: content_block_stop
data: {"index":0,"type":"content_block_stop"}

event: message_delta
data: {"delta":{"stop_reason":"tool_use","stop_sequence":null},"type":"message_delta","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

`,
	"tool_call/google": `{"candidates":[{"content":{"parts":[{"functionCall":{"args":{"city":"Paris"},"id":"call_7_1","name":"get_weather"}}],"role":"model"},"finishReason":"STOP"}],"responseId":"gemini_resp_7","usageMetadata":{"cachedContentTokenCount":0,"candidatesTokenCount":5,"promptTokenCount":10}}`,
	"tool_call/google/stream": `data: {"candidates":[{"content":{"parts":[{"functionCall":{"args":{"city":"Paris"},"id":"call_7_1","name":"get_weather"}}],"role":"model"}}]}

data: {"candidates":[{"finishReason":"STOP"}],"responseId":"gemini_resp_7","usageMetadata":{"cachedContentTokenCount":0,"candidatesTokenCount":5,"promptTokenCount":10}}

`,
	"tool_call/openai": `{"id":"resp_7","output":[{"arguments":"{\"city\":\"Paris\"}","call_id":"call_7_1","id":"fc_7_1","name":"get_weather","status":"completed","type":"function_call"}],"status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cache_write_tokens":0,"cached_tokens":0},"output_tokens":5}}`,
	"tool_call/openai/stream": `event: response.completed
data: {"response":{"id":"resp_7","output":[{"arguments":"{\"city\":\"Paris\"}","call_id":"call_7_1","id":"fc_7_1","name":"get_weather","status":"completed","type":"function_call"}],"status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cache_write_tokens":0,"cached_tokens":0},"output_tokens":5}},"type":"response.completed"}

`,
	"tool_calls_with_text/anthropic": `{"content":[{"text":"Checking","type":"text"},{"id":"call_a","input":{"q":"x"},"name":"lookup","type":"tool_use"},{"id":"toolu_7_2","input":{},"name":"noop","type":"tool_use"}],"id":"msg_7","role":"assistant","stop_reason":"tool_use","stop_sequence":null,"type":"message","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}}`,
	"tool_calls_with_text/anthropic/stream": `event: message_start
data: {"message":{"content":[],"id":"msg_7","role":"assistant","stop_reason":null,"stop_sequence":null,"type":"message","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}},"type":"message_start"}

event: content_block_start
data: {"content_block":{"text":"","type":"text"},"index":0,"type":"content_block_start"}

event: content_block_delta
data: {"delta":{"text":"Chec","type":"text_delta"},"index":0,"type":"content_block_delta"}

event: content_block_delta
data: {"delta":{"text":"king","type":"text_delta"},"index":0,"type":"content_block_delta"}

event: content_block_stop
data: {"index":0,"type":"content_block_stop"}

event: content_block_start
data: {"content_block":{"id":"call_a","input":{},"name":"lookup","type":"tool_use"},"index":1,"type":"content_block_start"}

event: content_block_delta
data: {"delta":{"partial_json":"{\"q\"","type":"input_json_delta"},"index":1,"type":"content_block_delta"}

event: content_block_delta
data: {"delta":{"partial_json":":\"x\"","type":"input_json_delta"},"index":1,"type":"content_block_delta"}

event: content_block_delta
data: {"delta":{"partial_json":"}","type":"input_json_delta"},"index":1,"type":"content_block_delta"}

event: content_block_stop
data: {"index":1,"type":"content_block_stop"}

event: content_block_start
data: {"content_block":{"id":"toolu_7_2","input":{},"name":"noop","type":"tool_use"},"index":2,"type":"content_block_start"}

event: content_block_delta
data: {"delta":{"partial_json":"{}","type":"input_json_delta"},"index":2,"type":"content_block_delta"}

event: content_block_stop
data: {"index":2,"type":"content_block_stop"}

event: message_delta
data: {"delta":{"stop_reason":"tool_use","stop_sequence":null},"type":"message_delta","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

`,
	"tool_calls_with_text/google": `{"candidates":[{"content":{"parts":[{"text":"Checking"},{"functionCall":{"args":{"q":"x"},"id":"call_a","name":"lookup"}},{"functionCall":{"args":{},"id":"call_7_2","name":"noop"}}],"role":"model"},"finishReason":"STOP"}],"responseId":"gemini_resp_7","usageMetadata":{"cachedContentTokenCount":0,"candidatesTokenCount":5,"promptTokenCount":10}}`,
	"tool_calls_with_text/google/stream": `data: {"candidates":[{"content":{"parts":[{"text":"Chec"}],"role":"model"}}]}

data: {"candidates":[{"content":{"parts":[{"text":"king"}],"role":"model"}}]}

data: {"candidates":[{"content":{"parts":[{"functionCall":{"args":{"q":"x"},"id":"call_a","name":"lookup"}}],"role":"model"}}]}

data: {"candidates":[{"content":{"parts":[{"functionCall":{"args":{},"id":"call_7_2","name":"noop"}}],"role":"model"}}]}

data: {"candidates":[{"finishReason":"STOP"}],"responseId":"gemini_resp_7","usageMetadata":{"cachedContentTokenCount":0,"candidatesTokenCount":5,"promptTokenCount":10}}

`,
	"tool_calls_with_text/openai": `{"id":"resp_7","output":[{"arguments":"{\"q\":\"x\"}","call_id":"call_a","id":"fc_7_1","name":"lookup","status":"completed","type":"function_call"},{"arguments":"{}","call_id":"call_7_2","id":"fc_7_2","name":"noop","status":"completed","type":"function_call"},{"content":[{"annotations":[],"text":"Checking","type":"output_text"}],"id":"msg_7","role":"assistant","status":"completed","type":"message"}],"status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cache_write_tokens":0,"cached_tokens":0},"output_tokens":5}}`,
	"tool_calls_with_text/openai/stream": `event: response.output_text.delta
data: {"content_index":0,"delta":"Chec","output_index":2,"type":"response.output_text.delta"}

event: response.output_text.delta
data: {"content_index":0,"delta":"king","output_index":2,"type":"response.output_text.delta"}

event: response.completed
data: {"response":{"id":"resp_7","output":[{"arguments":"{\"q\":\"x\"}","call_id":"call_a","id":"fc_7_1","name":"lookup","status":"completed","type":"function_call"},{"arguments":"{}","call_id":"call_7_2","id":"fc_7_2","name":"noop","status":"completed","type":"function_call"},{"content":[{"annotations":[],"text":"Checking","type":"output_text"}],"id":"msg_7","role":"assistant","status":"completed","type":"message"}],"status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cache_write_tokens":0,"cached_tokens":0},"output_tokens":5}},"type":"response.completed"}

`,
	"blank_text_tool_call/anthropic": `{"content":[{"text":"\n\n","type":"text"},{"id":"toolu_7_1","input":{},"name":"get_weather","type":"tool_use"}],"id":"msg_7","role":"assistant","stop_reason":"tool_use","stop_sequence":null,"type":"message","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}}`,
	"blank_text_tool_call/anthropic/stream": `event: message_start
data: {"message":{"content":[],"id":"msg_7","role":"assistant","stop_reason":null,"stop_sequence":null,"type":"message","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}},"type":"message_start"}

event: content_block_start
data: {"content_block":{"text":"","type":"text"},"index":0,"type":"content_block_start"}

event: content_block_delta
data: {"delta":{"text":"\n\n","type":"text_delta"},"index":0,"type":"content_block_delta"}

event: content_block_stop
data: {"index":0,"type":"content_block_stop"}

event: content_block_start
data: {"content_block":{"id":"toolu_7_1","input":{},"name":"get_weather","type":"tool_use"},"index":1,"type":"content_block_start"}

event: content_block_delta
data: {"delta":{"partial_json":"{}","type":"input_json_delta"},"index":1,"type":"content_block_delta"}

event: content_block_stop
data: {"index":1,"type":"content_block_stop"}

event: message_delta
data: {"delta":{"stop_reason":"tool_use","stop_sequence":null},"type":"message_delta","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

`,
	"blank_text_tool_call/google": `{"candidates":[{"content":{"parts":[{"functionCall":{"args":{},"id":"call_7_1","name":"get_weather"}}],"role":"model"},"finishReason":"STOP"}],"responseId":"gemini_resp_7","usageMetadata":{"cachedContentTokenCount":0,"candidatesTokenCount":5,"promptTokenCount":10}}`,
	"blank_text_tool_call/google/stream": `data: {"candidates":[{"content":{"parts":[{"functionCall":{"args":{},"id":"call_7_1","name":"get_weather"}}],"role":"model"}}]}

data: {"candidates":[{"finishReason":"STOP"}],"responseId":"gemini_resp_7","usageMetadata":{"cachedContentTokenCount":0,"candidatesTokenCount":5,"promptTokenCount":10}}

`,
	"blank_text_tool_call/openai": `{"id":"resp_7","output":[{"arguments":"{}","call_id":"call_7_1","id":"fc_7_1","name":"get_weather","status":"completed","type":"function_call"}],"status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cache_write_tokens":0,"cached_tokens":0},"output_tokens":5}}`,
	"blank_text_tool_call/openai/stream": `event: response.completed
data: {"response":{"id":"resp_7","output":[{"arguments":"{}","call_id":"call_7_1","id":"fc_7_1","name":"get_weather","status":"completed","type":"function_call"}],"status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cache_write_tokens":0,"cached_tokens":0},"output_tokens":5}},"type":"response.completed"}

`,
	"no_ids_signature/anthropic": `{"content":[{"id":"toolu_7_1","input":{"city":"Paris"},"name":"get_weather","type":"tool_use"}],"id":"msg_7","role":"assistant","stop_reason":"tool_use","stop_sequence":null,"type":"message","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}}`,
	"no_ids_signature/anthropic/stream": `event: message_start
data: {"message":{"content":[],"id":"msg_7","role":"assistant","stop_reason":null,"stop_sequence":null,"type":"message","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}},"type":"message_start"}

event: content_block_start
data: {"content_block":{"id":"toolu_7_1","input":{},"name":"get_weather","type":"tool_use"},"index":0,"type":"content_block_start"}

event: content_block_delta
data: {"delta":{"partial_json":"{\"ci","type":"input_json_delta"},"index":0,"type":"content_block_delta"}

event: content_block_delta
data: {"delta":{"partial_json":"ty\":","type":"input_json_delta"},"index":0,"type":"content_block_delta"}

event: content_block_delta
data: {"delta":{"partial_json":"\"Par","type":"input_json_delta"},"index":0,"type":"content_block_delta"}

event: content_block_delta
data: {"delta":{"partial_json":"is\"}","type":"input_json_delta"},"index":0,"type":"content_block_delta"}

event: content_block_stop
data: {"index":0,"type":"content_block_stop"}

event: message_delta
data: {"delta":{"stop_reason":"tool_use","stop_sequence":null},"type":"message_delta","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

`,
	"no_ids_signature/google": `{"candidates":[{"content":{"parts":[{"functionCall":{"args":{"city":"Paris"},"name":"get_weather"},"thoughtSignature":"c2ln"}],"role":"model"},"finishReason":"STOP"}],"responseId":"gemini_resp_7","usageMetadata":{"cachedContentTokenCount":0,"candidatesTokenCount":5,"promptTokenCount":10}}`,
	"no_ids_signature/google/stream": `data: {"candidates":[{"content":{"parts":[{"functionCall":{"args":{"city":"Paris"},"name":"get_weather"},"thoughtSignature":"c2ln"}],"role":"model"}}]}

data: {"candidates":[{"finishReason":"STOP"}],"responseId":"gemini_resp_7","usageMetadata":{"cachedContentTokenCount":0,"candidatesTokenCount":5,"promptTokenCount":10}}

`,
	"no_ids_signature/openai": `{"id":"resp_7","output":[{"arguments":"{\"city\":\"Paris\"}","call_id":"call_7_1","id":"fc_7_1","name":"get_weather","status":"completed","type":"function_call"}],"status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cache_write_tokens":0,"cached_tokens":0},"output_tokens":5}}`,
	"no_ids_signature/openai/stream": `event: response.completed
data: {"response":{"id":"resp_7","output":[{"arguments":"{\"city\":\"Paris\"}","call_id":"call_7_1","id":"fc_7_1","name":"get_weather","status":"completed","type":"function_call"}],"status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cache_write_tokens":0,"cached_tokens":0},"output_tokens":5}},"type":"response.completed"}

`,
	"text_signature/anthropic": `{"content":[{"text":"Hi","type":"text"}],"id":"msg_7","role":"assistant","stop_reason":"end_turn","stop_sequence":null,"type":"message","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}}`,
	"text_signature/anthropic/stream": `event: message_start
data: {"message":{"content":[],"id":"msg_7","role":"assistant","stop_reason":null,"stop_sequence":null,"type":"message","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}},"type":"message_start"}

event: content_block_start
data: {"content_block":{"text":"","type":"text"},"index":0,"type":"content_block_start"}

event: content_block_delta
data: {"delta":{"text":"Hi","type":"text_delta"},"index":0,"type":"content_block_delta"}

event: content_block_stop
data: {"index":0,"type":"content_block_stop"}

event: message_delta
data: {"delta":{"stop_reason":"end_turn","stop_sequence":null},"type":"message_delta","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

`,
	"text_signature/google": `{"candidates":[{"content":{"parts":[{"text":"Hi","thoughtSignature":"c2ln"}],"role":"model"},"finishReason":"STOP"}],"responseId":"gemini_resp_7","usageMetadata":{"cachedContentTokenCount":0,"candidatesTokenCount":5,"promptTokenCount":10}}`,
	"text_signature/google/stream": `data: {"candidates":[{"content":{"parts":[{"text":"Hi"}],"role":"model"}}]}

data: {"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"c2ln"}],"role":"model"}}]}

data: {"candidates":[{"finishReason":"STOP"}],"responseId":"gemini_resp_7","usageMetadata":{"cachedContentTokenCount":0,"candidatesTokenCount":5,"promptTokenCount":10}}

`,
	"text_signature/openai": `{"id":"resp_7","output":[{"content":[{"annotations":[],"text":"Hi","type":"output_text"}],"id":"msg_7","role":"assistant","status":"completed","type":"message"}],"status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cache_write_tokens":0,"cached_tokens":0},"output_tokens":5}}`,
	"text_signature/openai/stream": `event: response.output_text.delta
data: {"content_index":0,"delta":"Hi","output_index":0,"type":"response.output_text.delta"}

event: response.completed
data: {"response":{"id":"resp_7","output":[{"content":[{"annotations":[],"text":"Hi","type":"output_text"}],"id":"msg_7","role":"assistant","status":"completed","type":"message"}],"status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cache_write_tokens":0,"cached_tokens":0},"output_tokens":5}},"type":"response.completed"}

`,
	"refusal/anthropic": `{"content":[{"text":"I cannot help with that","type":"text"}],"id":"msg_7","role":"assistant","stop_reason":"refusal","stop_sequence":null,"type":"message","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}}`,
	"refusal/anthropic/stream": `event: message_start
data: {"message":{"content":[],"id":"msg_7","role":"assistant","stop_reason":null,"stop_sequence":null,"type":"message","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}},"type":"message_start"}

event: content_block_start
data: {"content_block":{"text":"","type":"text"},"index":0,"type":"content_block_start"}

event: content_block_delta
data: {"delta":{"text":"I ca","type":"text_delta"},"index":0,"type":"content_block_delta"}

event: content_block_delta
data: {"delta":{"text":"nnot","type":"text_delta"},"index":0,"type":"content_block_delta"}

event: content_block_delta
data: {"delta":{"text":" hel","type":"text_delta"},"index":0,"type":"content_block_delta"}

event: content_block_delta
data: {"delta":{"text":"p wi","type":"text_delta"},"index":0,"type":"content_block_delta"}

event: content_block_delta
data: {"delta":{"text":"th t","type":"text_delta"},"index":0,"type":"content_block_delta"}

event: content_block_delta
data: {"delta":{"text":"hat","type":"text_delta"},"index":0,"type":"content_block_delta"}

event: content_block_stop
data: {"index":0,"type":"content_block_stop"}

event: message_delta
data: {"delta":{"stop_reason":"refusal","stop_sequence":null},"type":"message_delta","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":10,"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

`,
	"refusal/google": `{"candidates":[{"content":{"parts":[{"text":"I cannot help with that"}],"role":"model"},"finishReason":"SAFETY"}],"responseId":"gemini_resp_7","usageMetadata":{"cachedContentTokenCount":0,"candidatesTokenCount":5,"promptTokenCount":10}}`,
	"refusal/google/stream": `data: {"candidates":[{"content":{"parts":[{"text":"I ca"}],"role":"model"}}]}

data: {"candidates":[{"content":{"parts":[{"text":"nnot"}],"role":"model"}}]}

data: {"candidates":[{"content":{"parts":[{"text":" hel"}],"role":"model"}}]}

data: {"candidates":[{"content":{"parts":[{"text":"p wi"}],"role":"model"}}]}

data: {"candidates":[{"content":{"parts":[{"text":"th t"}],"role":"model"}}]}

data: {"candidates":[{"content":{"parts":[{"text":"hat"}],"role":"model"}}]}

data: {"candidates":[{"finishReason":"SAFETY"}],"responseId":"gemini_resp_7","usageMetadata":{"cachedContentTokenCount":0,"candidatesTokenCount":5,"promptTokenCount":10}}

`,
	"refusal/openai": `{"id":"resp_7","output":[{"content":[{"refusal":"I cannot help with that","type":"refusal"}],"id":"msg_7_refusal","role":"assistant","status":"completed","type":"message"}],"status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cache_write_tokens":0,"cached_tokens":0},"output_tokens":5}}`,
	"refusal/openai/stream": `event: response.completed
data: {"response":{"id":"resp_7","output":[{"content":[{"refusal":"I cannot help with that","type":"refusal"}],"id":"msg_7_refusal","role":"assistant","status":"completed","type":"message"}],"status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cache_write_tokens":0,"cached_tokens":0},"output_tokens":5}},"type":"response.completed"}

`,
	"usage/anthropic": `{"content":[{"text":"ok","type":"text"}],"id":"msg_7","role":"assistant","stop_reason":"end_turn","stop_sequence":null,"type":"message","usage":{"cache_creation_input_tokens":10,"cache_read_input_tokens":30,"input_tokens":60,"output_tokens":20}}`,
	"usage/anthropic/stream": `event: message_start
data: {"message":{"content":[],"id":"msg_7","role":"assistant","stop_reason":null,"stop_sequence":null,"type":"message","usage":{"cache_creation_input_tokens":10,"cache_read_input_tokens":30,"input_tokens":60,"output_tokens":20}},"type":"message_start"}

event: content_block_start
data: {"content_block":{"text":"","type":"text"},"index":0,"type":"content_block_start"}

event: content_block_delta
data: {"delta":{"text":"ok","type":"text_delta"},"index":0,"type":"content_block_delta"}

event: content_block_stop
data: {"index":0,"type":"content_block_stop"}

event: message_delta
data: {"delta":{"stop_reason":"end_turn","stop_sequence":null},"type":"message_delta","usage":{"cache_creation_input_tokens":10,"cache_read_input_tokens":30,"input_tokens":60,"output_tokens":20}}

event: message_stop
data: {"type":"message_stop"}

`,
	"usage/google": `{"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"},"finishReason":"STOP"}],"responseId":"gemini_resp_7","usageMetadata":{"cachedContentTokenCount":30,"candidatesTokenCount":20,"promptTokenCount":100}}`,
	"usage/google/stream": `data: {"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"}}]}

data: {"candidates":[{"finishReason":"STOP"}],"responseId":"gemini_resp_7","usageMetadata":{"cachedContentTokenCount":30,"candidatesTokenCount":20,"promptTokenCount":100}}

`,
	"usage/openai": `{"id":"resp_7","output":[{"content":[{"annotations":[],"text":"ok","type":"output_text"}],"id":"msg_7","role":"assistant","status":"completed","type":"message"}],"status":"completed","usage":{"input_tokens":100,"input_tokens_details":{"cache_write_tokens":10,"cached_tokens":30},"output_tokens":20}}`,
	"usage/openai/stream": `event: response.output_text.delta
data: {"content_index":0,"delta":"ok","output_index":0,"type":"response.output_text.delta"}

event: response.completed
data: {"response":{"id":"resp_7","output":[{"content":[{"annotations":[],"text":"ok","type":"output_text"}],"id":"msg_7","role":"assistant","status":"completed","type":"message"}],"status":"completed","usage":{"input_tokens":100,"input_tokens_details":{"cache_write_tokens":10,"cached_tokens":30},"output_tokens":20}},"type":"response.completed"}

`,
	"raw_body/anthropic":          `{"custom":true}`,
	"raw_body/google":             `{"custom":true}`,
	"raw_body/openai":             `{"custom":true}`,
	"status_error/anthropic":      `{"error":{"message":"HTTP 503 error","type":"api_error"},"type":"error"}`,
	"status_error/google":         `{"error":{"code":503,"message":"HTTP 503 error","status":"UNAVAILABLE"}}`,
	"status_error/openai":         `{"error":{"code":"http_503","message":"HTTP 503 error","type":"api_error"}}`,
	"status_error_body/anthropic": `{"error":"slow down"}`,
	"status_error_body/google":    `{"error":"slow down"}`,
	"status_error_body/openai":    `{"error":"slow down"}`,
}
