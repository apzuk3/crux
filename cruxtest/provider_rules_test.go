package cruxtest_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"crux.foo"
	"crux.foo/cruxtest"
	"github.com/stretchr/testify/require"
)

func rulesSession(t *testing.T, provider crux.Provider, model string, mock *cruxtest.Mock, opts ...crux.AgentOption) *crux.Session {
	t.Helper()
	options := []crux.AgentOption{crux.WithProvider(provider),
		crux.WithToolsRegistry([]string{"get_weather"}, registerMockTools(t))}
	options = append(options, opts...)
	a, err := crux.New("rules", model, options...)
	require.NoError(t, err)
	s, err := crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client()))
	require.NoError(t, err)
	return s
}

func runOrStream(t *testing.T, s *crux.Session, stream bool, input any) ([]crux.Chunk, error) {
	t.Helper()
	if !stream {
		_, err := s.Run(t.Context(), input)
		return nil, err
	}
	var chunks []crux.Chunk
	for chunk, err := range s.Stream(t.Context(), input) {
		if err != nil {
			return chunks, err
		}
		chunks = append(chunks, chunk)
	}
	return chunks, nil
}

func TestAnthropicBlankTextBeforeToolUseIsNotReplayed(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			mock := cruxtest.NewMock(cruxtest.WithProvider(crux.ProviderAnthropic))
			mock.Expect().ReturnToolCall("get_weather", WeatherArgs{City: "Paris"}).WithBlankText()
			mock.Expect().ReturnText("Sunny.")
			s := rulesSession(t, crux.ProviderAnthropic, "test-model", mock)
			_, err := runOrStream(t, s, stream, "Weather?")
			require.NoError(t, err)
			require.Contains(t, mock.Requests()[1].BodyString(), `"tool_use"`)
			mock.AssertAllConsumed(t)
		})
	}
}

func TestGeminiSignatureOnlyChunkJoinsPreviousPart(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			mock := cruxtest.NewMock(cruxtest.WithProvider(crux.ProviderGoogle))
			mock.Expect().ReturnText("Hello.").WithThoughtSignature("sig-text")
			mock.Expect().ReturnToolCall("get_weather", WeatherArgs{City: "Paris"}).WithThoughtSignature("sig-call").WithoutCallIDs()
			mock.Expect().ReturnText("Sunny.")
			s := rulesSession(t, crux.ProviderGoogle, "test-model", mock)
			chunks, err := runOrStream(t, s, stream, "Hi")
			require.NoError(t, err)
			for _, chunk := range chunks {
				require.NotEmpty(t, chunk.Delta)
			}
			_, err = runOrStream(t, s, stream, "Weather?")
			require.NoError(t, err)
			mock.AssertAllConsumed(t)

			var body struct {
				Contents []struct {
					Role  string           `json:"role"`
					Parts []map[string]any `json:"parts"`
				} `json:"contents"`
			}
			require.NoError(t, mock.Requests()[1].UnmarshalBody(&body))
			model := body.Contents[1]
			require.Equal(t, "model", model.Role)
			require.Len(t, model.Parts, 1)
			require.Equal(t, "Hello.", model.Parts[0]["text"])
			require.NotEmpty(t, model.Parts[0]["thoughtSignature"])

			// The call without an id is answered by name, still without an id.
			require.NoError(t, mock.Requests()[2].UnmarshalBody(&body))
			last := body.Contents[len(body.Contents)-1]
			response := last.Parts[0]["functionResponse"].(map[string]any)
			require.Equal(t, "get_weather", response["name"])
			require.NotContains(t, response, "id")
		})
	}
}

func TestGeminiStructuredOutputWithTools(t *testing.T) {
	cases := []struct {
		model      string
		tools      bool
		wireSchema bool
	}{
		{crux.Gemini2_5Flash, true, false},
		{crux.Gemini2_5Flash, false, true},
		{crux.Gemini3FlashPreview, true, true},
		{"models/gemini-3.1-pro-preview", true, true},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%s tools=%v", tc.model, tc.tools), func(t *testing.T) {
			mock := cruxtest.NewMock(cruxtest.WithProvider(crux.ProviderGoogle))
			mock.Expect().ReturnText(`{"status":"ok","score":2,"note":null}`)
			options := []crux.AgentOption{crux.WithProvider(crux.ProviderGoogle), crux.WithOutputSchemaFrom[structuredAnswer]()}
			if tc.tools {
				options = append(options, crux.WithToolsRegistry([]string{"get_weather"}, registerMockTools(t)))
			}
			a, err := crux.New("structured", tc.model, options...)
			require.NoError(t, err)
			s, err := crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client()))
			require.NoError(t, err)
			_, err = s.Run(t.Context(), "Rate it")
			require.NoError(t, err)

			var body struct {
				GenerationConfig  map[string]any `json:"generationConfig"`
				SystemInstruction *struct {
					Parts []map[string]any `json:"parts"`
				} `json:"systemInstruction"`
			}
			require.NoError(t, mock.Requests()[0].UnmarshalBody(&body))
			_, hasSchema := body.GenerationConfig["responseJsonSchema"]
			_, hasMIME := body.GenerationConfig["responseMimeType"]
			require.Equal(t, tc.wireSchema, hasSchema)
			require.Equal(t, tc.wireSchema, hasMIME)
			if !tc.wireSchema {
				require.NotNil(t, body.SystemInstruction)
				require.Contains(t, body.SystemInstruction.Parts[0]["text"], `"status"`)
			}
		})
	}
}

func TestOpenAIStreamsCommentaryAsReasoning(t *testing.T) {
	event := func(v map[string]any) string {
		data, _ := json.Marshal(v)
		return fmt.Sprintf("event: %s\ndata: %s\n\n", v["type"], data)
	}
	message := func(id, phase, text string) map[string]any {
		return map[string]any{"id": id, "type": "message", "role": "assistant", "status": "completed", "phase": phase,
			"content": []any{map[string]any{"type": "output_text", "text": text, "annotations": []any{}}}}
	}
	commentary, final := message("msg_c", "commentary", "Let me think."), message("msg_f", "final_answer", "Done.")
	var sse strings.Builder
	for i, item := range []map[string]any{commentary, final} {
		added := map[string]any{"id": item["id"], "type": "message", "role": "assistant", "status": "in_progress", "phase": item["phase"], "content": []any{}}
		sse.WriteString(event(map[string]any{"type": "response.output_item.added", "output_index": i, "item": added}))
		text := item["content"].([]any)[0].(map[string]any)["text"]
		sse.WriteString(event(map[string]any{"type": "response.output_text.delta", "item_id": item["id"], "output_index": i, "content_index": 0, "delta": text}))
	}
	sse.WriteString(event(map[string]any{"type": "response.completed", "response": map[string]any{
		"id": "resp_1", "status": "completed", "output": []any{commentary, final},
		"usage": map[string]any{"input_tokens": 1, "output_tokens": 1},
	}}))

	mock := cruxtest.NewMock(cruxtest.WithProvider(crux.ProviderOpenAI))
	mock.Expect().ReturnRaw(http.StatusOK, []byte(sse.String()))
	s := rulesSession(t, crux.ProviderOpenAI, "test-model", mock)
	chunks, err := runOrStream(t, s, true, "Go")
	require.NoError(t, err)
	require.Equal(t, []crux.Chunk{
		{Kind: crux.ChunkReasoning, Delta: "Let me think.", Turn: 1},
		{Kind: crux.ChunkText, Delta: "Done.", Turn: 1},
	}, chunks)
	out, ok := s.FinalOutput()
	require.True(t, ok)
	require.Equal(t, "Done.", out)
}

func TestMockRejectsInvalidRequests(t *testing.T) {
	cases := []struct {
		name     string
		provider crux.Provider
		url      string
		body     string
		want     string
	}{
		{"anthropic blank text", crux.ProviderAnthropic, "https://api.anthropic.com/v1/messages",
			`{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]},{"role":"assistant","content":[{"type":"text","text":"\n\n"}]}]}`,
			"non-whitespace"},
		{"anthropic empty content", crux.ProviderAnthropic, "https://api.anthropic.com/v1/messages",
			`{"messages":[{"role":"user","content":[]}]}`, "must not be empty"},
		{"anthropic result not first", crux.ProviderAnthropic, "https://api.anthropic.com/v1/messages",
			`{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"f","input":{}}]},{"role":"user","content":[{"type":"text","text":"x"},{"type":"tool_result","tool_use_id":"t1"}]}]}`,
			"must come first"},
		{"anthropic unknown result", crux.ProviderAnthropic, "https://api.anthropic.com/v1/messages",
			`{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"f","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"t2"}]}]}`,
			"has no tool_use"},
		{"gemini signature only", crux.ProviderGoogle, "https://generativelanguage.googleapis.com/v1beta/models/m:generateContent",
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]},{"role":"model","parts":[{"text":"a"},{"thoughtSignature":"c2ln"}]}]}`,
			"must carry data"},
		{"openai orphan output", crux.ProviderOpenAI, "https://api.openai.com/v1/responses",
			`{"input":[{"role":"user","content":"hi"},{"type":"function_call_output","call_id":"c1","output":"x"}]}`,
			"no tool call found"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mock := cruxtest.NewMock(cruxtest.WithProvider(tc.provider))
			mock.Expect().ReturnText("unused")
			req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, tc.url, bytes.NewReader([]byte(tc.body)))
			require.NoError(t, err)
			resp, err := mock.Client().Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			require.NoError(t, err)
			require.Equal(t, http.StatusBadRequest, resp.StatusCode)
			require.Contains(t, string(body), tc.want)
		})
	}
}
