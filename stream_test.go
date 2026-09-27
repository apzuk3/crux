package crux_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/apzuk3/crux"
	"github.com/apzuk3/crux/cruxtest"
	"github.com/stretchr/testify/require"
)

var streamProviders = []crux.Provider{crux.ProviderOpenAI, crux.ProviderAnthropic, crux.ProviderGoogle, crux.ProviderOpenrouter, crux.ProviderXAI, crux.ProviderDeepSeek, crux.ProviderOllama}

func streamSession(t *testing.T, provider crux.Provider, mock *cruxtest.Mock, opts ...crux.AgentOption) *crux.Session {
	t.Helper()
	options := append(mock.AgentOptions(), crux.WithProvider(provider))
	options = append(options, opts...)
	a, err := crux.New("stream-test", "test-model", options...)
	require.NoError(t, err)
	s, err := crux.NewSession(t.Context(), a)
	require.NoError(t, err)
	return s
}

func collectStream(s *crux.Session, ctx context.Context, input any) ([]crux.Chunk, error) {
	var chunks []crux.Chunk
	for chunk, err := range s.Stream(ctx, input) {
		if err != nil {
			return chunks, err
		}
		chunks = append(chunks, chunk)
	}
	return chunks, nil
}

func TestStreamToolsAndUsage(t *testing.T) {
	for _, provider := range streamProviders {
		t.Run(string(provider), func(t *testing.T) {
			mock := cruxtest.NewMock(cruxtest.WithProvider(provider))
			mock.Expect().ReturnText("Checking.").ReturnToolCall("weather", map[string]string{"city": "Paris"})
			mock.Expect().ReturnText("Sunny ☀ in Paris").WithUsage(cruxtest.TokenUsage{InputTokens: 17, OutputTokens: 8})
			reg := crux.NewToolsRegistry()
			calls := 0
			crux.RegisterToolWithRegistry(reg, "weather", "Weather", func(ctx context.Context, args map[string]string) (string, *crux.StateDelta, error) {
				calls++
				require.Equal(t, "Paris", args["city"])
				return "Sunny", nil, nil
			})
			s := streamSession(t, provider, mock, crux.WithToolsRegistry([]string{"weather"}, reg))
			chunks, err := collectStream(s, t.Context(), "Weather?")
			require.NoError(t, err)
			require.Equal(t, 1, calls)
			texts := map[int]string{}
			for _, c := range chunks {
				require.Equal(t, crux.ChunkText, c.Kind)
				texts[c.Turn] += c.Delta
			}
			require.Equal(t, map[int]string{1: "Checking.", 2: "Sunny ☀ in Paris"}, texts)
			answer, ok := s.FinalOutput()
			require.True(t, ok)
			require.Equal(t, texts[2], answer)
			logs := s.Logs()
			require.Equal(t, 17, logs[len(logs)-1].Usage.InputTokens)
			require.Equal(t, 8, logs[len(logs)-1].Usage.OutputTokens)
			require.Contains(t, mock.Requests()[1].BodyString(), "Sunny")
			mock.AssertAllConsumed(t)
		})
	}
}

func TestStreamApprovalAndRepair(t *testing.T) {
	for _, provider := range streamProviders[:3] {
		t.Run(string(provider), func(t *testing.T) {
			mock := cruxtest.NewMock(cruxtest.WithProvider(provider))
			mock.Expect().ReturnToolCall("approve_me", map[string]any{})
			mock.Expect().ReturnText(`{"value":"wrong"}`)
			mock.Expect().ReturnText(`{"value":42}`)
			reg := crux.NewToolsRegistry()
			calls := 0
			crux.RegisterToolWithRegistry(reg, "approve_me", "Approval test", func(ctx context.Context, args map[string]any) (string, *crux.StateDelta, error) {
				calls++
				return "done", nil, nil
			}, crux.WithApprovalNeeded(true))
			type result struct {
				Value int `json:"value"`
			}
			s := streamSession(t, provider, mock, crux.WithToolsRegistry([]string{"approve_me"}, reg), crux.WithOutputSchemaFrom[result](), crux.WithMaxRepairs(1))
			_, err := collectStream(s, t.Context(), "Do it")
			require.ErrorIs(t, err, crux.ErrApprovalNeeded)
			require.Zero(t, calls)
			require.NoError(t, s.Approve(t.Context(), s.PendingApprovals()[0].ID))
			chunks, err := collectStream(s, t.Context(), nil)
			require.NoError(t, err)
			require.Equal(t, 1, calls)
			require.Equal(t, 2, chunks[len(chunks)-1].Turn)
			text, ok := s.FinalOutput()
			require.True(t, ok)
			require.Equal(t, `{"value":42}`, text)
		})
	}
}

func sse(events ...string) string {
	var b strings.Builder
	for _, event := range events {
		var value struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal([]byte(event), &value)
		if value.Type != "" {
			fmt.Fprintf(&b, "event: %s\n", value.Type)
		}
		fmt.Fprintf(&b, "data: %s\n\n", event)
	}
	return b.String()
}

func TestStreamReasoningAndReplay(t *testing.T) {
	cases := []struct {
		provider       crux.Provider
		body, metadata string
	}{
		{crux.ProviderOpenAI, sse(
			`{"type":"response.reasoning_summary_text.delta","delta":"Think"}`,
			`{"type":"response.output_text.delta","delta":"Hello"}`,
			`{"type":"response.completed","response":{"status":"completed","output":[{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Think"}],"encrypted_content":"encrypted-test"},{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello","annotations":[]}]}],"usage":{"input_tokens":10,"output_tokens":5}}}`,
		), "encrypted-test"},
		{crux.ProviderAnthropic, sse(
			`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Think"}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed-test"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello"}}`,
			`{"type":"content_block_stop","index":1}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`,
			`{"type":"message_stop"}`,
		), "signed-test"},
		{crux.ProviderGoogle, sse(
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"Think","thought":true}]}}]}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello","thoughtSignature":"c2lnbmVk"}]}}]}`,
			`{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":3,"thoughtsTokenCount":2}}`,
		), "c2lnbmVk"},
	}
	for _, tc := range cases {
		t.Run(string(tc.provider), func(t *testing.T) {
			mock := cruxtest.NewMock(cruxtest.WithProvider(tc.provider))
			mock.Expect().ReturnRaw(200, []byte(tc.body))
			mock.Expect().ReturnText("Again")
			s := streamSession(t, tc.provider, mock)
			chunks, err := collectStream(s, t.Context(), "Hi")
			require.NoError(t, err)
			require.Equal(t, []crux.Chunk{{Kind: crux.ChunkReasoning, Delta: "Think", Turn: 1}, {Kind: crux.ChunkText, Delta: "Hello", Turn: 1}}, chunks)
			logs := s.Logs()
			require.Equal(t, 5, logs[len(logs)-1].Usage.OutputTokens)
			_, err = collectStream(s, t.Context(), "Continue")
			require.NoError(t, err)
			require.Contains(t, mock.Requests()[1].BodyString(), tc.metadata)
		})
	}
}

// The server refuses to finish until the consumer observes the first delta.
// This detects implementations that buffer the entire response before yielding.
func TestStreamIncrementalAndEarlyExit(t *testing.T) {
	for _, mode := range []string{"complete", "break", "cancel", "truncated"} {
		t.Run(mode, func(t *testing.T) {
			observed := make(chan struct{})
			closed := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(closed)
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprint(w, sse(`{"type":"response.output_text.delta","delta":"Hi"}`))
				w.(http.Flusher).Flush()
				select {
				case <-observed:
				case <-r.Context().Done():
					return
				}
				if mode == "break" || mode == "cancel" {
					<-r.Context().Done()
					return
				}
				if mode == "complete" {
					fmt.Fprint(w, sse(`{"type":"response.completed","response":{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hi"}]}]}}`))
				}
			}))
			defer server.Close()
			a := crux.Must(crux.New("test", "test-model", crux.WithProvider(crux.ProviderOpenAI), crux.WithAPIKey("test"), crux.WithBaseURL(server.URL), crux.WithHTTPClient(server.Client())))
			s := crux.MustSession(crux.NewSession(t.Context(), a))
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			iterator := s.Stream(ctx, "Hi")
			require.Empty(t, s.Logs(), "iterator must be lazy")
			var streamErr error
			for chunk, err := range iterator {
				if err != nil {
					streamErr = err
					break
				}
				require.Equal(t, "Hi", chunk.Delta)
				close(observed)
				if mode == "break" {
					break
				}
				if mode == "cancel" {
					cancel()
				}
			}
			if mode == "cancel" {
				require.ErrorIs(t, streamErr, context.Canceled)
			} else if mode == "truncated" {
				require.ErrorContains(t, streamErr, "terminal response")
			} else {
				require.NoError(t, streamErr)
			}
			_, complete := s.FinalOutput()
			require.Equal(t, mode == "complete", complete)
			if !complete {
				require.Len(t, s.Logs(), 1)
			}
			select {
			case <-closed:
			case <-time.After(time.Second):
				t.Fatal("request was not closed")
			}
			for _, err := range iterator {
				require.ErrorContains(t, err, "already consumed")
			}
		})
	}
}

func TestStreamRejectsIncompleteOrFailedOutput(t *testing.T) {
	cases := []struct {
		name     string
		provider crux.Provider
		body     string
	}{
		{"openai incomplete", crux.ProviderOpenAI, sse(`{"type":"response.output_text.delta","delta":"partial"}`, `{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}`)},
		{"openai error", crux.ProviderOpenAI, sse(`{"type":"error","message":"failed"}`)},
		{"anthropic truncated", crux.ProviderAnthropic, sse(`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[]}}`, `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"partial"}}`)},
		{"anthropic error", crux.ProviderAnthropic, sse(`{"type":"error","error":{"type":"overloaded_error","message":"busy"}}`)},
		{"gemini truncated", crux.ProviderGoogle, sse(`{"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]}}]}`)},
		{"gemini blocked", crux.ProviderGoogle, sse(`{"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]},"finishReason":"SAFETY"}]}`)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mock := cruxtest.NewMock(cruxtest.WithProvider(tc.provider))
			mock.Expect().ReturnRaw(200, []byte(tc.body))
			s := streamSession(t, tc.provider, mock)
			_, err := collectStream(s, t.Context(), "Hello")
			require.Error(t, err)
			require.Len(t, s.Logs(), 1, "partial provider output must not be committed")
			_, ok := s.FinalOutput()
			require.False(t, ok)
		})
	}
}

func TestStreamAnthropicPausedTurn(t *testing.T) {
	response := func(text, stop string) []byte {
		return []byte(sse(
			`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"`+text+`"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"message_delta","delta":{"stop_reason":"`+stop+`"},"usage":{"output_tokens":5}}`,
			`{"type":"message_stop"}`,
		))
	}
	mock := cruxtest.NewMock(cruxtest.WithProvider(crux.ProviderAnthropic))
	mock.Expect().ReturnRaw(200, response("Searching. ", "pause_turn"))
	mock.Expect().ReturnRaw(200, response("Found it.", "end_turn"))
	s := streamSession(t, crux.ProviderAnthropic, mock)
	chunks, err := collectStream(s, t.Context(), "Search")
	require.NoError(t, err)
	require.Equal(t, []crux.Chunk{{Kind: crux.ChunkText, Delta: "Searching. ", Turn: 1}, {Kind: crux.ChunkText, Delta: "Found it.", Turn: 1}}, chunks)
	require.Contains(t, mock.Requests()[1].BodyString(), "Searching.")
	logs := s.Logs()
	require.Equal(t, 20, logs[len(logs)-1].Usage.InputTokens)
	require.Equal(t, 10, logs[len(logs)-1].Usage.OutputTokens)
}
