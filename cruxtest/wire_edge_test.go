package cruxtest_test

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/apzuk3/crux"
	"github.com/apzuk3/crux/cruxtest"
	"github.com/stretchr/testify/require"
)

var wireModels = []string{crux.OpenAIGPT5_6Sol, crux.ClaudeHaiku4_5, crux.Gemini2_5Flash}

func sseBody(events ...string) []byte {
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
	return []byte(b.String())
}

// lastEntry returns the session's last conversation entry, skipping the
// entries that only record a run's progress.
func lastEntry(t *testing.T, s *crux.Session) crux.Entry {
	t.Helper()
	logs := conversation(s.Logs())
	require.NotEmpty(t, logs)
	return logs[len(logs)-1]
}

// conversation drops the run, turn and tool-started records from logs.
func conversation(logs []crux.Entry) []crux.Entry {
	return slices.DeleteFunc(logs, func(e crux.Entry) bool {
		switch e.Kind {
		case crux.KindRunStarted, crux.KindRunFinished, crux.KindTurnStarted, crux.KindToolStarted:
			return true
		}
		return false
	})
}

func TestEmptyFinalAnswerKeepsUsage(t *testing.T) {
	for _, model := range wireModels {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%v", model, stream), func(t *testing.T) {
				mock := cruxtest.NewMock()
				mock.Expect().ReturnEmpty().WithUsage(cruxtest.TokenUsage{InputTokens: 12, OutputTokens: 3})
				mock.Expect().ReturnText("second")
				s := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, model)))

				var out string
				var err error
				if stream {
					for _, err = range s.Stream(t.Context(), "say nothing") {
						if err != nil {
							break
						}
					}
					out, _ = s.FinalOutput()
				} else {
					out, err = s.Run(t.Context(), "say nothing")
				}
				require.NoError(t, err)
				require.Equal(t, "", out)
				last := lastEntry(t, s)
				require.Equal(t, crux.KindAssistant, last.Kind)
				require.NotNil(t, last.Usage)
				require.Equal(t, 12, last.Usage.InputTokens)
				require.Equal(t, 3, last.Usage.OutputTokens)
				require.Equal(t, 1, mock.Calls())

				// The empty answer replays without sending empty content.
				out, err = s.Run(t.Context(), "now say something")
				require.NoError(t, err)
				require.Equal(t, "second", out)
				require.NotContains(t, mock.Requests()[1].BodyString(), `"text":""`)
				mock.AssertAllConsumed(t)
			})
		}
	}
}

func TestAnthropicToolUseWithoutContentFails(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnRaw(200, []byte(`{"id":"msg_1","type":"message","role":"assistant","content":[],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`))
	s := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, crux.ClaudeHaiku4_5)))
	_, err := s.Run(t.Context(), "hi")
	require.ErrorContains(t, err, "no content")
}

func TestAnthropicPausedTurnContinuationLimit(t *testing.T) {
	mock := cruxtest.NewMock()
	for range 11 {
		mock.Expect().ReturnRaw(200, []byte(`{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"working"}],"stop_reason":"pause_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}
	s := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, crux.ClaudeHaiku4_5)))
	_, err := s.Run(t.Context(), "hi")
	require.ErrorContains(t, err, "continuations")
	// One initial request plus ten continuations.
	require.Equal(t, 11, mock.Calls())
}

func TestGeminiSkipsEmptyParts(t *testing.T) {
	t.Run("generate", func(t *testing.T) {
		mock := cruxtest.NewMock()
		mock.Expect().ReturnRaw(200, []byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":""},{"text":"hello"},{}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2}}`))
		s := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, crux.Gemini2_5Flash)))
		out, err := s.Run(t.Context(), "hi")
		require.NoError(t, err)
		require.Equal(t, "hello", out)
		require.Len(t, conversation(s.Logs()), 2)
	})
	t.Run("stream", func(t *testing.T) {
		mock := cruxtest.NewMock()
		mock.Expect().ReturnRaw(200, sseBody(
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"hel"}]}}]}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":""}]}}]}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"lo"}]}}]}`,
			`{"candidates":[{"content":{"role":"model","parts":[{}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2}}`,
		))
		s := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, crux.Gemini2_5Flash)))
		for _, err := range s.Stream(t.Context(), "hi") {
			require.NoError(t, err)
		}
		out, ok := s.FinalOutput()
		require.True(t, ok)
		require.Equal(t, "hello", out)
	})
}

func TestAnthropicEmptyToolOutputSendsNoEmptyText(t *testing.T) {
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "quiet", "Returns nothing", func(ctx context.Context, in struct{}) (string, *crux.StateDelta, error) {
		return "", nil, nil
	})
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("quiet", map[string]any{})
	mock.Expect().ReturnText("done")
	agent := newMockAgent(t, mock, crux.ClaudeHaiku4_5, crux.WithToolsRegistry([]string{"quiet"}, reg))
	s := crux.MustSession(crux.NewSession(t.Context(), agent))
	out, err := s.Run(t.Context(), "go")
	require.NoError(t, err)
	require.Equal(t, "done", out)

	var body struct {
		Messages []struct {
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, mock.Requests()[1].UnmarshalBody(&body))
	var found bool
	for _, message := range body.Messages {
		for _, block := range message.Content {
			if block["type"] == "tool_result" {
				found = true
				require.Empty(t, block["content"])
			}
		}
	}
	require.True(t, found)
	require.NotContains(t, mock.Requests()[1].BodyString(), `"text":""`)
}

func TestAnthropicAuthTokenIsNotAnAPIKey(t *testing.T) {
	for _, key := range []string{"ANTHROPIC_API_KEY", "ANTHROPIC_APIKEY", "ANTHROPIC_KEY", "ANTHROPIC_PROFILE"} {
		t.Setenv(key, "")
	}
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "bearer-token")

	mock := cruxtest.NewMock()
	mock.Expect().ReturnText("hi")
	agent, err := crux.New("auth", crux.ClaudeHaiku4_5, crux.WithHTTPClient(mock.Client()))
	require.NoError(t, err)
	s := crux.MustSession(crux.NewSession(t.Context(), agent))
	_, err = s.Run(t.Context(), "hello")
	require.NoError(t, err)

	header := mock.Requests()[0].Header
	require.Empty(t, header.Get("X-Api-Key"))
	require.Equal(t, "Bearer bearer-token", header.Get("Authorization"))
}

func TestUsageInputTokensIncludeCache(t *testing.T) {
	cases := []struct {
		model                   string
		body                    string
		input, cacheRead, write int
	}{
		{crux.ClaudeHaiku4_5, `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":50,"output_tokens":5,"cache_read_input_tokens":30,"cache_creation_input_tokens":20}}`, 100, 30, 20},
		{crux.Gemini2_5Flash, `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":40,"toolUsePromptTokenCount":7,"candidatesTokenCount":5}}`, 107, 40, 0},
		{crux.OpenAIGPT5_6Sol, `{"id":"resp_1","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok","annotations":[]}]}],"usage":{"input_tokens":100,"output_tokens":5,"input_tokens_details":{"cached_tokens":40}}}`, 100, 40, 0},
	}
	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			mock := cruxtest.NewMock()
			mock.Expect().ReturnRaw(200, []byte(tc.body))
			s := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, tc.model)))
			_, err := s.Run(t.Context(), "hi")
			require.NoError(t, err)
			usage := lastEntry(t, s).Usage
			require.NotNil(t, usage)
			require.Equal(t, tc.input, usage.InputTokens)
			require.Equal(t, tc.cacheRead, usage.CacheReadTokens)
			require.Equal(t, tc.write, usage.CacheWriteTokens)
			require.Equal(t, 5, usage.OutputTokens)
		})
	}
}

func TestMockUsageMatchesCruxUsage(t *testing.T) {
	for _, model := range wireModels {
		t.Run(model, func(t *testing.T) {
			mock := cruxtest.NewMock()
			mock.Expect().ReturnText("ok").WithUsage(cruxtest.TokenUsage{InputTokens: 100, OutputTokens: 5, CacheReadTokens: 30})
			s := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, model)))
			_, err := s.Run(t.Context(), "hi")
			require.NoError(t, err)
			usage := lastEntry(t, s).Usage
			require.Equal(t, 100, usage.InputTokens)
			require.Equal(t, 30, usage.CacheReadTokens)
		})
	}
}

func TestSafetyStopsAreRefusals(t *testing.T) {
	cases := []struct {
		name   string
		model  string
		stream bool
		body   []byte
	}{
		{"gemini safety", crux.Gemini2_5Flash, false, []byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]},"finishReason":"SAFETY"}]}`)},
		{"gemini prohibited", crux.Gemini2_5Flash, false, []byte(`{"candidates":[{"finishReason":"PROHIBITED_CONTENT"}]}`)},
		{"gemini recitation", crux.Gemini2_5Flash, false, []byte(`{"candidates":[{"finishReason":"RECITATION"}]}`)},
		{"gemini blocked prompt", crux.Gemini2_5Flash, false, []byte(`{"promptFeedback":{"blockReason":"PROHIBITED_CONTENT"}}`)},
		{"gemini blocked prompt stream", crux.Gemini2_5Flash, true, sseBody(`{"promptFeedback":{"blockReason":"SAFETY"}}`)},
		{"gemini spii stream", crux.Gemini2_5Flash, true, sseBody(`{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]}}]}`, `{"candidates":[{"finishReason":"SPII"}]}`)},
		{"openai content filter", crux.OpenAIGPT5_6Sol, false, []byte(`{"id":"resp_1","status":"incomplete","incomplete_details":{"reason":"content_filter"},"output":[]}`)},
		{"openai content filter stream", crux.OpenAIGPT5_6Sol, true, sseBody(`{"type":"response.incomplete","response":{"id":"resp_1","status":"incomplete","incomplete_details":{"reason":"content_filter"},"output":[]}}`)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mock := cruxtest.NewMock()
			mock.Expect().ReturnRaw(200, tc.body)
			s := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, tc.model)))
			var err error
			if tc.stream {
				for _, err = range s.Stream(t.Context(), "hi") {
					if err != nil {
						break
					}
				}
			} else {
				_, err = s.Run(t.Context(), "hi")
			}
			require.ErrorIs(t, err, crux.ErrRefused)
		})
	}

	t.Run("gemini mock refusal", func(t *testing.T) {
		mock := cruxtest.NewMock()
		mock.Expect().ReturnRefusal("no")
		s := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, crux.Gemini2_5Flash)))
		_, err := s.Run(t.Context(), "hi")
		require.ErrorIs(t, err, crux.ErrRefused)
	})
}

func TestOllamaWebSearchNeedsRealKey(t *testing.T) {
	for _, key := range []string{"OLLAMA_API_KEY", "OLLAMA_APIKEY", "OLLAMA_KEY"} {
		t.Setenv(key, "")
	}
	mock := cruxtest.NewMock()
	_, err := crux.New("ollama", "llama3", crux.WithProvider(crux.ProviderOllama), crux.WithHTTPClient(mock.Client()), crux.WithWebSearch())
	require.ErrorContains(t, err, "Ollama API key is required")
}

type listItem struct {
	Name string `json:"name"`
}

func TestNonObjectOutputSchema(t *testing.T) {
	for _, provider := range []crux.Provider{crux.ProviderOpenAI, crux.ProviderXAI, crux.ProviderOpenrouter, crux.ProviderDeepSeek} {
		t.Run(string(provider), func(t *testing.T) {
			mock := cruxtest.NewMock(cruxtest.WithProvider(provider))
			agent, err := crux.New("list", "test-model", append(mock.AgentOptions(), crux.WithProvider(provider), crux.WithOutputSchemaFrom[[]listItem]())...)
			require.NoError(t, err)
			s := crux.MustSession(crux.NewSession(t.Context(), agent))
			_, err = s.Run(t.Context(), "list")
			require.ErrorContains(t, err, "object at the root")
			require.Equal(t, 0, mock.Calls())
		})
	}
	t.Run("google", func(t *testing.T) {
		mock := cruxtest.NewMock()
		mock.Expect().ReturnText(`[{"name":"a"},{"name":"b"}]`)
		agent := newMockAgent(t, mock, crux.Gemini2_5Flash, crux.WithOutputSchemaFrom[[]listItem]())
		s := crux.MustSession(crux.NewSession(t.Context(), agent))
		var items []listItem
		require.NoError(t, s.RunInto(t.Context(), &items, "list"))
		require.Equal(t, []listItem{{"a"}, {"b"}}, items)
	})
}

func TestOpenAIReasoningSummaryAndUnknownItems(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnRaw(200, []byte(`{"id":"resp_1","status":"completed","output":[
		{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"first"},{"type":"summary_text","text":"second"}]},
		{"id":"x_1","type":"mystery_call","status":"completed"},
		{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok","annotations":[]}]}
	],"usage":{"input_tokens":1,"output_tokens":1}}`))
	mock.Expect().ReturnText("again")
	s := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, crux.OpenAIGPT5_6Sol)))
	out, err := s.Run(t.Context(), "hi")
	require.NoError(t, err)
	require.Equal(t, "ok", out)
	logs := conversation(s.Logs())
	require.Equal(t, crux.KindReasoning, logs[1].Kind)
	require.Equal(t, "first\n\nsecond", logs[1].Reasoning.Summary)
	require.Equal(t, crux.KindProviderTool, logs[2].Kind)

	_, err = s.Run(t.Context(), "more")
	require.NoError(t, err)
	require.NotContains(t, mock.Requests()[1].BodyString(), "mystery_call")
}

func TestAnthropicUnknownBlockIsKept(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnRaw(200, []byte(`{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"mystery_block","id":"m1"},{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	mock.Expect().ReturnText("again")
	s := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, crux.ClaudeHaiku4_5)))
	out, err := s.Run(t.Context(), "hi")
	require.NoError(t, err)
	require.Equal(t, "ok", out)
	require.Equal(t, crux.KindProviderTool, conversation(s.Logs())[1].Kind)
	_, err = s.Run(t.Context(), "more")
	require.NoError(t, err)
	require.NotContains(t, mock.Requests()[1].BodyString(), "mystery_block")
}

func TestGeminiOtherPartsAreKeptAndReplayed(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnRaw(200, []byte(`{"candidates":[{"content":{"role":"model","parts":[{"executableCode":{"language":"PYTHON","code":"print(1)"}},{"text":"ok"}]},"finishReason":"STOP"}]}`))
	mock.Expect().ReturnText("again")
	s := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, crux.Gemini2_5Flash)))
	out, err := s.Run(t.Context(), "hi")
	require.NoError(t, err)
	require.Equal(t, "ok", out)
	require.Equal(t, crux.KindProviderTool, conversation(s.Logs())[1].Kind)
	_, err = s.Run(t.Context(), "more")
	require.NoError(t, err)
	require.Contains(t, mock.Requests()[1].BodyString(), "print(1)")
}

func TestGeminiMaxTokensClamped(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnText("ok")
	s := crux.MustSession(crux.NewSession(t.Context(), newMockAgent(t, mock, crux.Gemini2_5Flash, crux.WithMaxTokens(math.MaxInt))))
	_, err := s.Run(t.Context(), "hi")
	require.NoError(t, err)
	var body struct {
		GenerationConfig struct {
			MaxOutputTokens int64 `json:"maxOutputTokens"`
		} `json:"generationConfig"`
	}
	require.NoError(t, mock.Requests()[0].UnmarshalBody(&body))
	require.Equal(t, int64(math.MaxInt32), body.GenerationConfig.MaxOutputTokens)
}
