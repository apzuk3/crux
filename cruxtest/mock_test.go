package cruxtest_test

import (
	"context"
	"net/http"
	"testing"

	"github.com/apzuk3/crux"
	"github.com/apzuk3/crux/cruxtest"
	"github.com/stretchr/testify/require"
)

type WeatherArgs struct {
	City string `json:"city"`
}

func registerMockTools(t *testing.T) crux.ToolsRegistry {
	t.Helper()
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "get_weather", "Get weather for a city", func(ctx context.Context, in WeatherArgs) (string, *crux.StateDelta, error) {
		return "Sunny 22C in " + in.City, nil, nil
	})
	return reg
}

func TestOpenAIMock_TextPrompt(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnText("Hello from OpenAI mock!")

	agent, err := crux.New(
		"test-agent",
		crux.OpenAIGPT5_6Sol,
		crux.WithHTTPClient(mock.Client()),
		crux.WithAPIKey("mock-api-key"),
	)
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), agent)
	require.NoError(t, err)

	out, err := sess.Run(context.Background(), "Hi")
	require.NoError(t, err)
	require.Equal(t, "Hello from OpenAI mock!", out)

	mock.AssertAllConsumed(t)
	mock.AssertTurnCount(t, 1)

	reqs := mock.Requests()
	require.Len(t, reqs, 1)
	require.Contains(t, reqs[0].BodyString(), "Hi")
}

func TestOpenAIMock_ToolCalling(t *testing.T) {
	tools := registerMockTools(t)
	mock := cruxtest.NewMock()

	// Turn 1: Model requests tool call
	mock.Expect().ReturnToolCall("get_weather", WeatherArgs{City: "Paris"})
	// Turn 2: Model returns final text after seeing tool result
	mock.Expect().ReturnText("The weather in Paris is Sunny 22C.")

	agent, err := crux.New(
		"weather-agent",
		crux.OpenAIGPT5_6Sol,
		crux.WithHTTPClient(mock.Client()),
		crux.WithAPIKey("mock-api-key"),
		crux.WithToolsRegistry([]string{"get_weather"}, tools),
	)
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), agent)
	require.NoError(t, err)

	out, err := sess.Run(context.Background(), "What's the weather in Paris?")
	require.NoError(t, err)
	require.Equal(t, "The weather in Paris is Sunny 22C.", out)

	mock.AssertAllConsumed(t)
	mock.AssertTurnCount(t, 2)
}

func TestOpenAIMock_StructuredOutput(t *testing.T) {
	type SentimentResult struct {
		Sentiment string  `json:"sentiment"`
		Score     float64 `json:"score"`
	}

	mock := cruxtest.NewMock()
	mock.Expect().ReturnJSON(SentimentResult{
		Sentiment: "positive",
		Score:     0.98,
	})

	agent, err := crux.New(
		"sentiment-agent",
		crux.OpenAIGPT5_6Sol,
		crux.WithHTTPClient(mock.Client()),
		crux.WithAPIKey("mock-api-key"),
		crux.WithOutputSchemaFrom[SentimentResult](),
	)
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), agent)
	require.NoError(t, err)

	var res SentimentResult
	err = sess.RunInto(context.Background(), "I love Crux!", &res)
	require.NoError(t, err)
	require.Equal(t, "positive", res.Sentiment)
	require.Equal(t, 0.98, res.Score)

	mock.AssertAllConsumed(t)
}

func TestAnthropicMock_TextPrompt(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnText("Hello from Anthropic mock!")

	agent, err := crux.New(
		"anthropic-agent",
		crux.ClaudeHaiku4_5,
		crux.WithHTTPClient(mock.Client()),
		crux.WithAPIKey("mock-api-key"),
	)
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), agent)
	require.NoError(t, err)

	out, err := sess.Run(context.Background(), "Hello Claude")
	require.NoError(t, err)
	require.Equal(t, "Hello from Anthropic mock!", out)

	mock.AssertAllConsumed(t)
	mock.AssertTurnCount(t, 1)
}

func TestAnthropicMock_ToolCalling(t *testing.T) {
	tools := registerMockTools(t)
	mock := cruxtest.NewMock()

	mock.Expect().ReturnToolCall("get_weather", map[string]any{"city": "Tokyo"})
	mock.Expect().ReturnText("The weather in Tokyo is Sunny 22C in Tokyo.")

	agent, err := crux.New(
		"anthropic-agent",
		crux.ClaudeHaiku4_5,
		crux.WithHTTPClient(mock.Client()),
		crux.WithAPIKey("mock-api-key"),
		crux.WithToolsRegistry([]string{"get_weather"}, tools),
	)
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), agent)
	require.NoError(t, err)

	out, err := sess.Run(context.Background(), "How is Tokyo?")
	require.NoError(t, err)
	require.Equal(t, "The weather in Tokyo is Sunny 22C in Tokyo.", out)

	mock.AssertAllConsumed(t)
	mock.AssertTurnCount(t, 2)
}

func TestAnthropicMock_Refusal(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnRefusal("I cannot process this request.")

	agent, err := crux.New(
		"anthropic-agent",
		crux.ClaudeHaiku4_5,
		crux.WithHTTPClient(mock.Client()),
		crux.WithAPIKey("mock-api-key"),
	)
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), agent)
	require.NoError(t, err)

	_, err = sess.Run(context.Background(), "Do something bad")
	require.Error(t, err)
	require.Contains(t, err.Error(), "refused")
}

func TestGeminiMock_TextPrompt(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnText("Hello from Gemini mock!")

	agent, err := crux.New(
		"gemini-agent",
		crux.Gemini3_5FlashLite,
		crux.WithHTTPClient(mock.Client()),
		crux.WithAPIKey("mock-api-key"),
	)
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), agent)
	require.NoError(t, err)

	out, err := sess.Run(context.Background(), "Hello Gemini")
	require.NoError(t, err)
	require.Equal(t, "Hello from Gemini mock!", out)

	mock.AssertAllConsumed(t)
	mock.AssertTurnCount(t, 1)
}

func TestGeminiMock_ToolCalling(t *testing.T) {
	tools := registerMockTools(t)
	mock := cruxtest.NewMock()

	mock.Expect().ReturnToolCall("get_weather", map[string]any{"city": "Berlin"})
	mock.Expect().ReturnText("The weather in Berlin is Sunny 22C in Berlin.")

	agent, err := crux.New(
		"gemini-agent",
		crux.Gemini3_5FlashLite,
		crux.WithHTTPClient(mock.Client()),
		crux.WithAPIKey("mock-api-key"),
		crux.WithToolsRegistry([]string{"get_weather"}, tools),
	)
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), agent)
	require.NoError(t, err)

	out, err := sess.Run(context.Background(), "How is Berlin?")
	require.NoError(t, err)
	require.Equal(t, "The weather in Berlin is Sunny 22C in Berlin.", out)

	mock.AssertAllConsumed(t)
	mock.AssertTurnCount(t, 2)
}

func TestMock_ErrorStatus(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnError(http.StatusTooManyRequests, `{"error":{"message":"rate limit exceeded"}}`)

	agent, err := crux.New(
		"error-agent",
		crux.OpenAIGPT5_6Sol,
		crux.WithHTTPClient(mock.Client()),
		crux.WithAPIKey("mock-api-key"),
	)
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), agent)
	require.NoError(t, err)

	_, err = sess.Run(context.Background(), "Trigger 429")
	require.Error(t, err)
}

func TestMock_UnexpectedRequestError(t *testing.T) {
	mock := cruxtest.NewMock()
	// No expectations configured!

	agent, err := crux.New(
		"agent",
		crux.OpenAIGPT5_6Sol,
		crux.WithHTTPClient(mock.Client()),
		crux.WithAPIKey("mock-api-key"),
	)
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), agent)
	require.NoError(t, err)

	_, err = sess.Run(context.Background(), "Hi")
	require.Error(t, err)
	require.Contains(t, err.Error(), "no more expected turns")
}

func TestMock_AgentOptionsHelper(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnText("Works with AgentOptions!")

	agent, err := crux.New(
		"helper-agent",
		crux.OpenAIGPT5_6Sol,
		mock.AgentOptions()...,
	)
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), agent)
	require.NoError(t, err)

	out, err := sess.Run(context.Background(), "Test helper")
	require.NoError(t, err)
	require.Equal(t, "Works with AgentOptions!", out)
}

func TestMock_TokenUsage(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().
		ReturnText("Response with usage").
		WithUsage(cruxtest.TokenUsage{
			InputTokens:      150,
			OutputTokens:     42,
			CacheReadTokens:  20,
			CacheWriteTokens: 10,
		})

	agent, err := crux.New(
		"usage-agent",
		crux.ClaudeHaiku4_5,
		crux.WithHTTPClient(mock.Client()),
		crux.WithAPIKey("mock-key"),
	)
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), agent)
	require.NoError(t, err)

	out, err := sess.Run(context.Background(), "Test usage")
	require.NoError(t, err)
	require.Equal(t, "Response with usage", out)

	last := lastEntry(t, sess)
	require.NotNil(t, last.Usage)
	require.Equal(t, 150, last.Usage.InputTokens)
	require.Equal(t, 42, last.Usage.OutputTokens)
}

func TestMock_ParallelToolCalls(t *testing.T) {
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "tool_a", "Tool A", func(ctx context.Context, in map[string]any) (string, *crux.StateDelta, error) {
		return "result_a", nil, nil
	})
	crux.RegisterToolWithRegistry(reg, "tool_b", "Tool B", func(ctx context.Context, in map[string]any) (string, *crux.StateDelta, error) {
		return "result_b", nil, nil
	})

	mock := cruxtest.NewMock()
	// Turn 1: Parallel tool calling
	mock.Expect().ReturnToolCalls(
		cruxtest.NewToolCall("tool_a", map[string]any{"x": 1}),
		cruxtest.NewToolCall("tool_b", map[string]any{"y": 2}),
	)
	// Turn 2: Synthesize
	mock.Expect().ReturnText("Both tools executed successfully.")

	agent, err := crux.New(
		"parallel-agent",
		crux.OpenAIGPT5_6Sol,
		crux.WithHTTPClient(mock.Client()),
		crux.WithAPIKey("mock-key"),
		crux.WithToolsRegistry([]string{"tool_a", "tool_b"}, reg),
	)
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), agent)
	require.NoError(t, err)

	out, err := sess.Run(context.Background(), "Execute both")
	require.NoError(t, err)
	require.Equal(t, "Both tools executed successfully.", out)

	mock.AssertAllConsumed(t)
	mock.AssertTurnCount(t, 2)
}

func TestMock_OpenRouterAndDeepSeek(t *testing.T) {
	for _, model := range []string{crux.DeepSeekFlash, crux.XAIGrok4_20, crux.OpenRouterXAIGrok4_20} {
		t.Run(model, func(t *testing.T) {
			mock := cruxtest.NewMock()
			mock.Expect().ReturnText("Hello from " + model)

			agent, err := crux.New(
				"provider-agent",
				model,
				crux.WithHTTPClient(mock.Client()),
				crux.WithAPIKey("mock-key"),
			)
			require.NoError(t, err)

			sess, err := crux.NewSession(t.Context(), agent)
			require.NoError(t, err)

			out, err := sess.Run(context.Background(), "Ping")
			require.NoError(t, err)
			require.Equal(t, "Hello from "+model, out)

			mock.AssertAllConsumed(t)
		})
	}
}
