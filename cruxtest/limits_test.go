package cruxtest_test

import (
	"errors"
	"net/http"
	"testing"
	"time"

	"crux.foo"
	"crux.foo/cruxtest"
	"github.com/stretchr/testify/require"
)

func limitSession(t *testing.T, mock *cruxtest.Mock, model string, retries int) *crux.Session {
	t.Helper()
	a, err := crux.New("limits", model, crux.WithMaxRetries(retries))
	require.NoError(t, err)
	return crux.MustSession(crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client())))
}

func TestProviderLimitErrors(t *testing.T) {
	tests := []struct {
		name       string
		provider   crux.Provider
		model      string
		status     int
		body       string
		header     [2]string
		want       error
		retryAfter time.Duration
	}{
		{"openai rate limit", crux.ProviderOpenAI, crux.OpenAIGPT5_6Sol, 429,
			`{"error":{"message":"Rate limit reached","type":"requests","code":"rate_limit_exceeded"}}`,
			[2]string{"retry-after-ms", "1500"}, crux.ErrRateLimited, 1500 * time.Millisecond},
		{"openai insufficient quota is not a rate limit", crux.ProviderOpenAI, crux.OpenAIGPT5_6Sol, 429,
			`{"error":{"message":"You exceeded your current quota","type":"insufficient_quota","code":"insufficient_quota"}}`,
			[2]string{}, crux.ErrInsufficientCredits, 0},
		{"openai overloaded", crux.ProviderOpenAI, crux.OpenAIGPT5_6Sol, 503,
			`{"error":{"message":"The server is overloaded","type":"server_error","code":null}}`,
			[2]string{}, crux.ErrOverloaded, 0},
		{"anthropic rate limit", crux.ProviderAnthropic, crux.ClaudeHaiku4_5, 429,
			`{"type":"error","error":{"type":"rate_limit_error","message":"Number of request tokens has exceeded your per-minute rate limit"}}`,
			[2]string{"Retry-After", "7"}, crux.ErrRateLimited, 7 * time.Second},
		{"anthropic overloaded", crux.ProviderAnthropic, crux.ClaudeHaiku4_5, 529,
			`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`,
			[2]string{}, crux.ErrOverloaded, 0},
		{"anthropic credit balance", crux.ProviderAnthropic, crux.ClaudeHaiku4_5, 400,
			`{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API."}}`,
			[2]string{}, crux.ErrInsufficientCredits, 0},
		{"gemini resource exhausted", crux.ProviderGoogle, crux.Gemini2_5Flash, 429,
			`{"error":{"code":429,"message":"Quota exceeded","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"30s"}]}}`,
			[2]string{}, crux.ErrRateLimited, 30 * time.Second},
		{"gemini unavailable", crux.ProviderGoogle, crux.Gemini2_5Flash, 503,
			`{"error":{"code":503,"message":"The model is overloaded","status":"UNAVAILABLE"}}`,
			[2]string{}, crux.ErrOverloaded, 0},
		{"openrouter credits", crux.ProviderOpenrouter, crux.OpenRouterChatModelGPT5_4Mini, 402,
			`{"error":{"code":402,"message":"Insufficient credits"}}`,
			[2]string{}, crux.ErrInsufficientCredits, 0},
		{"openrouter upstream rate limit", crux.ProviderOpenrouter, crux.OpenRouterChatModelGPT5_4Mini, 429,
			`{"error":{"code":429,"message":"Provider returned error","metadata":{"provider_name":"OpenAI","error_type":"rate_limit_exceeded"}}}`,
			[2]string{"Retry-After", "2"}, crux.ErrRateLimited, 2 * time.Second},
		{"openrouter provider unavailable", crux.ProviderOpenrouter, crux.OpenRouterChatModelGPT5_4Mini, 502,
			`{"error":{"code":502,"message":"Provider returned error","metadata":{"error_type":"provider_unavailable"}}}`,
			[2]string{}, crux.ErrOverloaded, 0},
		{"deepseek insufficient balance", crux.ProviderDeepSeek, crux.DeepSeekFlash, 402,
			`{"error":{"message":"Insufficient Balance","type":"unknown_error","code":"invalid_request_error"}}`,
			[2]string{}, crux.ErrInsufficientCredits, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mock := cruxtest.NewMock(cruxtest.WithProvider(tt.provider))
			turn := mock.Expect().ReturnError(tt.status, tt.body)
			if tt.header[0] != "" {
				turn.WithHeader(tt.header[0], tt.header[1])
			}
			_, err := limitSession(t, mock, tt.model, 0).Run(t.Context(), "hi")
			require.ErrorIs(t, err, tt.want)
			for _, other := range []error{crux.ErrRateLimited, crux.ErrOverloaded, crux.ErrInsufficientCredits} {
				if other != tt.want {
					require.NotErrorIs(t, err, other)
				}
			}
			var perr *crux.ProviderError
			require.ErrorAs(t, err, &perr)
			require.Equal(t, tt.provider, perr.Provider)
			require.Equal(t, tt.status, perr.StatusCode)
			require.Equal(t, tt.retryAfter, perr.RetryAfter)
		})
	}
}

func TestProviderErrorUnrelated(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnError(http.StatusBadRequest, `{"error":{"message":"Invalid value for 'input'","type":"invalid_request_error","code":"invalid_value"}}`)
	_, err := limitSession(t, mock, crux.OpenAIGPT5_6Sol, 0).Run(t.Context(), "hi")
	require.Error(t, err)
	var perr *crux.ProviderError
	require.False(t, errors.As(err, &perr))
}

func TestProviderLimitInBody(t *testing.T) {
	tests := []struct {
		name     string
		provider crux.Provider
		model    string
		stream   bool
		body     string
		want     error
	}{
		{"anthropic overloaded event", crux.ProviderAnthropic, crux.ClaudeHaiku4_5, true,
			"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
			crux.ErrOverloaded},
		{"openrouter response.error event", crux.ProviderOpenrouter, crux.OpenRouterChatModelGPT5_4Mini, true,
			"event: response.error\ndata: {\"type\":\"response.error\",\"error\":{\"code\":\"rate_limit_exceeded\",\"message\":\"Rate limit exceeded\"}}\n\n",
			crux.ErrRateLimited},
		{"openrouter error event with a number code", crux.ProviderOpenrouter, crux.OpenRouterChatModelGPT5_4Mini, true,
			"data: {\"error\":{\"code\":429,\"message\":\"Rate limit exceeded\"}}\n\n",
			crux.ErrRateLimited},
		{"openrouter failed response", crux.ProviderOpenrouter, crux.OpenRouterChatModelGPT5_4Mini, false,
			`{"id":"resp_1","object":"response","created_at":1,"status":"failed","model":"openai/gpt-5.4-mini","output":[],"error":{"code":"server_error","message":"Upstream overloaded"},"error_type":"provider_overloaded"}`,
			crux.ErrOverloaded},
		{"openai failed response", crux.ProviderOpenAI, crux.OpenAIGPT5_6Sol, false,
			`{"id":"resp_1","object":"response","created_at":1,"status":"failed","model":"gpt-5.6-sol","output":[],"error":{"code":"rate_limit_exceeded","message":"Rate limit reached"}}`,
			crux.ErrRateLimited},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mock := cruxtest.NewMock(cruxtest.WithProvider(tt.provider))
			mock.Expect().ReturnRaw(http.StatusOK, []byte(tt.body))
			_, err := runOrStream(t, limitSession(t, mock, tt.model, 0), tt.stream, "hi")
			require.ErrorIs(t, err, tt.want)
			var perr *crux.ProviderError
			require.ErrorAs(t, err, &perr)
			require.Zero(t, perr.StatusCode)
		})
	}
}

func TestProviderLimitInStreamIsRetried(t *testing.T) {
	mock := cruxtest.NewMock(cruxtest.WithProvider(crux.ProviderOpenrouter))
	mock.Expect().ReturnRaw(http.StatusOK, []byte("event: response.error\ndata: {\"type\":\"response.error\",\"error\":{\"code\":\"rate_limit_exceeded\",\"message\":\"Rate limit exceeded\"}}\n\n"))
	mock.Expect().ReturnText("hello")
	chunks, err := runOrStream(t, limitSession(t, mock, crux.OpenRouterChatModelGPT5_4Mini, 1), true, "hi")
	require.NoError(t, err)
	require.NotEmpty(t, chunks)
	mock.AssertAllConsumed(t)
}

func TestProviderLimitAfterDeltaIsNotRetried(t *testing.T) {
	mock := cruxtest.NewMock(cruxtest.WithProvider(crux.ProviderAnthropic))
	mock.Expect().ReturnRaw(http.StatusOK, []byte(
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-haiku-4-5\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n"+
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"+
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}\n\n"+
			"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"))
	chunks, err := runOrStream(t, limitSession(t, mock, crux.ClaudeHaiku4_5, 2), true, "hi")
	require.ErrorIs(t, err, crux.ErrOverloaded)
	require.NotEmpty(t, chunks)
	mock.AssertTurnCount(t, 1)
}

func TestDecideLimitErrors(t *testing.T) {
	for _, tt := range []struct {
		name   string
		model  string
		status int
		want   error
	}{
		{"typesafe rate limit", crux.Jev, 429, crux.ErrRateLimited},
		{"typesafe credits", crux.Jev, 402, crux.ErrInsufficientCredits},
		{"openrouter decisions credits", crux.OpenRouterDecisionModelJev1_13, 402, crux.ErrInsufficientCredits},
	} {
		t.Run(tt.name, func(t *testing.T) {
			mock := cruxtest.NewMock()
			mock.Expect().ReturnError(tt.status, `{"detail":"limit"}`).WithHeader("Retry-After", "3")
			d := crux.MustDecider(crux.NewDecider(tt.model, crux.WithMaxRetries(0))).WithHTTPClient(mock.Client())
			_, err := crux.Decide[routed](t.Context(), d, "blank page")
			require.ErrorIs(t, err, tt.want)
			var perr *crux.ProviderError
			require.ErrorAs(t, err, &perr)
			require.Equal(t, tt.status, perr.StatusCode)
			require.Equal(t, 3*time.Second, perr.RetryAfter)
		})
	}
}
