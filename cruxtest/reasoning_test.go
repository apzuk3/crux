package cruxtest_test

import (
	"encoding/json"
	"testing"

	"crux.foo"
	"crux.foo/cruxtest"
	"github.com/stretchr/testify/require"
)

// reasoningRequest sends one request with the given effort and returns its body.
func reasoningRequest(t *testing.T, provider crux.Provider, model string, opts ...crux.AgentOption) map[string]any {
	t.Helper()
	mock := cruxtest.NewMock()
	mock.Expect().ReturnText("ok")
	a, err := crux.New("r", model, append([]crux.AgentOption{crux.WithProvider(provider)}, opts...)...)
	require.NoError(t, err)
	s, err := crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client()))
	require.NoError(t, err)
	_, err = s.Run(t.Context(), "hi")
	require.NoError(t, err)
	var body map[string]any
	require.NoError(t, json.Unmarshal([]byte(mock.Requests()[0].BodyString()), &body))
	return body
}

func TestReasoningAnthropic(t *testing.T) {
	tests := []struct {
		name      string
		model     string
		effort    crux.ReasoningEffort
		thinking  map[string]any
		effortOut any
		maxTokens float64
	}{
		{"adaptive", crux.ClaudeOpus4_8, crux.ReasoningHigh,
			map[string]any{"type": "adaptive", "display": "summarized"}, "high", 16384},
		{"off", crux.ClaudeSonnet5, crux.ReasoningOff, map[string]any{"type": "disabled"}, nil, 16384},
		{"off on a model that always thinks", crux.ClaudeOpus5_5, crux.ReasoningOff, nil, "low", 16384},
		{"off on fable", crux.ClaudeFable5_1, crux.ReasoningOff, nil, "low", 16384},
		{"budget before 4.6", crux.ClaudeHaiku4_5, crux.ReasoningMedium,
			map[string]any{"type": "enabled", "budget_tokens": float64(8192)}, nil, 8192 + 16384},
		{"dated budget model", crux.ClaudeSonnet4_5_20250929, crux.ReasoningLow,
			map[string]any{"type": "enabled", "budget_tokens": float64(2048)}, nil, 2048 + 16384},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := reasoningRequest(t, crux.ProviderAnthropic, tt.model, crux.WithReasoning(tt.effort))
			require.Equal(t, tt.maxTokens, body["max_tokens"])
			if tt.thinking == nil {
				require.NotContains(t, body, "thinking")
			} else {
				require.Equal(t, tt.thinking, body["thinking"])
			}
			config, _ := body["output_config"].(map[string]any)
			if tt.effortOut == nil {
				require.NotContains(t, config, "effort")
			} else {
				require.Equal(t, tt.effortOut, config["effort"])
			}
		})
	}

	t.Run("budget stays below max tokens", func(t *testing.T) {
		body := reasoningRequest(t, crux.ProviderAnthropic, crux.ClaudeHaiku4_5,
			crux.WithReasoning(crux.ReasoningMax), crux.WithMaxTokens(4000))
		require.Equal(t, float64(4000), body["max_tokens"])
		require.Equal(t, float64(3999), body["thinking"].(map[string]any)["budget_tokens"])
	})

	t.Run("unset sends nothing", func(t *testing.T) {
		body := reasoningRequest(t, crux.ProviderAnthropic, crux.ClaudeOpus4_8)
		require.NotContains(t, body, "thinking")
	})

	t.Run("temperature is rejected", func(t *testing.T) {
		_, err := crux.New("r", crux.ClaudeOpus4_8, crux.WithReasoning(crux.ReasoningLow), crux.WithTemperature(0.2))
		require.ErrorContains(t, err, "temperature")
		_, err = crux.New("r", crux.ClaudeOpus4_8, crux.WithReasoning(crux.ReasoningOff), crux.WithTemperature(0.2))
		require.NoError(t, err)
	})

	t.Run("unknown effort", func(t *testing.T) {
		_, err := crux.New("r", crux.ClaudeOpus4_8, crux.WithReasoning("extreme"))
		require.ErrorContains(t, err, "extreme")
	})
}

func TestReasoningOpenAI(t *testing.T) {
	body := reasoningRequest(t, crux.ProviderOpenAI, crux.OpenAIGPT5_4, crux.WithReasoning(crux.ReasoningMedium))
	require.Equal(t, map[string]any{"effort": "medium", "summary": "auto"}, body["reasoning"])

	body = reasoningRequest(t, crux.ProviderOpenAI, crux.OpenAIGPT5_4, crux.WithReasoning(crux.ReasoningOff))
	require.Equal(t, map[string]any{"effort": "none"}, body["reasoning"])

	body = reasoningRequest(t, crux.ProviderXAI, "grok-5", crux.WithReasoning(crux.ReasoningHigh))
	require.Equal(t, map[string]any{"effort": "high"}, body["reasoning"], "summary is OpenAI only")

	// The grok-4.20 family picks reasoning by model name and rejects the effort.
	for _, model := range []string{crux.XAIGrok4_20Reasoning, crux.XAIGrok4_20, crux.XAIGrok4_20NonReasoning} {
		_, err := crux.New("r", model, crux.WithReasoning(crux.ReasoningLow))
		require.ErrorContains(t, err, "-reasoning", model)
	}
	_, err := crux.New("r", crux.XAIGrok4_20Reasoning)
	require.NoError(t, err)
}

func TestReasoningGemini(t *testing.T) {
	thinking := func(body map[string]any) any {
		return body["generationConfig"].(map[string]any)["thinkingConfig"]
	}
	body := reasoningRequest(t, crux.ProviderGoogle, crux.Gemini3_8Flash, crux.WithReasoning(crux.ReasoningMax))
	require.Equal(t, map[string]any{"thinkingLevel": "HIGH", "includeThoughts": true}, thinking(body))

	body = reasoningRequest(t, crux.ProviderGoogle, crux.Gemini2_5Flash, crux.WithReasoning(crux.ReasoningOff))
	require.Equal(t, map[string]any{"thinkingBudget": float64(0)}, thinking(body))
}

func TestReasoningChangesAgentID(t *testing.T) {
	plain := crux.Must(crux.New("r", crux.ClaudeOpus4_8))
	reasoning := crux.Must(crux.New("r", crux.ClaudeOpus4_8, crux.WithReasoning(crux.ReasoningHigh)))
	require.NotEqual(t, plain.ID(), reasoning.ID())

	forked, err := crux.MustSession(crux.NewSession(t.Context(), reasoning)).Fork(t.Context())
	require.NoError(t, err)
	require.Equal(t, reasoning.ID(), forked.Agent().ID(), "a fork keeps the reasoning setting")
}
