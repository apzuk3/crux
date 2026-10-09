//go:build evals

package evals

import (
	"context"
	"os"
	"slices"
	"strings"
	"testing"

	"crux.foo"
	"github.com/stretchr/testify/require"
)

// apiKeyEnv lists the environment variables each provider reads its API key from.
var apiKeyEnv = map[crux.Provider][]string{
	crux.ProviderOpenAI:     {"OPENAI_API_KEY", "OPENAI_APIKEY", "OPENAI_KEY"},
	crux.ProviderAnthropic:  {"ANTHROPIC_API_KEY", "ANTHROPIC_APIKEY", "ANTHROPIC_KEY", "ANTHROPIC_AUTH_TOKEN"},
	crux.ProviderGoogle:     {"GOOGLE_API_KEY", "GOOGLE_APIKEY", "GOOGLE_KEY", "GEMINI_API_KEY", "GEMINI_APIKEY", "GEMINI_KEY"},
	crux.ProviderXAI:        {"XAI_API_KEY", "XAI_APIKEY", "XAI_KEY"},
	crux.ProviderOpenrouter: {"OPENROUTER_API_KEY", "OPENROUTER_APIKEY", "OPENROUTER_KEY"},
	crux.ProviderDeepSeek:   {"DEEPSEEK_API_KEY", "DEEPSEEK_APIKEY", "DEEPSEEK_KEY"},
	crux.ProviderOllama:     {"OLLAMA_API_KEY", "OLLAMA_APIKEY", "OLLAMA_KEY", "OLLAMA_HOST"},
}

func hasAPIKey(provider crux.Provider) bool {
	return slices.ContainsFunc(apiKeyEnv[provider], func(env string) bool { return os.Getenv(env) != "" })
}

// providerCase is one provider and model of an eval matrix.
type providerCase struct {
	name     string
	provider crux.Provider
	model    string
}

// newSessionOrSkip builds the agent and a session for it, skipping the test
// when the provider's API key is not set.
func newSessionOrSkip(ctx context.Context, t *testing.T, name, modelname string, opts ...crux.AgentOption) *crux.Session {
	t.Helper()
	agent, err := crux.New(name, modelname, opts...)
	if err != nil && strings.Contains(err.Error(), "API key for provider") {
		t.Skipf("Skipping %s: %v", modelname, err)
	}
	require.NoError(t, err)

	if !hasAPIKey(agent.Provider()) {
		t.Skipf("Skipping %s: API key for provider %s not set in environment", modelname, agent.Provider())
	}

	sess, err := crux.NewSession(ctx, agent)
	require.NoError(t, err)
	return sess
}

func containsAny(s string, subs ...string) bool {
	return slices.ContainsFunc(subs, func(sub string) bool { return strings.Contains(s, sub) })
}

func requireToolCalled(t *testing.T, tag string, logs []crux.Entry, name string) {
	t.Helper()
	var called bool
	for _, entry := range logs {
		if entry.Kind == crux.KindToolCall && entry.ToolCall != nil && entry.ToolCall.Name == name {
			called = true
			t.Logf("[%s] ToolCall Args: %s", tag, string(entry.ToolCall.Args))
		}
	}
	require.True(t, called, "expected the %s tool to be called", name)
}

func logToolResults(t *testing.T, tag string, logs []crux.Entry) {
	t.Helper()
	for _, entry := range logs {
		if entry.Kind == crux.KindToolResult && entry.ToolResult != nil {
			t.Logf("[%s] ToolResult Output: %s", tag, entry.ToolResult.Output)
		}
	}
}

func requireNoOpaque(t *testing.T, logs []crux.Entry) {
	t.Helper()
	for _, entry := range logs {
		require.Empty(t, entry.Opaque, "expected Opaque entries to be stripped across providers")
	}
}
