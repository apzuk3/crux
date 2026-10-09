//go:build evals

// Scenario: Native Web Search, Source Grounding & Multi-Turn History Replay
//
// This live evaluation test verifies Crux's provider-native web search capabilities across all
// supported model providers (OpenAI, Anthropic, Google Gemini, and xAI). It exercises the full
// search lifecycle: configuration of geographic context, execution of provider-native search tools,
// inspection of grounding metadata and server-tool audit logs, and preservation of provider tool entries
// across multi-turn conversation history.
//
// Test Workflow:
// 1. Agent Initialization with Search & Geolocation:
//    - Initializes agents with crux.WithWebSearch(crux.WithUserLocation(crux.UserLocation{Country: "US"})).
//    - Crux maps geographic context to each provider's native schema (approximate location for OpenAI/Anthropic,
//      coordinates/location for Gemini).
//
// 2. Turn 1 (Native Web Search & Grounding):
//    - Instructions require real-time information retrieval:
//      "You are a real-time information bot. You MUST use provider-native web search to answer questions about recent events. Cite official URLs and dates in your response."
//    - User Prompt: "What was the latest stable version of Go (golang) released, and what is its release date? Use web search."
//    - The model decides autonomously to invoke native web search for real-time information.
//
// 3. Response & Grounding Verification:
//    - Validates that the response is non-empty and contains valid HTTP/HTTPS URLs or citations.
//    - Provider-specific audit checks via agent.Logs():
//      * Google Gemini: asserts presence of non-empty 'gemini.candidate.grounding_metadata' in entry.Opaque.
//      * Anthropic: asserts presence of KindProviderTool entries ('server_tool_use' / 'web_search_tool_result').
//      * OpenAI & xAI: asserts presence of KindProviderTool entries ('web_search_call').
//
// 4. Multi-Turn Serialization & Replay:
//    - Executes Turn 2 ("Summarize your previous answer in one short sentence.") on the same agent instance.
//    - Verifies that replaying conversation history containing provider tool entries serializes correctly back
//      to each provider's native API without schema errors or payload rejections.
//
// Evaluation Matrix:
// - OpenAI: GPT-4.1 (crux.OpenAIGPT4_1Mini) & GPT-5 (crux.OpenAIGPT5Mini)
// - Anthropic: Claude Haiku 4.5 (crux.ClaudeHaiku4_5)
// - Google Gemini: Gemini 2.5 Flash (crux.Gemini2_5Flash)
// - xAI: Grok 4.20 (crux.XAIGrok4_20)

package evals

import (
	"context"
	"regexp"
	"testing"
	"time"

	"crux.foo"
	"github.com/stretchr/testify/require"
)

const (
	webSearchInstructions = "You are a real-time information bot. You MUST use provider-native web search to answer questions about recent events. Cite official URLs and dates in your response."
	webSearchUserPrompt   = "What was the latest stable version of Go (golang) released, and what is its release date? Use web search."
	webSearchTurn2Prompt  = "Summarize your previous answer in one short sentence."
)

var urlPattern = regexp.MustCompile(`https?://[^\s)\]">]+`)

func Test_WebSearchEval(t *testing.T) {
	t.Parallel()

	testCases := []providerCase{
		{
			name:     "OpenAI_GPT4_1",
			provider: crux.ProviderOpenAI,
			model:    crux.OpenAIGPT4_1Mini,
		},
		{
			name:     "OpenAI_GPT5",
			provider: crux.ProviderOpenAI,
			model:    crux.OpenAIGPT5Mini,
		},
		{
			name:     "Anthropic_ClaudeHaiku4_5",
			provider: crux.ProviderAnthropic,
			model:    crux.ClaudeHaiku4_5,
		},
		{
			name:     "Gemini_2_5_Flash",
			provider: crux.ProviderGoogle,
			model:    crux.Gemini2_5Flash,
		},
		{
			name:     "xAI_Grok4_20",
			provider: crux.ProviderXAI,
			model:    crux.XAIGrok4_20,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			runWebSearchCase(t, tc)
		})
	}
}

func runWebSearchCase(t *testing.T, tc providerCase) {
	t.Parallel()

	if !hasAPIKey(tc.provider) {
		t.Skipf("Skipping %s: API key for provider %s not set in environment", tc.name, tc.provider)
	}

	agent, err := crux.New(
		"websearch-eval-"+tc.name,
		tc.model,
		crux.WithProvider(tc.provider),
		crux.WithWebSearch(crux.WithUserLocation(crux.UserLocation{Country: "US"})),
		crux.WithInstructions(webSearchInstructions),
		crux.WithMaxTurns(10),
	)
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	sess, err := crux.NewSession(ctx, agent)
	require.NoError(t, err)

	// Turn 1: Execute prompt requiring live web search and official citations
	out1, err := sess.Run(ctx, webSearchUserPrompt)
	require.NoError(t, err, "turn 1 web search query failed")
	require.NotEmpty(t, out1, "expected non-empty response from web search turn")
	t.Logf("[%s] Turn 1 Output:\n%s", tc.name, out1)

	logs := sess.Logs()

	// Validate URL links or citations in the final response
	require.True(t, hasCitation(out1, logs), "expected response to contain valid URL links or citations")

	// Provider-specific audit validations
	requireSearchAudit(t, tc.name, tc.provider, logs)

	// Turn 2: Verify multi-turn execution following search turn does not fail to serialize provider tool entries
	out2, err := sess.Run(ctx, webSearchTurn2Prompt)
	require.NoError(t, err, "turn 2 multi-turn execution failed after search turn")
	require.NotEmpty(t, out2, "expected non-empty response from turn 2")
	t.Logf("[%s] Turn 2 Output:\n%s", tc.name, out2)
}

// hasCitation reports whether the answer, or the provider metadata behind
// it, carries a URL. Gemini attaches source URLs via candidate grounding
// metadata.
func hasCitation(out string, logs []crux.Entry) bool {
	if urlPattern.MatchString(out) {
		return true
	}
	for _, entry := range logs {
		if urlPattern.Match(entry.Opaque["gemini.candidate.grounding_metadata"]) ||
			urlPattern.Match(entry.Opaque["anthropic.message.content_block"]) {
			return true
		}
	}
	return false
}

func requireSearchAudit(t *testing.T, tag string, provider crux.Provider, logs []crux.Entry) {
	t.Helper()
	switch provider {
	case crux.ProviderGoogle:
		raw, ok := firstOpaque(logs, "gemini.candidate.grounding_metadata")
		if ok {
			t.Logf("[%s] Found gemini.candidate.grounding_metadata: %s", tag, string(raw))
		}
		require.True(t, ok, "expected Gemini logs to contain 'gemini.candidate.grounding_metadata' in Opaque")

	case crux.ProviderAnthropic:
		require.True(t, hasKind(t, tag, logs, crux.KindProviderTool), "expected Anthropic logs to contain KindProviderTool server tool events")

	case crux.ProviderOpenAI, crux.ProviderXAI:
		require.True(t, hasKind(t, tag, logs, crux.KindProviderTool), "expected logs to contain KindProviderTool server tool events")
	}
}

// firstOpaque returns the first non-empty value stored under key.
func firstOpaque(logs []crux.Entry, key string) ([]byte, bool) {
	for _, entry := range logs {
		if raw := entry.Opaque[key]; len(raw) > 0 {
			return raw, true
		}
	}
	return nil, false
}

// hasKind reports whether the log has an entry of the kind, logging the
// opaque keys of each one.
func hasKind(t *testing.T, tag string, logs []crux.Entry, kind crux.Kind) bool {
	t.Helper()
	var found bool
	for _, entry := range logs {
		if entry.Kind == kind {
			found = true
			t.Logf("[%s] Found entry of kind %v with opaque keys: %v", tag, kind, getOpaqueKeys(entry.Opaque))
		}
	}
	return found
}

func getOpaqueKeys(m map[string][]byte) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
