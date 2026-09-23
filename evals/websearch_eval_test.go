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
// - OpenAI: GPT-4.1 (crux.ChatModelGPT4_1Mini) & GPT-5 (crux.ChatModelGPT5Mini)
// - Anthropic: Claude Haiku 4.5 (crux.ClaudeHaiku4_5)
// - Google Gemini: Gemini 2.5 Flash (crux.Gemini2_5Flash)
// - xAI: Grok 4.20 (crux.XAIGrok4_20)

package evals

import (
	"context"
	"os"
	"regexp"
	"testing"
	"time"

	"github.com/apzuk3/crux"
	"github.com/stretchr/testify/require"
)

const (
	webSearchInstructions = "You are a real-time information bot. You MUST use provider-native web search to answer questions about recent events. Cite official URLs and dates in your response."
	webSearchUserPrompt   = "What was the latest stable version of Go (golang) released, and what is its release date? Use web search."
	webSearchTurn2Prompt  = "Summarize your previous answer in one short sentence."
)

var urlPattern = regexp.MustCompile(`https?://[^\s)\]">]+`)

func hasWebSearchAPIKey(provider crux.Provider) bool {
	var envVars []string
	switch provider {
	case crux.ProviderOpenAI:
		envVars = []string{"OPENAI_API_KEY", "OPENAI_APIKEY", "OPENAI_KEY"}
	case crux.ProviderAnthropic:
		envVars = []string{"ANTHROPIC_API_KEY", "ANTHROPIC_APIKEY", "ANTHROPIC_KEY", "ANTHROPIC_AUTH_TOKEN"}
	case crux.ProviderGoogle:
		envVars = []string{"GOOGLE_API_KEY", "GOOGLE_APIKEY", "GOOGLE_KEY", "GEMINI_API_KEY", "GEMINI_APIKEY", "GEMINI_KEY"}
	case crux.ProviderXAI:
		envVars = []string{"XAI_API_KEY", "XAI_APIKEY", "XAI_KEY"}
	}
	for _, env := range envVars {
		if os.Getenv(env) != "" {
			return true
		}
	}
	return false
}

func Test_WebSearchEval(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		provider crux.Provider
		model    string
	}{
		{
			name:     "OpenAI_GPT4_1",
			provider: crux.ProviderOpenAI,
			model:    crux.ChatModelGPT4_1Mini,
		},
		{
			name:     "OpenAI_GPT5",
			provider: crux.ProviderOpenAI,
			model:    crux.ChatModelGPT5Mini,
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
			t.Parallel()

			if !hasWebSearchAPIKey(tc.provider) {
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

			// Turn 1: Execute prompt requiring live web search and official citations
			out1, err := agent.Run(ctx, webSearchUserPrompt)
			require.NoError(t, err, "turn 1 web search query failed")
			require.NotEmpty(t, out1, "expected non-empty response from web search turn")
			t.Logf("[%s] Turn 1 Output:\n%s", tc.name, out1)

			logs := agent.Logs()

			// Validate URL links or citations in the final response
			hasURLOrCitation := urlPattern.MatchString(out1)
			if !hasURLOrCitation {
				// Gemini attaches source URLs via candidate grounding metadata
				for _, entry := range logs {
					if raw, ok := entry.Opaque["gemini.candidate.grounding_metadata"]; ok && urlPattern.MatchString(string(raw)) {
						hasURLOrCitation = true
						break
					}
					if raw, ok := entry.Opaque["anthropic.message.content_block"]; ok && urlPattern.MatchString(string(raw)) {
						hasURLOrCitation = true
						break
					}
				}
			}
			require.True(t, hasURLOrCitation, "expected response to contain valid URL links or citations")

			// Provider-specific audit validations
			switch tc.provider {
			case crux.ProviderGoogle:
				var sawGroundingMetadata bool
				for _, entry := range logs {
					if len(entry.Opaque["gemini.candidate.grounding_metadata"]) > 0 {
						sawGroundingMetadata = true
						t.Logf("[%s] Found gemini.candidate.grounding_metadata: %s", tc.name, string(entry.Opaque["gemini.candidate.grounding_metadata"]))
						break
					}
				}
				require.True(t, sawGroundingMetadata, "expected Gemini logs to contain 'gemini.candidate.grounding_metadata' in Opaque")

			case crux.ProviderAnthropic:
				var sawProviderTool bool
				for _, entry := range logs {
					if entry.Kind == crux.KindProviderTool {
						sawProviderTool = true
						t.Logf("[%s] Found Anthropic KindProviderTool entry with opaque keys: %v", tc.name, getOpaqueKeys(entry.Opaque))
					}
				}
				require.True(t, sawProviderTool, "expected Anthropic logs to contain KindProviderTool server tool events")

			case crux.ProviderOpenAI, crux.ProviderXAI:
				var sawProviderTool bool
				for _, entry := range logs {
					if entry.Kind == crux.KindProviderTool {
						sawProviderTool = true
						t.Logf("[%s] Found KindProviderTool entry with opaque keys: %v", tc.name, getOpaqueKeys(entry.Opaque))
					}
				}
				require.True(t, sawProviderTool, "expected logs to contain KindProviderTool server tool events")
			}

			// Turn 2: Verify multi-turn execution following search turn does not fail to serialize provider tool entries
			out2, err := agent.Run(ctx, webSearchTurn2Prompt)
			require.NoError(t, err, "turn 2 multi-turn execution failed after search turn")
			require.NotEmpty(t, out2, "expected non-empty response from turn 2")
			t.Logf("[%s] Turn 2 Output:\n%s", tc.name, out2)
		})
	}
}

func getOpaqueKeys(m map[string][]byte) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
