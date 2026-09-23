// Scenario: Cross-Provider Session Forking & State Portability
//
// This live evaluation test verifies Crux's core abstraction of provider-neutral session
// history and its ability to seamlessly migrate an active agent conversation across distinct
// LLM providers (Anthropic <-> OpenAI <-> Gemini).
//
// Rather than locking sessions to a single provider's proprietary message schema or state,
// Crux normalizes user queries, assistant responses, tool calls, and tool execution results
// into a canonical representation.
//
// Test Workflow:
// 1. Turn 1 (Source Provider A):
//    - An agent is created on Provider A (e.g. Anthropic Claude) with the 'calculate_tax' tool.
//    - Prompt: "Calculate the total for an item priced at 100 with a 10% rate using the calculate_tax tool."
//    - Provider A calls 'calculate_tax(amount: 100, rate: 0.10)' which returns 10.
//    - Provider A synthesizes the tool output into its response.
//
// 2. Cross-Provider Forking:
//    - The active agent session is forked to Provider B (e.g. OpenAI GPT or Google Gemini) using agent.Fork().
//    - Crux sanitizes provider-specific opaque metadata (e.g., Anthropic block tokens, OpenAI output IDs)
//      while preserving the complete semantic history (User -> ToolCall -> ToolResult -> Assistant).
//    - The test asserts all Opaque maps are stripped across provider boundaries.
//
// 3. Turn 2 (Target Provider B Replay & Audit):
//    - Provider B is given new auditor instructions: "Verify the previous calculation and state: 'VERIFIED: total is <total>'."
//    - Crux translates the canonical history into Provider B's native API message formats.
//    - Provider B parses the replayed tool execution and calculation history from Provider A without error.
//    - Provider B outputs the verified total ("VERIFIED: total is 110"), proving end-to-end multi-provider portability.
//
// Evaluation Matrix:
// Tests all 6 pairwise permutations across Anthropic, OpenAI, and Google Gemini.

package evals

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/apzuk3/crux"
	"github.com/stretchr/testify/require"
)

const (
	turn1Instructions = "You are an assistant that computes order totals. You MUST call the 'calculate_tax' tool to obtain the tax amount before answering. Do not guess tax values."
	turn1Prompt       = "Calculate the total for an item priced at 100 with a 10% rate using the calculate_tax tool."

	forkedInstructions = "You are a billing auditor reviewing a previous assistant's tax calculation. Read the conversation history, verify the calculated total, and state whether the tax was calculated correctly."
	forkedPrompt       = "Verify the previous calculation and state: 'VERIFIED: total is <total>'."
)

type CalculateTaxInput struct {
	Amount float64 `json:"amount" jsonschema:"description=The base amount"`
	Rate   float64 `json:"rate" jsonschema:"description=The tax rate as a decimal (e.g. 0.10 for 10%)"`
}

func setupTaxTools(t *testing.T) crux.ToolsRegistry {
	t.Helper()
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "calculate_tax", "Calculate tax for a given amount and decimal rate (e.g. 0.10 for 10%)", func(ctx context.Context, in CalculateTaxInput) (float64, *crux.StateDelta, error) {
		return in.Amount * in.Rate, nil, nil
	})
	return reg
}

func hasProviderAPIKey(provider crux.Provider) bool {
	var envVars []string
	switch provider {
	case crux.ProviderOpenAI:
		envVars = []string{"OPENAI_API_KEY", "OPENAI_APIKEY", "OPENAI_KEY"}
	case crux.ProviderAnthropic:
		envVars = []string{"ANTHROPIC_API_KEY", "ANTHROPIC_APIKEY", "ANTHROPIC_KEY", "ANTHROPIC_AUTH_TOKEN"}
	case crux.ProviderGoogle:
		envVars = []string{"GOOGLE_API_KEY", "GOOGLE_APIKEY", "GOOGLE_KEY", "GEMINI_API_KEY", "GEMINI_APIKEY", "GEMINI_KEY"}
	}
	for _, env := range envVars {
		if os.Getenv(env) != "" {
			return true
		}
	}
	return false
}

func Test_CrossProviderForking(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name        string
		fromProv    crux.Provider
		toProv      crux.Provider
		sourceModel string
		targetModel string
	}{
		{
			name:        "Anthropic_to_OpenAI",
			fromProv:    crux.ProviderAnthropic,
			toProv:      crux.ProviderOpenAI,
			sourceModel: crux.ClaudeHaiku4_5,
			targetModel: crux.ChatModelGPT4_1Mini,
		},
		{
			name:        "OpenAI_to_Anthropic",
			fromProv:    crux.ProviderOpenAI,
			toProv:      crux.ProviderAnthropic,
			sourceModel: crux.ChatModelGPT4_1Mini,
			targetModel: crux.ClaudeHaiku4_5,
		},
		{
			name:        "OpenAI_to_Gemini",
			fromProv:    crux.ProviderOpenAI,
			toProv:      crux.ProviderGoogle,
			sourceModel: crux.ChatModelGPT4_1Mini,
			targetModel: crux.Gemini3_5FlashLite,
		},
		{
			name:        "Gemini_to_OpenAI",
			fromProv:    crux.ProviderGoogle,
			toProv:      crux.ProviderOpenAI,
			sourceModel: crux.Gemini3_5FlashLite,
			targetModel: crux.ChatModelGPT4_1Mini,
		},
		{
			name:        "Gemini_to_Anthropic",
			fromProv:    crux.ProviderGoogle,
			toProv:      crux.ProviderAnthropic,
			sourceModel: crux.Gemini3_5FlashLite,
			targetModel: crux.ClaudeHaiku4_5,
		},
		{
			name:        "Anthropic_to_Gemini",
			fromProv:    crux.ProviderAnthropic,
			toProv:      crux.ProviderGoogle,
			sourceModel: crux.ClaudeHaiku4_5,
			targetModel: crux.Gemini3_5FlashLite,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			if !hasProviderAPIKey(tc.fromProv) {
				t.Skipf("Skipping %s: API key for source provider %s not set in environment", tc.name, tc.fromProv)
			}
			if !hasProviderAPIKey(tc.toProv) {
				t.Skipf("Skipping %s: API key for target provider %s not set in environment", tc.name, tc.toProv)
			}

			tools := setupTaxTools(t)
			agent, err := crux.New(
				"tax-calculator",
				tc.sourceModel,
				crux.WithInstructions(turn1Instructions),
				crux.WithToolsRegistry([]string{"calculate_tax"}, tools),
				crux.WithMaxTurns(10),
			)
			require.NoError(t, err)

			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()

			// Turn 1 on Provider A
			out1, err := agent.Run(ctx, turn1Prompt)
			require.NoError(t, err)
			require.NotEmpty(t, out1)
			t.Logf("[%s] Turn 1 out: %s", tc.name, out1)

			// Verify Provider A invoked calculate_tax tool
			var sawTaxToolCall bool
			for _, entry := range agent.Logs() {
				if entry.Kind == crux.KindToolCall && entry.ToolCall != nil && entry.ToolCall.Name == "calculate_tax" {
					sawTaxToolCall = true
					t.Logf("[%s] ToolCall Args: %s", tc.name, string(entry.ToolCall.Args))
				}
				if entry.Kind == crux.KindToolResult && entry.ToolResult != nil {
					t.Logf("[%s] ToolResult Output: %s", tc.name, entry.ToolResult.Output)
				}
			}
			require.True(t, sawTaxToolCall, "expected Turn 1 to invoke calculate_tax tool")

			// Fork onto Provider B
			forkedAgent, err := agent.Fork(
				crux.WithModel(tc.targetModel),
				crux.WithInstructions(forkedInstructions),
			)
			require.NoError(t, err)

			// Verify forking strips Opaque provider-specific entries without crashing
			for _, entry := range forkedAgent.Logs() {
				require.Empty(t, entry.Opaque, "expected Opaque entries to be stripped across providers")
			}

			// Turn 2 on Provider B
			out2, err := forkedAgent.Run(ctx, forkedPrompt)
			require.NoError(t, err)
			require.NotEmpty(t, out2)
			t.Logf("[%s] Turn 2 out: %s", tc.name, out2)

			// Verify Provider B parsed replayed history and emitted verified total
			require.Contains(t, strings.ToUpper(out2), "VERIFIED")
			require.Contains(t, out2, "110")
		})
	}
}
