//go:build evals

// Scenario: Hierarchical Agent Delegation & State Delta Propagation
//
// This live evaluation test verifies Crux's hierarchical subagent delegation abstraction
// (crux.WithSubAgent) across major LLM providers (OpenAI, Anthropic, and Google Gemini).
// It exercises the complete subagent lifecycle:
//
// 1. Tool Wrapping & Reflection:
//    - The child agent ('summary-agent') defines a structured output contract via
//      crux.WithOutputSchemaFrom[SummaryOutput]().
//    - The parent coordinator ('coordinator-agent') registers the child via
//      crux.WithSubAgent(subAgent, "Summarizes technical incident reports into concise summaries").
//    - Crux automatically exposes the child as a callable tool ('agent_summary-agent')
//      and maps the subagent's OutputSchema as the tool's input parameter schema.
//
// 2. Autonomous Child Turn Execution:
//    - The parent coordinator is prompted to summarize an incident report using its summary agent.
//    - The parent LLM delegates the workload by invoking 'agent_summary-agent'.
//    - Crux intercepts the tool call, spawns an isolated execution turn for subAgent.Run(),
//      and captures the child's structured output.
//
// 3. State Delta Propagation & History Tracking:
//    - Crux automatically logs a KindStateDelta entry with Set[subAgent.Name] = output in the parent session.
//    - The subagent's execution result is recorded in the parent's session history as a KindToolResult.
//
// 4. Output Synthesis & Incorporation:
//    - The parent coordinator receives the child's structured output and incorporates the exact
//      summary (including the 'SUMMARY:' prefix) into its final response to the user.
//
// Evaluation Matrix:
// - OpenAI: crux.ChatModelGPT4_1Mini
// - Anthropic: crux.ClaudeHaiku4_5
// - Google Gemini: crux.Gemini3_5FlashLite

package evals

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/apzuk3/crux"
	"github.com/stretchr/testify/require"
)

type SummaryOutput struct {
	Summary string `json:"summary" jsonschema:"description=The text to summarize on input; the 1-sentence summary prefixed with SUMMARY: on output"`
}

const (
	childAgentInstructions = "You are a summarization bot. Given any text, return a 1-sentence summary with the word 'SUMMARY:' prefix."
	parentInstructions     = "You are a coordinator assistant. You have access to a specialist agent 'agent_summary-agent'. When asked to summarize documents, you MUST delegate the text to the subagent tool and return its exact output."
	incidentReportPrompt   = "Please summarize this incident report using your summary agent: 'Database connection pool exhausted at 04:00 UTC due to unindexed query on orders table. Re-indexed table at 04:30 UTC, service restored.'"
	subAgentDescription    = "Summarizes technical incident reports into concise summaries"
)

func Test_SubAgentDelegation(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		provider crux.Provider
		model    string
	}{
		{
			name:     "OpenAI",
			provider: crux.ProviderOpenAI,
			model:    crux.ChatModelGPT4_1Mini,
		},
		{
			name:     "Anthropic",
			provider: crux.ProviderAnthropic,
			model:    crux.ClaudeHaiku4_5,
		},
		{
			name:     "Gemini",
			provider: crux.ProviderGoogle,
			model:    crux.Gemini3_5FlashLite,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			if !hasProviderAPIKey(tc.provider) {
				t.Skipf("Skipping %s: API key for provider %s not set in environment", tc.name, tc.provider)
			}

			// 1. Child subagent with output schema
			subAgent, err := crux.New(
				"summary-agent",
				tc.model,
				crux.WithInstructions(childAgentInstructions),
				crux.WithOutputSchemaFrom[SummaryOutput](),
				crux.WithMaxTurns(5),
			)
			require.NoError(t, err)

			// 2. Parent coordinator with subagent tool
			parentAgent, err := crux.New(
				"coordinator-agent",
				tc.model,
				crux.WithInstructions(parentInstructions),
				crux.WithSubAgent(subAgent, subAgentDescription),
				crux.WithMaxTurns(10),
			)
			require.NoError(t, err)

			ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
			defer cancel()

			parentSess, err := crux.NewSession(ctx, parentAgent)
			require.NoError(t, err)

			finalAnswer, err := parentSess.Run(ctx, incidentReportPrompt)
			require.NoError(t, err)
			require.NotEmpty(t, finalAnswer)
			t.Logf("[%s] Parent final answer: %s", tc.name, finalAnswer)

			// 3. Assert parent invoked 'agent_summary-agent'
			var toolCallID string
			var sawSubAgentToolCall bool
			var sawSubAgentToolResult bool
			for _, entry := range parentSess.Logs() {
				if entry.Kind == crux.KindToolCall && entry.ToolCall != nil && entry.ToolCall.Name == "agent_summary-agent" {
					sawSubAgentToolCall = true
					toolCallID = entry.ToolCall.ID
					t.Logf("[%s] ToolCall Args: %s", tc.name, string(entry.ToolCall.Args))
				}
				if entry.Kind == crux.KindToolResult && entry.ToolResult != nil && entry.ToolResult.CallID == toolCallID && toolCallID != "" {
					sawSubAgentToolResult = true
					t.Logf("[%s] ToolResult Output: %s", tc.name, entry.ToolResult.Output)
				}
			}
			require.True(t, sawSubAgentToolCall, "expected parent to invoke agent_summary-agent")
			require.True(t, sawSubAgentToolResult, "expected parent to receive tool result from agent_summary-agent")

			// 4. Assert state delta populated: Set[subAgent.Name()]
			state := parentSess.StateSnapshot()
			rawVal, exists := state[subAgent.Name()]
			require.True(t, exists, "expected parent state snapshot to contain key for subagent %q", subAgent.Name())

			rawOutput, ok := rawVal.(string)
			require.True(t, ok, "expected subagent state value to be string")
			require.NotEmpty(t, rawOutput)
			t.Logf("[%s] Subagent snapshot output: %s", tc.name, rawOutput)

			var summaryOut SummaryOutput
			err = json.Unmarshal([]byte(rawOutput), &summaryOut)
			require.NoError(t, err, "subagent output should parse into SummaryOutput")
			require.NotEmpty(t, summaryOut.Summary, "summary field must not be empty")
			require.Contains(t, strings.ToUpper(summaryOut.Summary), "SUMMARY:", "subagent summary should contain SUMMARY: prefix")

			// 6. Assert parent incorporates the subagent output in its final answer
			require.Contains(t, strings.ToUpper(finalAnswer), "SUMMARY:", "parent final answer should incorporate SUMMARY: prefix")
			hasIncidentDetail := strings.Contains(strings.ToLower(finalAnswer), "order") ||
				strings.Contains(strings.ToLower(finalAnswer), "database") ||
				strings.Contains(strings.ToLower(finalAnswer), "restored") ||
				strings.Contains(strings.ToLower(finalAnswer), "04:30") ||
				strings.Contains(strings.ToLower(finalAnswer), "index")
			require.True(t, hasIncidentDetail, "parent final answer should incorporate incident report content from subagent")
		})
	}
}
