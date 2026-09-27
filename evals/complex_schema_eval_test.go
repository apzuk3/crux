// Scenario: Complex Structured Output Validation & Strict Schema Enforcement
//
// This live evaluation test verifies Crux's typed structured output extraction and strict JSON schema
// enforcement across major LLM providers (OpenAI, Anthropic, Google Gemini, and xAI). It exercises
// the full structured output lifecycle:
//
// 1. Schema Reflection:
//    - Automatic Go type reflection via crux.WithOutputSchemaFrom[ProjectReport]() generating a compliant
//      JSON schema containing nested objects, slices, booleans, floating-point numbers, and strings.
//
// 2. Provider-Native Schema Enforcement:
//    - OpenAI / xAI: strict structured outputs mode (ResponseFormatTextJSONSchemaConfigParam with strict=true).
//    - Anthropic: OutputConfigParam with JSONOutputFormatParam schema constraint.
//    - Google Gemini: responseJsonSchema and responseMimeType="application/json".
//
// 3. Execution & Unmarshaling via agent.RunInto:
//    - The agent processes unstructured natural language input and constrains its response to the JSON schema.
//    - agent.RunInto(ctx, prompt, &report) decodes the model output directly into the target Go struct pointer.
//
// 4. Exact Field & Nested Data Assertions:
//    - Verifies all top-level scalar fields (Name, StartDate, LeadEmail), floating-point precision (Budget),
//      string slices (Tags), and nested struct slices (Milestones with boolean flags and due dates)
//      decode cleanly without unmarshaling errors or fidelity loss.
//
// Evaluation Matrix:
// - OpenAI: crux.ChatModelGPT4_1Mini (strict schema mode)
// - Anthropic: crux.ClaudeHaiku4_5 (JSON output format)
// - Google Gemini: crux.Gemini3_5FlashLite (responseJsonSchema)
// - xAI: crux.XAIGrok4_20 (OpenAI-compatible strict format)

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

type Milestone struct {
	Name      string `json:"name"`
	DueDate   string `json:"due_date"`
	Completed bool   `json:"completed"`
}

type ProjectReport struct {
	Name       string      `json:"name"`
	StartDate  string      `json:"start_date"`
	Budget     float64     `json:"budget"`
	LeadEmail  string      `json:"lead_email"`
	Tags       []string    `json:"tags"`
	Milestones []Milestone `json:"milestones"`
}

const (
	complexSchemaInstructions = "You are an entity extractor. Extract the project metadata, milestones, and assigned engineers from the prompt into the exact requested output schema."
	complexSchemaPrompt       = `Project 'Apollo' started on 2026-01-10 with budget 50000.50. Lead is Sarah (sarah@example.com). Milestones: 'Alpha' due 2026-03-01 (completed), 'Beta' due 2026-06-01 (pending). Assigned tags: ["infra", "core"].`
)

func hasComplexSchemaAPIKey(provider crux.Provider) bool {
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

func Test_ComplexStructuredOutputValidation(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		provider crux.Provider
		model    string
	}{
		{
			name:     "OpenAI_GPT4_1Mini",
			provider: crux.ProviderOpenAI,
			model:    crux.ChatModelGPT4_1Mini,
		},
		{
			name:     "Anthropic_ClaudeHaiku4_5",
			provider: crux.ProviderAnthropic,
			model:    crux.ClaudeHaiku4_5,
		},
		{
			name:     "Gemini_3_5FlashLite",
			provider: crux.ProviderGoogle,
			model:    crux.Gemini3_5FlashLite,
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

			if !hasComplexSchemaAPIKey(tc.provider) {
				t.Skipf("Skipping %s: API key for provider %s not set in environment", tc.name, tc.provider)
			}

			agent, err := crux.New(
				"entity-extractor",
				tc.model,
				crux.WithInstructions(complexSchemaInstructions),
				crux.WithOutputSchemaFrom[ProjectReport](),
				crux.WithMaxTurns(5),
			)
			if err != nil && strings.Contains(err.Error(), "API key for provider") {
				t.Skipf("Skipping %s: %v", tc.name, err)
			}
			require.NoError(t, err)

			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()

			sess, err := crux.NewSession(ctx, agent)
			require.NoError(t, err)

			var report ProjectReport
			err = sess.RunInto(ctx, complexSchemaPrompt, &report)
			require.NoError(t, err, "sess.RunInto should execute and decode response into ProjectReport struct")

			// Validate top-level scalar and numeric fields
			require.Equal(t, "Apollo", report.Name, "project name should match")
			require.Equal(t, "2026-01-10", report.StartDate, "start date should match")
			require.InDelta(t, 50000.50, report.Budget, 0.01, "budget float should decode accurately")
			require.Equal(t, "sarah@example.com", report.LeadEmail, "lead email should match")

			// Validate string slice
			require.Len(t, report.Tags, 2, "tags slice should contain exactly 2 tags")
			require.ElementsMatch(t, []string{"infra", "core"}, report.Tags, "tags should match expected values")

			// Validate nested struct slice
			require.Len(t, report.Milestones, 2, "milestones slice should contain exactly 2 milestones")

			milestoneMap := make(map[string]Milestone)
			for _, m := range report.Milestones {
				milestoneMap[m.Name] = m
			}

			alpha, hasAlpha := milestoneMap["Alpha"]
			require.True(t, hasAlpha, "expected milestone 'Alpha' to be present")
			require.Equal(t, "2026-03-01", alpha.DueDate, "Alpha due date should match")
			require.True(t, alpha.Completed, "Alpha should be completed")

			beta, hasBeta := milestoneMap["Beta"]
			require.True(t, hasBeta, "expected milestone 'Beta' to be present")
			require.Equal(t, "2026-06-01", beta.DueDate, "Beta due date should match")
			require.False(t, beta.Completed, "Beta should not be completed")

			// Verify session history integrity
			logs := sess.Logs()
			require.NotEmpty(t, logs, "session history should contain entries")
			var hasAssistantEntry bool
			for _, entry := range logs {
				if entry.Kind == crux.KindAssistant {
					hasAssistantEntry = true
					require.NotEmpty(t, entry.Text(), "assistant entry text should not be empty")
				}
			}
			require.True(t, hasAssistantEntry, "session history must record an assistant response entry")
		})
	}
}
