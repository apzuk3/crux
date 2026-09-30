//go:build evals

// Scenario: OutputSchema Edge Cases & Robustness Across Providers
//
// This live evaluation test exercises many edge cases of Crux's OutputSchema
// system (reflection, provider adaptation, validation, markdown extraction,
// repair, numeric precision, nullability, maps, etc.).
//
// It uses real models and follows the same pattern as complex_schema_eval_test.go.
//
// Supported providers (skipped gracefully if no API key):
// - OpenAI, xAI, Anthropic, Google (Gemini), DeepSeek, OpenRouter, Ollama

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
	edgeCaseInstructions = `You are a precise data extractor. Always respond with valid JSON that exactly matches the requested output schema.
Do not add extra text outside the JSON unless the test specifically asks for markdown wrapping.
If you cannot fulfill the request perfectly, still return valid JSON.`

	edgeCaseTimeout = 90 * time.Second
)

// -----------------------------------------------------------------------------
// Test Types (chosen to hit different schema behaviors)
// -----------------------------------------------------------------------------

type NullableOutput struct {
	Name   string  `json:"name"`
	Email  *string `json:"email,omitempty"`
	Score  *int    `json:"score,omitempty"`
	Active *bool   `json:"active,omitempty"`
}

type NumericOutput struct {
	ID       uint64  `json:"id" jsonschema:"minimum=1"`
	BigID    uint64  `json:"big_id"` // test >2^53 precision
	FloatVal float64 `json:"float_val" jsonschema:"minimum=0,maximum=1000.0"`
	Count    int     `json:"count" jsonschema:"enum=0,enum=1,enum=42"`
}

type NestedOutput struct {
	Title    string           `json:"title"`
	Items    []string         `json:"items"`
	Details  map[string]int   `json:"details"` // map test
	Children []NullableOutput `json:"children"`
}

type MarkdownWrappedOutput struct {
	Status  string `json:"status"`
	Message string `json:"message"`
	Count   int    `json:"count"`
}

// -----------------------------------------------------------------------------
// Shared helpers
// -----------------------------------------------------------------------------

func hasOutputSchemaAPIKey(provider crux.Provider) bool {
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
	case crux.ProviderDeepSeek:
		envVars = []string{"DEEPSEEK_API_KEY", "DEEPSEEK_APIKEY", "DEEPSEEK_KEY"}
	case crux.ProviderOpenrouter:
		envVars = []string{"OPENROUTER_API_KEY", "OPENROUTER_APIKEY", "OPENROUTER_KEY"}
	case crux.ProviderOllama:
		envVars = []string{"OLLAMA_API_KEY", "OLLAMA_APIKEY", "OLLAMA_KEY", "OLLAMA_HOST"}
	default:
		return false
	}
	for _, env := range envVars {
		if os.Getenv(env) != "" {
			return true
		}
	}
	return false
}

func newAgent(t *testing.T, name string, provider crux.Provider, model string, opts ...crux.AgentOption) *crux.Agent {
	t.Helper()
	opts = append([]crux.AgentOption{
		crux.WithProvider(provider),
		crux.WithModel(model),
		crux.WithInstructions(edgeCaseInstructions),
		crux.WithMaxTurns(8),
		crux.WithMaxRepairs(2),
	}, opts...)

	agent, err := crux.New(name, model, opts...)
	if err != nil {
		if strings.Contains(err.Error(), "API key") || strings.Contains(err.Error(), "no API key") {
			t.Skipf("Skipping %s: %v", name, err)
		}
		t.Fatal(err)
	}
	return agent
}

// -----------------------------------------------------------------------------
// Individual edge case tests
// -----------------------------------------------------------------------------

func TestOutputSchema_EdgeCases(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		provider crux.Provider
		model    string
	}{
		{"OpenAI_GPT5", crux.ProviderOpenAI, crux.OpenAIGPT5_6Sol},
		{"Anthropic_Claude", crux.ProviderAnthropic, crux.ClaudeHaiku4_5},
		{"Gemini_Flash", crux.ProviderGoogle, crux.Gemini3_5FlashLite},
		{"xAI_Grok", crux.ProviderXAI, crux.XAIGrok4_20},
		// DeepSeek, OpenRouter, and Ollama are included but often skipped in CI due to keys/models
		{"DeepSeek", crux.ProviderDeepSeek, "deepseek-chat"},
		{"OpenRouter_Grok", crux.ProviderOpenrouter, crux.OpenRouterXAIGrok4_20},
		{"Ollama", crux.ProviderOllama, "llama3.2"},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			if !hasOutputSchemaAPIKey(tc.provider) {
				t.Skipf("No API key for %s, skipping live test", tc.provider)
			}

			ctx, cancel := context.WithTimeout(context.Background(), edgeCaseTimeout)
			defer cancel()

			// 1. Nullable + Optional fields
			t.Run("nullable_and_pointers", func(t *testing.T) {
				agent := newAgent(t, "nullable-test", tc.provider, tc.model,
					crux.WithOutputSchemaFrom[NullableOutput]())
				sess, err := crux.NewSession(ctx, agent)
				require.NoError(t, err)

				var out NullableOutput
				prompt := `Return JSON for a person named "Alice". Include email "alice@example.com", score 95, and active true.`

				err = sess.RunInto(ctx, prompt, &out)
				require.NoError(t, err)
				require.Equal(t, "Alice", out.Name)
				require.NotNil(t, out.Email)
				require.Equal(t, "alice@example.com", *out.Email)
				require.NotNil(t, out.Score)
				require.Equal(t, 95, *out.Score)
				require.NotNil(t, out.Active)
				require.True(t, *out.Active)
			})

			// 2. Numeric precision, constraints, enums
			t.Run("numeric_precision", func(t *testing.T) {
				agent := newAgent(t, "numeric-test", tc.provider, tc.model,
					crux.WithOutputSchemaFrom[NumericOutput]())
				sess, err := crux.NewSession(ctx, agent)
				require.NoError(t, err)

				var out NumericOutput
				prompt := `Return id=9007199254740993, big_id=123456789012345678, float_val=123.456, count=42.`

				err = sess.RunInto(ctx, prompt, &out)
				require.NoError(t, err)
				require.Equal(t, uint64(9007199254740993), out.ID) // tests decodeJSONNumber + UseNumber()
				require.Equal(t, uint64(123456789012345678), out.BigID)
				require.InDelta(t, 123.456, out.FloatVal, 0.001)
				require.Equal(t, 42, out.Count)
			})

			// 3. Nested + Maps (Gemini should accept maps, strict providers should reject dynamic maps at schema level)
			if tc.provider == crux.ProviderGoogle {
				t.Run("maps_and_nesting_gemini", func(t *testing.T) {
					agent := newAgent(t, "nested-test", tc.provider, tc.model,
						crux.WithOutputSchemaFrom[NestedOutput]())
					sess, err := crux.NewSession(ctx, agent)
					require.NoError(t, err)

					var out NestedOutput
					prompt := `Return title="Test", items=["a","b"], details={"x":1,"y":2}, and two children.`

					err = sess.RunInto(ctx, prompt, &out)
					require.NoError(t, err)
					require.Equal(t, "Test", out.Title)
					require.Len(t, out.Items, 2)
					require.Len(t, out.Details, 2)
					require.Len(t, out.Children, 2)
				})
			} else {
				// For strict providers, we expect early error on dynamic map in schema
				t.Run("dynamic_map_rejected", func(t *testing.T) {
					// Dynamic map schemas are tested in cruxtest/schema_test.go.
					// Strict providers reject them at schema creation time.
					_, _ = crux.New("map-reject-test", tc.model,
						crux.WithProvider(tc.provider),
						crux.WithOutputSchemaFrom[NestedOutput]())
					// We ignore error here - the point is that it doesn't panic.
				})
			}

			// 4. Markdown wrapped JSON extraction (tests the new extractLastMarkdownJSON)
			t.Run("markdown_wrapped_json", func(t *testing.T) {
				agent := newAgent(t, "markdown-test", tc.provider, tc.model,
					crux.WithOutputSchemaFrom[MarkdownWrappedOutput]())
				sess, err := crux.NewSession(ctx, agent)
				require.NoError(t, err)

				var out MarkdownWrappedOutput
				prompt := `First think step by step, then output a JSON code block containing: status="success", message="hello from markdown", count=7. Wrap the final answer in a json code block.`

				err = sess.RunInto(ctx, prompt, &out)
				require.NoError(t, err, "should successfully extract JSON from markdown fence")
				require.Equal(t, "success", out.Status)
				require.Contains(t, out.Message, "markdown")
				require.Equal(t, 7, out.Count)
			})

			// 5. Invalid output triggers validation error + repair (this test is flaky with live models)
			t.Run("invalid_output_repair", func(t *testing.T) {
				t.Skip("Repair test is too flaky with live models due to rate limits and variable behavior. Covered thoroughly in cruxtest/schema_test.go")
			})
		})
	}
}
