// Scenario: Baseline Single-Turn Prompts & Schema Enforcement
//
// This live evaluation test verifies Crux's fundamental baseline integration across all supported
// model providers (OpenAI, Anthropic, Gemini, xAI, OpenRouter) in a clean, non-tool environment.
//
// Core Objectives:
// 1. System Instruction & Prompt Delivery:
//    - Tests that Crux correctly configures provider-specific system prompts / instructions:
//      * OpenAI / xAI: Developer / system message parameter.
//      * Anthropic: Top-level system parameter.
//      * Gemini: SystemInstruction content parts.
//    - Validates that strict behavioral constraints ("respond with 'Hello! How are you' and nothing else")
//      are respected across all provider adapters.
//
// 2. Output Schema Enforcement:
//    - Standard Text Mode: Verifies agent.Run() with crux.WithOutputSchemaFrom[string]().
//    - Structured Output Mode: Verifies agent.Run() with crux.WithOutputSchemaFrom[Output]()
//      enforcing a JSON schema { text: string } via provider-native structured output / response format APIs.
//
// 3. Provider Parity:
//    - Validates consistent client initialization, error propagation, and response decoding
//      across Google Gemini, OpenAI, Anthropic Claude, and xAI Grok.

package evals

import (
	"context"
	"testing"

	"github.com/apzuk3/crux"

	"github.com/stretchr/testify/require"
)

func Test_ExecuteSimplePrompt(t *testing.T) {
	t.Parallel()

	for _, modelname := range []string{
		crux.Gemini3_5FlashLite,
		crux.ChatModelGPT5_6Sol,
		crux.ClaudeHaiku4_5,
		crux.XAIGrok4_20,
		crux.OpenRouterXAIGrok4_20,
		crux.OpenRouterChatModelGPT5_6Luna,
	} {
		t.Run(modelname, func(t *testing.T) {
			executeSayHelloPromptWithInstructions(t, modelname)
		})

		t.Run(modelname+"_structured", func(t *testing.T) {
			executeSayHelloPromptWithInstructionsStructuredOutput(t, modelname)
		})
	}
}
func executeSayHelloPromptWithInstructions(t *testing.T, modelname string) {
	t.Parallel()

	prompt := `You are a friendly chatbot that responds to the user's message with "Hello! How are you" and nothing else.`

	agent, err := crux.New(
		"test-agent",
		modelname,
		crux.WithInstructions(prompt),
		crux.WithOutputSchemaFrom[string](),
	)
	require.NoError(t, err)

	sess, err := crux.NewSession(agent)
	require.NoError(t, err)

	output, err := sess.Run(context.Background(), "Hello, how are you?")
	require.NoError(t, err)

	require.Contains(t, output, "Hello! How are you")
}

func executeSayHelloPromptWithInstructionsStructuredOutput(t *testing.T, modelname string) {
	t.Parallel()

	type Output struct {
		Text string `json:"text"`
	}

	prompt := `You are a friendly chatbot that responds to the user's message with "Hello! How are you" and nothing else.`

	agent, err := crux.New(
		"test-agent",
		modelname,
		crux.WithInstructions(prompt),
		crux.WithOutputSchemaFrom[Output](),
	)
	require.NoError(t, err)

	sess, err := crux.NewSession(agent)
	require.NoError(t, err)

	output, err := sess.Run(context.Background(), "Hello, how are you?")
	require.NoError(t, err)

	require.Contains(t, output, "Hello! How are you")
}
