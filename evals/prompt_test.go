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

	output, err := agent.Run(context.Background(), "Hello, how are you?")
	require.NoError(t, err)

	require.Contains(t, output, "Hello! How are you")
}
