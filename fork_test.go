package crux

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestForkCrossProviderInference(t *testing.T) {
	agent := &Agent{
		Provider: ProviderAnthropic,
		Model:    ClaudeHaiku4_5,
		apikey:   "anthropic-fake-key",
		sessionLogs: []Entry{
			{
				Kind:    KindAssistant,
				Content: []ContentPart{{Kind: ContentKindText, Text: "hello"}},
				Opaque:  map[string][]byte{"anthropic.message.content_block": []byte(`{}`)},
			},
		},
	}

	// Fork with WithModel pointing to an OpenAI model
	forked, err := agent.Fork(WithModel(ChatModelGPT4_1Mini))
	require.NoError(t, err)

	require.Equal(t, ProviderOpenAI, forked.Provider)
	require.Equal(t, ChatModelGPT4_1Mini, forked.Model)
	require.Len(t, forked.sessionLogs, 1)
	require.Nil(t, forked.sessionLogs[0].Opaque)
}
