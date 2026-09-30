package crux

import (
	"testing"

	"github.com/invopop/jsonschema"
	"github.com/stretchr/testify/require"
)

func TestForkCrossProviderInference(t *testing.T) {
	agent := &Agent{
		provider: ProviderAnthropic,
		model:    ClaudeHaiku4_5,
		apiKey:   "anthropic-fake-key",
	}
	session, err := NewSession(t.Context(), agent, WithSessionLogs([]Entry{
		{
			Kind:    KindAssistant,
			Content: []ContentPart{{Kind: ContentKindText, Text: "hello"}},
			Opaque:  map[string][]byte{"anthropic.message.content_block": []byte(`{}`)},
		},
	}))
	require.NoError(t, err)

	// Fork with WithModel pointing to an OpenAI model
	forked, err := session.Fork(t.Context(), WithModel(OpenAIGPT4_1Mini))
	require.NoError(t, err)

	require.Equal(t, ProviderOpenAI, forked.agent.provider)
	require.Equal(t, OpenAIGPT4_1Mini, forked.agent.model)
	require.Len(t, forked.logs, 1)
	require.Nil(t, forked.logs[0].Opaque)
}

func TestForkInheritsConfiguration(t *testing.T) {
	schema := &jsonschema.Schema{Type: "object"}
	lat := 37.7749
	lon := -122.4194

	agent := &Agent{
		name:         "parent-agent",
		model:        OpenAIGPT4_1Mini,
		provider:     ProviderOpenAI,
		maxTurns:     42,
		instructions: "be helpful",
		outputSchema: schema,
		maxRepairs:   3,
		apiKey:       "secret-openai-key",
		tools:        []Tool{{name: "custom_tool"}},
		searchOptions: &SearchOptions{
			UserLocation: &UserLocation{
				City:      "San Francisco",
				Latitude:  &lat,
				Longitude: &lon,
			},
		},
	}
	session, err := NewSession(t.Context(), agent, WithSessionLogs([]Entry{
		{
			Kind:    KindUser,
			Content: []ContentPart{{Kind: ContentKindText, Text: "ping"}},
		},
	}))
	require.NoError(t, err)

	forked, err := session.Fork(t.Context())
	require.NoError(t, err)

	require.Equal(t, agent.name, forked.agent.name)
	require.Equal(t, agent.model, forked.agent.model)
	require.Equal(t, agent.provider, forked.agent.provider)
	require.Equal(t, agent.maxTurns, forked.agent.maxTurns)
	require.Equal(t, agent.instructions, forked.agent.instructions)
	require.Equal(t, agent.outputSchema, forked.agent.outputSchema)
	require.Equal(t, agent.maxRepairs, forked.agent.maxRepairs)
	require.Equal(t, agent.apiKey, forked.agent.apiKey)
	require.Len(t, forked.agent.tools, 1)
	require.Equal(t, "custom_tool", forked.agent.tools[0].name)
	require.NotNil(t, forked.agent.searchOptions)
	require.NotNil(t, forked.agent.searchOptions.UserLocation)
	require.Equal(t, "San Francisco", forked.agent.searchOptions.UserLocation.City)
	// Mutating SearchOptions in child must not affect parent
	forked.agent.searchOptions.UserLocation.City = "New York"
	require.Equal(t, "San Francisco", agent.searchOptions.UserLocation.City)
	require.Len(t, forked.logs, 1)
}

func TestForkCrossProviderDefaultBaseURL(t *testing.T) {
	agent := &Agent{
		provider: ProviderAnthropic,
		model:    ClaudeHaiku4_5,
		apiKey:   "anthropic-fake-key",
	}
	session, err := NewSession(t.Context(), agent)
	require.NoError(t, err)

	forked, err := session.Fork(
		t.Context(),
		WithProvider(ProviderXAI),
		WithModel("grok-2"),
	)
	require.NoError(t, err)

	require.Equal(t, ProviderXAI, forked.agent.provider)
	require.Equal(t, "https://api.x.ai/v1", forked.agent.baseURL)
}

func TestForkUserOverrides(t *testing.T) {
	agent := &Agent{
		name:         "parent",
		model:        OpenAIGPT4_1Mini,
		provider:     ProviderOpenAI,
		instructions: "old instructions",
		maxTurns:     5,
		baseURL:      "https://custom-proxy.internal",
		apiKey:       "old-key",
	}
	session, err := NewSession(t.Context(), agent)
	require.NoError(t, err)

	forked, err := session.Fork(
		t.Context(),
		WithInstructions("new instructions"),
		WithMaxTurns(20),
		WithBaseURL("https://override.internal"),
		WithAPIKey("override-key"),
	)
	require.NoError(t, err)

	require.Equal(t, "new instructions", forked.agent.instructions)
	require.Equal(t, 20, forked.agent.maxTurns)
	require.Equal(t, "https://override.internal", forked.agent.baseURL)
	require.Equal(t, "override-key", forked.agent.apiKey)
}

func TestForkFromOffsetAndSanitization(t *testing.T) {
	agent := &Agent{
		provider: ProviderAnthropic,
		model:    ClaudeHaiku4_5,
		apiKey:   "anthropic-fake-key",
	}
	session, err := NewSession(t.Context(), agent, WithSessionLogs([]Entry{
		{Kind: KindUser, Content: []ContentPart{{Kind: ContentKindText, Text: "1"}}},
		{Kind: KindAssistant, Content: []ContentPart{{Kind: ContentKindText, Text: "2"}}},
		{Kind: KindProviderTool, Opaque: map[string][]byte{"tool": []byte("p")}},
		{Kind: KindUser, Content: []ContentPart{{Kind: ContentKindText, Text: "3"}}},
	}))
	require.NoError(t, err)

	// Offset out of bounds
	_, err = session.forkFrom(t.Context(), -1)
	require.Error(t, err)
	_, err = session.forkFrom(t.Context(), 5)
	require.Error(t, err)

	// Fork at offset 3 with cross-provider
	forked, err := session.forkFrom(t.Context(), 3, WithModel(OpenAIGPT4_1Mini))
	require.NoError(t, err)
	// Entry at offset 2 was KindProviderTool, which should be deleted across providers
	require.Len(t, forked.logs, 2)
	require.Equal(t, KindUser, forked.logs[0].Kind)
	require.Equal(t, KindAssistant, forked.logs[1].Kind)
}

func TestForkKeepsExplicitSameProvider(t *testing.T) {
	agent := Must(New("local", "llama3.2", WithProvider(ProviderOllama), WithBaseURL("http://gpu-box:11434/v1")))
	session, err := NewSession(t.Context(), agent)
	require.NoError(t, err)

	forked, err := session.Fork(t.Context(), WithModel(ClaudeHaiku4_5), WithProvider(ProviderOllama))
	require.NoError(t, err)
	require.Equal(t, ProviderOllama, forked.agent.provider)
	require.Equal(t, "http://gpu-box:11434/v1", forked.agent.baseURL)

	inferred, err := session.Fork(t.Context(), WithModel(ClaudeHaiku4_5), WithAPIKey("k"))
	require.NoError(t, err)
	require.Equal(t, ProviderAnthropic, inferred.agent.provider)
	require.Empty(t, inferred.agent.baseURL)
}

func TestForkDropsUsage(t *testing.T) {
	agent := Must(New("usage", OpenAIGPT4_1Mini, WithAPIKey("k")))
	session, err := NewSession(t.Context(), agent, WithSessionLogs([]Entry{
		{Kind: KindUser, Content: []ContentPart{{Kind: ContentKindText, Text: "hi"}}},
		{Kind: KindAssistant, Content: []ContentPart{{Kind: ContentKindText, Text: "hello"}},
			Usage: &Usage{InputTokens: 10, OutputTokens: 5}},
	}))
	require.NoError(t, err)

	forked, err := session.Fork(t.Context())
	require.NoError(t, err)
	require.Equal(t, Usage{}, forked.Usage())
	require.Len(t, forked.logs, 2)
	require.Equal(t, 10, session.Usage().InputTokens)
}

func TestForkAcceptsRepeatedOpenCallID(t *testing.T) {
	agent := Must(New("dup", OpenAIGPT4_1Mini, WithAPIKey("k")))
	call := &ToolCall{ID: "call_1", Name: "noop", Args: []byte(`{}`)}
	logs := []Entry{
		{Kind: KindUser, Content: []ContentPart{{Kind: ContentKindText, Text: "go"}}},
		{Kind: KindToolCall, ToolCall: call},
		{Kind: KindToolCall, ToolCall: call},
		{Kind: KindToolResult, ToolResult: &ToolResult{CallID: "call_1", Output: "ok"}},
		{Kind: KindAssistant, Content: []ContentPart{{Kind: ContentKindText, Text: "done"}}},
	}
	session, err := NewSession(t.Context(), agent, WithSessionLogs(logs))
	require.NoError(t, err)
	require.False(t, session.hasUnexecutedToolCalls())

	forked, err := session.Fork(t.Context())
	require.NoError(t, err)
	require.Len(t, forked.logs, len(logs))

	_, err = session.forkFrom(t.Context(), 3)
	require.ErrorContains(t, err, `tool call "call_1" has no result`)
}
