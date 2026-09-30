package crux

import (
	"errors"

	"github.com/anthropics/anthropic-sdk-go"
)

type AnthropicModel = anthropic.Model

// Models newer than the SDK's constants.
const (
	ClaudeOpus5_5   AnthropicModel = "claude-opus-5-5"
	ClaudeSonnet5_5 AnthropicModel = "claude-sonnet-5-5"
)

const (
	ClaudeFable5_1           = anthropic.ModelClaudeFable5_1
	ClaudeMythos5_1          = anthropic.ModelClaudeMythos5_1
	ClaudeSonnet5            = anthropic.ModelClaudeSonnet5
	ClaudeFable5             = anthropic.ModelClaudeFable5
	ClaudeMythos5            = anthropic.ModelClaudeMythos5
	ClaudeOpus5              = anthropic.ModelClaudeOpus5
	ClaudeOpus4_8            = anthropic.ModelClaudeOpus4_8
	ClaudeOpus4_7            = anthropic.ModelClaudeOpus4_7
	ClaudeOpus4_6            = anthropic.ModelClaudeOpus4_6
	ClaudeSonnet4_6          = anthropic.ModelClaudeSonnet4_6
	ClaudeHaiku4_5           = anthropic.ModelClaudeHaiku4_5
	ClaudeHaiku4_5_20251001  = anthropic.ModelClaudeHaiku4_5_20251001
	ClaudeOpus4_5            = anthropic.ModelClaudeOpus4_5
	ClaudeOpus4_5_20251101   = anthropic.ModelClaudeOpus4_5_20251101
	ClaudeSonnet4_5          = anthropic.ModelClaudeSonnet4_5
	ClaudeSonnet4_5_20250929 = anthropic.ModelClaudeSonnet4_5_20250929
)

func init() {
	registerProvider(ProviderAnthropic, providerSpec{
		models: []model{
			{Name: ClaudeOpus5_5}, {Name: ClaudeSonnet5_5},
			{Name: ClaudeFable5_1}, {Name: ClaudeMythos5_1},
			{Name: ClaudeSonnet5}, {Name: ClaudeFable5},
			{Name: ClaudeMythos5}, {Name: ClaudeOpus5},
			{Name: ClaudeOpus4_8}, {Name: ClaudeOpus4_7},
			{Name: ClaudeOpus4_6},
			{Name: ClaudeSonnet4_6}, {Name: ClaudeHaiku4_5},
			{Name: ClaudeHaiku4_5_20251001}, {Name: ClaudeOpus4_5},
			{Name: ClaudeOpus4_5_20251101}, {Name: ClaudeSonnet4_5},
			{Name: ClaudeSonnet4_5_20250929},
		},
		// ANTHROPIC_AUTH_TOKEN is a bearer token, not an API key; the SDK's
		// default options send it as Authorization when no key is set.
		// https://github.com/anthropics/anthropic-sdk-go/blob/v1.72.0/client.go
		envVars: []string{"ANTHROPIC_API_KEY", "ANTHROPIC_APIKEY", "ANTHROPIC_KEY"},
		step:    (*Agent).anthropicStep,
		schema:  adaptAnthropic,
		prepare: prepareAnthropic,
	})
}

// prepareAnthropic rejects settings the Messages API refuses.
func prepareAnthropic(a *Agent) error {
	if a.temperature != nil && a.reasoning != "" && a.reasoning != ReasoningOff {
		return errors.New("anthropic does not accept a temperature while the model reasons; remove WithTemperature or use WithReasoning(ReasoningOff)")
	}
	return nil
}
