package crux

import (
	"context"
	"errors"
)

// ollamaStep requires Ollama v0.13.3+ for stateless Responses support.
// JSON Schema via text.format is verified in local Ollama v0.34.0;
// Ollama Cloud does not currently support structured outputs.
func (a *Agent) ollamaStep(ctx context.Context, log []Entry) ([]Entry, error) {
	if a.searchOptions != nil && a.apikey == "" {
		return nil, errors.New("Ollama API key is required for Ollama web search")
	}

	return a.openAIstep(ctx, log)
}
