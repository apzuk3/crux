package crux

import (
	"context"
	"errors"
)

// openrouterStep applies OpenRouter's capability restrictions before delegating
// to the shared Responses step.
func (a *Agent) openrouterStep(ctx context.Context, log []Entry) ([]Entry, error) {
	if a.enableWebsearch {
		return nil, errors.New("native web search is not supported by this OpenRouter adapter")
	}
	return a.openAIstep(ctx, log)
}
