package crux

import (
	"context"
	"errors"
	"net/http"
)

// openrouterStep applies OpenRouter's capability restrictions before delegating
// to the shared Responses step.
func (a *Agent) openrouterStep(ctx context.Context, log []Entry, httpClient *http.Client, emit chunkSink) ([]Entry, error) {
	if a.searchOptions != nil {
		return nil, errors.New("native web search is not supported by this OpenRouter adapter")
	}
	return a.openAIstep(ctx, log, httpClient, emit)
}
