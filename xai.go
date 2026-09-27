package crux

import (
	"context"
	"net/http"
)

// xaiStep delegates to the shared Responses step, including native web search.
func (a *Agent) xaiStep(ctx context.Context, log []Entry, httpClient *http.Client, emit chunkSink) ([]Entry, error) {
	return a.openAIstep(ctx, log, httpClient, emit)
}
