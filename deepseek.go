package crux

import (
	"context"
	"errors"
	"net/http"
)

// deepseekStep uses stateless Responses with plain-text reasoning replay.
func (a *Agent) deepseekStep(ctx context.Context, log []Entry, httpClient *http.Client, emit chunkSink) ([]Entry, error) {
	if a.searchOptions != nil {
		return nil, errors.New("native web search is not supported by this DeepSeek adapter")
	}
	return a.openAIstep(ctx, log, httpClient, emit)
}
