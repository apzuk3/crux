package crux

import (
	"context"
	"errors"
)

// deepseekStep uses stateless Responses with plain-text reasoning replay.
func (a *Agent) deepseekStep(ctx context.Context, log []Entry) ([]Entry, error) {
	if a.SearchOptions != nil {
		return nil, errors.New("native web search is not supported by this DeepSeek adapter")
	}
	return a.openAIstep(ctx, log)
}
