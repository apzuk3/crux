package crux

import "context"

// xaiStep delegates to the shared Responses step, including native web search.
func (a *Agent) xaiStep(ctx context.Context, log []Entry) ([]Entry, error) {
	return a.openAIstep(ctx, log)
}
