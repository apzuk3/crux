package crux

import (
	"context"
	"errors"
	"iter"
)

// ChunkKind identifies readable content in a provider stream.
type ChunkKind string

const (
	ChunkText      ChunkKind = "text"
	ChunkReasoning ChunkKind = "reasoning"
)

// Chunk is provisional incremental content. Turn is one-based within this run.
// Reasoning contains only readable reasoning or summaries exposed by the provider.
type Chunk struct {
	Kind  ChunkKind
	Delta string
	Turn  int
}

type chunkSink func(Chunk) error

// Stream runs the conversation when iterated, yielding text and reasoning deltas.
// Each iterator is single-use. Breaking iteration cancels the active request and
// stops execution. Completed steps are retained; an interrupted step is not.
// Tools and approvals behave as in Run; use nil input to resume after approval.
// Deltas may include intermediate commentary and invalid output before a repair.
// After successful iteration, FinalOutput returns the authoritative final answer.
// Stream must not execute concurrently with other operations on the session.
func (s *Session) Stream(ctx context.Context, input any) iter.Seq2[Chunk, error] {
	used := false
	return func(yield func(Chunk, error) bool) {
		if used {
			yield(Chunk{}, errors.New("stream iterator already consumed"))
			return
		}
		used = true
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		stopped := false
		_, err := s.run(ctx, input, func(chunk Chunk) error {
			if !yield(chunk, nil) {
				stopped = true
				cancel()
				return ctx.Err()
			}
			return ctx.Err()
		})
		if err != nil && !stopped {
			yield(Chunk{}, err)
		}
	}
}

func emitChunk(emit chunkSink, kind ChunkKind, delta string) error {
	if delta == "" {
		return nil
	}
	return emit(Chunk{Kind: kind, Delta: delta})
}
