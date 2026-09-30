package crux

import (
	"context"
	"fmt"
	"slices"
)

// Fork creates a session with the full retained history and inherited
// configuration, then applies opts in order to the session's agent.
// Changing providers clears opaque data and removes provider-only tool events
// from the copied history. Portable entries remain in their original order.
//
// History and mutable search/tool-selection settings are copied. State values
// follow StateSnapshot's copying rules. Bound tools are inherited unless tool
// options are supplied, which resolve a fresh selection from the registry.
// The tools registry and output schema remain shared unless overridden.
// Token usage is not copied: the fork's Usage counts only its own requests.
// Fork must not run concurrently with writes to the session.
func (s *Session) Fork(ctx context.Context, opts ...AgentOption) (*Session, error) {
	return s.forkFrom(ctx, len(s.logs), opts...)
}

// forkFrom is Fork with only logs[:from] copied. from is an exclusive slice
// offset, not an Entry.Seq. Empty history is valid; a prefix with an
// unfinished local tool exchange or an unmatched tool result is not.
func (s *Session) forkFrom(ctx context.Context, from int, opts ...AgentOption) (*Session, error) {
	if from < 0 || from > len(s.logs) {
		return nil, fmt.Errorf("cannot fork at offset %d: must be between 0 and %d", from, len(s.logs))
	}
	if err := validateForkHistory(s.logs[:from]); err != nil {
		return nil, fmt.Errorf("cannot fork at offset %d: %w", from, err)
	}

	clonedAgent, err := s.agent.clone(opts...)
	if err != nil {
		return nil, err
	}

	forkedLogs := cloneEntries(s.logs[:from])
	for i := range forkedLogs {
		forkedLogs[i].Usage = nil
	}
	if clonedAgent.provider != s.agent.provider {
		for i := range forkedLogs {
			forkedLogs[i].Opaque = nil
		}
		forkedLogs = slices.DeleteFunc(forkedLogs, func(entry Entry) bool {
			return entry.Kind == KindProviderTool
		})
	}

	return NewSession(ctx, clonedAgent, WithSessionLogs(forkedLogs), WithStore(s.store))
}

func (a *Agent) clone(opts ...AgentOption) (*Agent, error) {
	allOpts := make([]AgentOption, 0, len(opts)+2)

	allOpts = append(allOpts, func(fork *Agent) error {
		fork.maxTurns = a.maxTurns
		fork.instructions = a.instructions
		fork.provider = "" // set after opts, so an explicit WithProvider is detectable
		fork.baseURL = a.baseURL
		fork.httpClient = a.httpClient
		fork.outputSchema = a.outputSchema
		fork.maxRepairs = a.maxRepairs
		fork.maxTokens = a.maxTokens
		fork.temperature = a.temperature
		fork.reasoning = a.reasoning
		fork.apiKey = a.apiKey
		fork.tools = slices.Clone(a.tools)
		fork.searchOptions = cloneSearchOptions(a.searchOptions)
		return nil
	})

	allOpts = append(allOpts, opts...)

	allOpts = append(allOpts, func(fork *Agent) error {
		if fork.provider == "" {
			fork.provider = a.provider
			if fork.model != a.model {
				if inferred := inferProvider(fork.model); inferred != "" {
					fork.provider = inferred
				}
			}
		}

		if fork.provider != a.provider {
			if fork.apiKey == a.apiKey {
				fork.apiKey = ""
			}
			if fork.baseURL == a.baseURL {
				fork.baseURL = ""
			}
		}
		return nil
	})

	return New(a.name, a.model, allOpts...)
}

func cloneSearchOptions(s *SearchOptions) *SearchOptions {
	if s == nil {
		return nil
	}
	search := *s
	search.UserLocation = cloneUserLocation(s.UserLocation)
	return &search
}

// cloneUserLocation copies the location with its coordinates, so the caller
// can change its own value afterwards.
func cloneUserLocation(l *UserLocation) *UserLocation {
	if l == nil {
		return nil
	}
	loc := *l
	if l.Latitude != nil {
		lat := *l.Latitude
		loc.Latitude = &lat
	}
	if l.Longitude != nil {
		long := *l.Longitude
		loc.Longitude = &long
	}
	return &loc
}

func validateForkHistory(entries []Entry) error {
	pending := make(map[string]bool)
	for _, entry := range entries {
		switch entry.Kind {
		case KindToolCall:
			if entry.ToolCall == nil || entry.ToolCall.ID == "" {
				return fmt.Errorf("tool call has no ID")
			}
			// A call repeated while still open is answered once, as in the run loop.
			pending[entry.ToolCall.ID] = true
		case KindToolResult:
			if entry.ToolResult == nil || !pending[entry.ToolResult.CallID] {
				return fmt.Errorf("tool result has no matching pending call")
			}
			delete(pending, entry.ToolResult.CallID)
		}
	}
	// Report in log order so errors are deterministic for parallel batches.
	for _, entry := range entries {
		if entry.Kind == KindToolCall && pending[entry.ToolCall.ID] {
			return fmt.Errorf("tool call %q has no result in retained history", entry.ToolCall.ID)
		}
	}
	return nil
}

func cloneEntries(entries []Entry) []Entry {
	result := slices.Clone(entries)
	for i := range result {
		e := &result[i]
		e.Content = slices.Clone(e.Content)
		if e.Reasoning != nil {
			value := *e.Reasoning
			e.Reasoning = &value
		}
		if e.ToolCall != nil {
			value := *e.ToolCall
			value.Args = slices.Clone(value.Args)
			e.ToolCall = &value
		}
		if e.ToolResult != nil {
			value := *e.ToolResult
			e.ToolResult = &value
		}
		if e.Delta != nil {
			value := *e.Delta
			value.Set = cloneState(value.Set)
			value.Delete = slices.Clone(value.Delete)
			e.Delta = &value
		}
		if e.Approval != nil {
			value := *e.Approval
			e.Approval = &value
		}
		if e.Usage != nil {
			value := *e.Usage
			e.Usage = &value
		}
		if e.Run != nil {
			value := *e.Run
			e.Run = &value
		}
		if e.Turn != nil {
			value := *e.Turn
			e.Turn = &value
		}
		if e.Response != nil {
			value := *e.Response
			e.Response = &value
		}
		if e.Opaque != nil {
			opaque := make(map[string][]byte, len(e.Opaque))
			for key, value := range e.Opaque {
				opaque[key] = slices.Clone(value)
			}
			e.Opaque = opaque
		}
	}
	return result
}
