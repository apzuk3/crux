package crux

import (
	"fmt"
	"slices"

	"github.com/google/uuid"
)

// Fork creates a session with the full retained history and inherited
// configuration, then applies opts in order to the session's agent. See ForkFrom for copying rules.
// Fork must not run concurrently with writes to the session.
func (s *Session) Fork(opts ...AgentOption) (*Session, error) {
	return s.ForkFrom(len(s.logs), opts...)
}

// ForkFrom creates a session with logs[:from] and inherited configuration, then
// applies opts in order to the session's agent. from is an exclusive slice offset, not an Entry.Seq.
// Empty history is valid; a prefix with an unfinished local tool exchange or
// an unmatched tool result is not.
// Changing providers clears opaque data and removes provider-only tool events
// from the copied history. Portable entries remain in their original order.
//
// History and mutable search/tool-selection settings are copied. State values
// follow StateSnapshot's copying rules. Bound tools are inherited unless tool
// options are supplied, which resolve a fresh selection from the registry.
// The tools registry and output schema remain shared unless overridden.
// Fork must not run concurrently with writes to the session.
func (s *Session) ForkFrom(from int, opts ...AgentOption) (*Session, error) {
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
	if clonedAgent.Provider != s.agent.Provider {
		for i := range forkedLogs {
			forkedLogs[i].Opaque = nil
		}
		forkedLogs = slices.DeleteFunc(forkedLogs, func(entry Entry) bool {
			return entry.Kind == KindProviderTool
		})
	}

	return &Session{
		id:         uuid.New(),
		agent:      clonedAgent,
		logs:       forkedLogs,
		httpClient: s.httpClient,
	}, nil
}

func (a *Agent) clone(opts ...AgentOption) (*Agent, error) {
	allOpts := make([]AgentOption, 0, len(opts)+2)

	allOpts = append(allOpts, func(fork *Agent) error {
		fork.maxTurns = a.maxTurns
		fork.instructions = a.instructions
		fork.Provider = a.Provider
		fork.baseURL = a.baseURL
		fork.httpClient = a.httpClient
		fork.outputSchema = a.outputSchema
		fork.maxRepairs = a.maxRepairs
		fork.apiKey = a.apiKey
		fork.tools = slices.Clone(a.tools)
		fork.searchOptions = cloneSearchOptions(a.searchOptions)
		return nil
	})

	allOpts = append(allOpts, opts...)

	allOpts = append(allOpts, func(fork *Agent) error {
		if fork.Provider == a.Provider && fork.model != a.model {
			if inferred := inferProvider(fork.model); inferred != "" {
				fork.Provider = inferred
			}
		}

		if fork.Provider != a.Provider {
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
	if s.UserLocation != nil {
		loc := *s.UserLocation
		if s.UserLocation.Latitude != nil {
			lat := *s.UserLocation.Latitude
			loc.Latitude = &lat
		}
		if s.UserLocation.Longitude != nil {
			long := *s.UserLocation.Longitude
			loc.Longitude = &long
		}
		search.UserLocation = &loc
	}
	return &search
}

func validateForkHistory(entries []Entry) error {
	pending := make(map[string]bool)
	for _, entry := range entries {
		switch entry.Kind {
		case KindToolCall:
			if entry.ToolCall == nil || entry.ToolCall.ID == "" {
				return fmt.Errorf("tool call has no ID")
			}
			id := entry.ToolCall.ID
			if pending[id] {
				return fmt.Errorf("duplicate pending tool call %q", id)
			}
			pending[id] = true
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
		if e.Compaction != nil {
			value := *e.Compaction
			e.Compaction = &value
		}
		if e.Approval != nil {
			value := *e.Approval
			e.Approval = &value
		}
		if e.Usage != nil {
			value := *e.Usage
			e.Usage = &value
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
