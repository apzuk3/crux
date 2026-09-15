package crux

import (
	"fmt"
	"slices"
)

// Fork creates an agent with the full retained history and inherited
// configuration, then applies opts in order. See ForkFrom for copying rules.
// Fork must not run concurrently with writes to the agent.
func (a *Agent) Fork(opts ...AgentOption) (*Agent, error) {
	return a.ForkFrom(len(a.logs), opts...)
}

// ForkFrom creates an agent with logs[:from] and inherited configuration, then
// applies opts in order. from is an exclusive slice offset, not an Entry.Seq.
// Empty history is valid; a prefix with an unfinished local tool exchange or
// an unmatched tool result is not.
// Changing providers clears opaque data and removes provider-only tool events
// from the copied history. Portable entries remain in their original order.
//
// History and mutable search/tool-selection settings are copied. State values
// follow StateSnapshot's copying rules. Bound tools are inherited unless tool
// options are supplied, which resolve a fresh selection from the registry.
// The tools registry and output schema remain shared unless overridden.
// Fork must not run concurrently with writes to the agent.
func (a *Agent) ForkFrom(from int, opts ...AgentOption) (*Agent, error) {
	if from < 0 || from > len(a.logs) {
		return nil, fmt.Errorf("cannot fork at offset %d: must be between 0 and %d", from, len(a.logs))
	}
	if err := validateForkHistory(a.logs[:from]); err != nil {
		return nil, fmt.Errorf("cannot fork at offset %d: %w", from, err)
	}

	fork := *a
	fork.logs = cloneEntries(a.logs[:from])
	fork.allowedTools = slices.Clone(a.allowedTools)
	fork.tools = slices.Clone(a.tools)
	if a.searchOptions != nil {
		search := *a.searchOptions
		if search.UserLocation != nil {
			WithUserLocation(*search.UserLocation)(&search)
		}
		fork.searchOptions = &search
	}
	for _, opt := range opts {
		opt(&fork)
	}
	if fork.tools == nil {
		fork.tools = fork.toolsRegistry.selected(fork.allowedTools)
	}
	if fork.provider != a.provider {
		for i := range fork.logs {
			fork.logs[i].Opaque = nil
		}
		fork.logs = slices.DeleteFunc(fork.logs, func(entry Entry) bool {
			return entry.Kind == KindProviderTool
		})
	}
	return &fork, nil
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
