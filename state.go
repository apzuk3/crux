package crux

import (
	"context"
	"encoding/json"
	"maps"
	"slices"
)

type stateContextKey struct{}

// StateSnapshot reconstructs the current state by replaying state deltas in log
// order. Set replaces whole values; Delete is applied after Set in each delta.
// An agent without state deltas returns an empty, non-nil map.
//
// Snapshots recursively copy JSON-like values (map[string]any and []any), byte
// slices, and json.RawMessage. Other values are copied by assignment; callers
// must treat other reference-bearing types as immutable. Values must be acyclic.
// StateSnapshot must not run concurrently with writes to the agent's logs.
func (a *Agent) StateSnapshot() map[string]any {
	state := make(map[string]any)
	for _, entry := range a.sessionLogs {
		if entry.Kind != KindStateDelta || entry.Delta == nil {
			continue
		}

		maps.Copy(state, entry.Delta.Set)

		for _, key := range entry.Delta.Delete {
			delete(state, key)
		}
	}
	return cloneState(state)
}

// ContextWithState attaches a snapshot of state to a child context. Copying
// follows StateSnapshot's value rules. A nil map is a present, nil snapshot.
func ContextWithState(ctx context.Context, state map[string]any) context.Context {
	return context.WithValue(ctx, stateContextKey{}, cloneState(state))
}

// StateFromContext retrieves a copy of the injected snapshot, using
// StateSnapshot's value rules. It returns (nil, false) when no state is present.
func StateFromContext(ctx context.Context) (map[string]any, bool) {
	state, ok := ctx.Value(stateContextKey{}).(map[string]any)
	if !ok {
		return nil, false
	}
	return cloneState(state), true
}

func cloneState(state map[string]any) map[string]any {
	if state == nil {
		return nil
	}
	copy := make(map[string]any, len(state))
	for key, value := range state {
		copy[key] = cloneStateValue(value)
	}
	return copy
}

func cloneStateValue(value any) any {
	switch value := value.(type) {
	case map[string]any:
		return cloneState(value)
	case []any:
		copy := slices.Clone(value)
		for i, item := range value {
			copy[i] = cloneStateValue(item)
		}
		return copy
	case []byte:
		return slices.Clone(value)
	case json.RawMessage:
		return slices.Clone(value)
	default:
		return value
	}
}
