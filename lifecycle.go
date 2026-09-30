package crux

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
)

// WithEntryHandler calls fn with every entry the session appends to its log,
// once the store has accepted it and in log order. It can be given more than
// once; handlers run in the order they were added, each with its own copy of
// the entry. Subagent sessions report to the same handlers; s is the session
// that appended the entry. Calls are serialised, so fn need not be safe for
// concurrent use, but it runs on the session's goroutine and must not call
// back into the session. History seeded with WithSessionLogs is not reported.
func WithEntryHandler(fn func(ctx context.Context, s *Session, e Entry)) SessionOption {
	return func(s *Session) error {
		if fn == nil {
			return errors.New("entry handler cannot be nil")
		}
		s.onEntry = append(s.onEntry, fn)
		return nil
	}
}

// turnInfo describes the agent a provider request is made with.
func (a *Agent) turnInfo() *TurnInfo {
	return &TurnInfo{
		AgentID:  a.id,
		Provider: a.provider,
		Model:    a.model,
	}
}

// hashTools hashes each tool's name, description and input schema.
func hashTools(tools []Tool) map[string]string {
	if len(tools) == 0 {
		return nil
	}
	hashes := make(map[string]string, len(tools))
	for _, tool := range tools {
		raw, _ := json.Marshal(map[string]any{
			"name":        tool.name,
			"description": tool.description,
			"schema":      tool.schema,
		})
		sum := sha256.Sum256(raw)
		hashes[tool.name] = hex.EncodeToString(sum[:])
	}
	return hashes
}
