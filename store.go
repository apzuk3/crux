package crux

import (
	"context"
	"errors"
	"sync"

	"github.com/google/uuid"
)

// Store defines the persistence contract for agent session state and logs.
type Store interface {
	// Append writes newly produced entries for the given session.
	// It receives the agent blueprint to record or update session and agent metadata.
	Append(ctx context.Context, sessionID uuid.UUID, agent *Agent, entries ...Entry) error

	// Get retrieves all log entries for the session in sequential order.
	// It returns ErrSessionNotFound when nothing was ever appended for the session.
	Get(ctx context.Context, sessionID uuid.UUID) ([]Entry, error)
}

var _ Store = (*MemoryStore)(nil)

// MemoryStore keeps session logs in process memory. It is the default store
// for new sessions and is safe for concurrent use. Its contents are lost when
// the process exits; use a persistent store such as store/gormstore to keep them.
type MemoryStore struct {
	mu       sync.RWMutex
	sessions map[uuid.UUID][]Entry
}

// NewMemoryStore returns an empty in-memory store.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{sessions: make(map[uuid.UUID][]Entry)}
}

// Append records entries for the session, creating it on first use.
func (m *MemoryStore) Append(ctx context.Context, sessionID uuid.UUID, agent *Agent, entries ...Entry) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if agent == nil {
		return errors.New("agent cannot be nil")
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.sessions[sessionID] = append(m.sessions[sessionID], cloneEntries(entries)...)
	return nil
}

// Get returns a copy of the session's entries in the order they were appended.
func (m *MemoryStore) Get(ctx context.Context, sessionID uuid.UUID) ([]Entry, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	m.mu.RLock()
	defer m.mu.RUnlock()
	entries, ok := m.sessions[sessionID]
	if !ok {
		return nil, ErrSessionNotFound
	}
	return cloneEntries(entries), nil
}
