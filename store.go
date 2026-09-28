package crux

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/google/uuid"
)

// Store defines the persistence contract for agent session state and logs.
type Store interface {
	// Append writes newly produced entries for the session. The session also
	// gives access to its ID and agent for recording session metadata.
	// Append must be all-or-nothing. Entries join the session only after
	// Append succeeds, so after an error the session is unchanged and the
	// entries are produced again: tool calls whose results were not stored
	// run again on the next Run or Resume. Append must fail with
	// ErrSessionConflict when the first entry's Seq is not greater than every
	// Seq already stored, which means another writer got there first.
	Append(ctx context.Context, session *Session, entries ...Entry) error

	// Get retrieves all log entries for the session in sequential order.
	// It returns ErrSessionNotFound when nothing was ever appended for the session.
	Get(ctx context.Context, sessionID uuid.UUID) ([]Entry, error)
}

var _ Store = (*MemoryStore)(nil)

// MemoryStore keeps session logs in process memory. It is the default store
// for new sessions and is safe for concurrent use. Its contents are lost when
// the process exits; use a persistent store such as GORMStore to keep them.
type MemoryStore struct {
	mu       sync.RWMutex
	sessions map[uuid.UUID]*memorySession
}

type memorySession struct {
	parentID uuid.UUID
	entries  []Entry
}

// NewMemoryStore returns an empty in-memory store.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{sessions: make(map[uuid.UUID]*memorySession)}
}

// Append records entries for the session, creating it on first use.
func (m *MemoryStore) Append(ctx context.Context, session *Session, entries ...Entry) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if session == nil {
		return errors.New("session cannot be nil")
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	stored, ok := m.sessions[session.id]
	if !ok {
		stored = &memorySession{parentID: session.parentID}
		m.sessions[session.id] = stored
	}
	if n := len(stored.entries); n > 0 && len(entries) > 0 && entries[0].Seq <= stored.entries[n-1].Seq {
		return fmt.Errorf("%w: session %s already has entry %d", ErrSessionConflict, session.id, entries[0].Seq)
	}
	stored.entries = append(stored.entries, cloneEntries(entries)...)
	return nil
}

// Get returns a copy of the session's entries in the order they were appended.
func (m *MemoryStore) Get(ctx context.Context, sessionID uuid.UUID) ([]Entry, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	m.mu.RLock()
	defer m.mu.RUnlock()
	stored, ok := m.sessions[sessionID]
	if !ok {
		return nil, ErrSessionNotFound
	}
	return cloneEntries(stored.entries), nil
}
