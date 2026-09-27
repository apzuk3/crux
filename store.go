package crux

import (
	"context"

	"github.com/google/uuid"
)

// Store defines the persistence contract for agent session state and logs.
type Store interface {
	// Append writes newly produced entries for the given session.
	// It receives the agent blueprint to record or update session and agent metadata.
	Append(ctx context.Context, sessionID uuid.UUID, agent *Agent, entries ...Entry) error

	// Get retrieves all log entries for the session in sequential order.
	Get(ctx context.Context, sessionID uuid.UUID) ([]Entry, error)
}
