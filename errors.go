package crux

import "errors"

var (
	// ErrToolNotFound is returned when an agent selects a tool that is not registered.
	ErrToolNotFound = errors.New("tool not found")
	// ErrApprovalNeeded is returned by Run when a tool call waits for Approve or Reject.
	ErrApprovalNeeded = errors.New("approval needed")
	// ErrOutputValidation wraps output that does not match the agent's output schema.
	ErrOutputValidation = errors.New("output validation failed")
	// ErrSessionNotFound is returned by a Store that has no entries for a session.
	ErrSessionNotFound = errors.New("session not found")
	// ErrSessionConflict is returned by a Store when another writer already
	// appended entries to the session. Load the session again with
	// WithSessionID and retry.
	ErrSessionConflict = errors.New("session was changed by another writer")
	// ErrMaxTurns is returned by Run when the agent used all its turns without a final answer.
	ErrMaxTurns = errors.New("max turns reached")
	// ErrRefused is returned by Run when the model refuses the request.
	ErrRefused = errors.New("model refused the request")
)
