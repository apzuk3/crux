package crux

import "errors"

var (
	ErrToolNotFound     = errors.New("tool not found")
	ErrApprovalNeeded   = errors.New("approval needed")
	ErrOutputValidation = errors.New("output validation failed")
	ErrSessionNotFound  = errors.New("session not found")
)
