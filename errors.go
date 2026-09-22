package crux

import "errors"

var (
	ErrToolNotFound   = errors.New("tool not found")
	ErrApprovalNeeded = errors.New("approval needed")
)
