package crux

import "fmt"

var (
	ErrToolNotFound   = fmt.Errorf("tool not found")
	ErrApprovalNeeded = fmt.Errorf("approval needed")
)
