package crux

import "fmt"

// RefusalError reports an explicit model refusal or provider content-policy block.
// It does not classify unmarked natural-language responses as refusals.
// Use errors.AsType[*RefusalError] to find it through wrapped errors.
type RefusalError struct {
	Provider Provider
	Model    string // requested model
	Reason   string // provider-native code or category, when available
	Message  string // provider explanation, when available
}

func (e *RefusalError) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("%s refused the request: %s", e.Provider, e.Message)
	}
	if e.Reason != "" {
		return fmt.Sprintf("%s refused the request (%s)", e.Provider, e.Reason)
	}
	return fmt.Sprintf("%s refused the request", e.Provider)
}
