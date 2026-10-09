package provider

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"
)

// LimitKind is the kind of limit a provider reported.
type LimitKind int

const (
	LimitRateLimited         LimitKind = iota + 1 // too many requests or tokens
	LimitOverloaded                               // no capacity for now
	LimitInsufficientCredits                      // no credit or quota left
)

// Retryable reports whether the same request may succeed later.
func (k LimitKind) Retryable() bool {
	return k == LimitRateLimited || k == LimitOverloaded
}

// LimitError is a provider saying a limit stopped the request. StatusCode is
// 0 when the error came in a response the server started with 200 (a stream
// event or a failed response); the SDKs retry only HTTP errors, so those are
// left to the caller.
type LimitError struct {
	Kind       LimitKind
	StatusCode int
	RetryAfter time.Duration // 0 when the provider gave none
	Err        error
}

func (e *LimitError) Error() string { return e.Err.Error() }
func (e *LimitError) Unwrap() error { return e.Err }

// Error codes and types providers use for each limit, lowercase. Credits come
// first: OpenAI reports insufficient_quota with status 429.
var (
	creditCodes = []string{
		"insufficient_quota",           // OpenAI
		"billing_hard_limit_reached",   // OpenAI
		"billing_not_active",           // OpenAI
		"billing_error",                // Anthropic
		"payment_required",             // OpenRouter
		"insufficient_balance",         // DeepSeek
		"insufficient_credits",         // OpenAI-compatible servers
		"insufficient_user_quota",      // OpenAI-compatible servers
		"exceeded_current_quota_error", // OpenAI-compatible servers
	}
	rateCodes = []string{
		"rate_limit_exceeded", // OpenAI, OpenRouter
		"rate_limit_error",    // Anthropic
		"resource_exhausted",  // Gemini
		"rate_limited",
		"too_many_requests",
	}
	overloadCodes = []string{
		"overloaded_error",     // Anthropic
		"server_is_overloaded", // OpenAI
		"slow_down",            // OpenAI
		"provider_overloaded",  // OpenRouter
		"provider_unavailable", // OpenRouter
		"unavailable",          // Gemini
		"service_unavailable",
		"engine_overloaded",
	}
	// Billing errors some providers send as a plain 400 or 403.
	creditMessages = []string{
		"credit balance is too low", // Anthropic
		"insufficient balance",      // DeepSeek
		"insufficient credits",
		"doesn't have any credits", // xAI
		"billing",                  // Gemini: billing not enabled
	}
)

// classifyLimit returns the limit that a status code (0 when unknown), the
// provider's error codes or types, and its message describe, or 0.
func classifyLimit(status int, codes []string, message string) LimitKind {
	has := func(list []string) bool {
		return slices.ContainsFunc(codes, func(code string) bool {
			return slices.Contains(list, strings.ToLower(strings.TrimSpace(code)))
		})
	}
	switch {
	case has(creditCodes):
		return LimitInsufficientCredits
	case has(rateCodes):
		return LimitRateLimited
	case has(overloadCodes):
		return LimitOverloaded
	}
	switch status {
	case http.StatusPaymentRequired:
		return LimitInsufficientCredits
	case http.StatusTooManyRequests:
		return LimitRateLimited
	case http.StatusBadGateway, http.StatusServiceUnavailable, 529:
		return LimitOverloaded
	case http.StatusBadRequest, http.StatusForbidden:
		message = strings.ToLower(message)
		for _, s := range creditMessages {
			if strings.Contains(message, s) {
				return LimitInsufficientCredits
			}
		}
	}
	return 0
}

// limitError returns err as a *LimitError when status, codes and message
// describe a limit, and err unchanged otherwise.
func limitError(err error, status int, header http.Header, codes []string, message string) error {
	kind := classifyLimit(status, codes, message)
	if kind == 0 {
		return err
	}
	return &LimitError{Kind: kind, StatusCode: status, RetryAfter: parseRetryAfter(header), Err: err}
}

// bodyLimit is limitError for an error that came in a response the server
// started with 200: raw is the error event or object, classified by its codes
// (and a number code as the status), and StatusCode stays 0.
func bodyLimit(err error, raw []byte, codes ...string) error {
	status, more, message := errorFields(raw)
	kind := classifyLimit(status, append(codes, more...), message)
	if kind == 0 {
		return err
	}
	return &LimitError{Kind: kind, Err: err}
}

// errorFields reads the codes and message of a provider error object, as
// OpenAI, Anthropic and OpenRouter shape them: code, type and, from
// OpenRouter, error_type (also in metadata). A number code is OpenRouter's
// HTTP status and is returned as status. raw may be the object itself or an
// envelope with it under "error".
func errorFields(raw []byte) (status int, codes []string, message string) {
	type object struct {
		Code      json.RawMessage `json:"code"`
		Type      string          `json:"type"`
		ErrorType string          `json:"error_type"`
		Message   string          `json:"message"`
		Metadata  struct {
			ErrorType string `json:"error_type"`
		} `json:"metadata"`
	}
	var envelope struct {
		object
		Error *object `json:"error"`
	}
	if json.Unmarshal(raw, &envelope) != nil {
		return 0, nil, ""
	}
	add := func(o *object) {
		var code string
		if json.Unmarshal(o.Code, &code) == nil {
			codes = append(codes, code)
		} else if n, err := strconv.Atoi(string(o.Code)); err == nil && status == 0 {
			status = n
		}
		codes = append(codes, o.Type, o.ErrorType, o.Metadata.ErrorType)
		if message == "" {
			message = o.Message
		}
	}
	if envelope.Error != nil {
		add(envelope.Error)
	}
	add(&envelope.object)
	return status, codes, message
}

// parseRetryAfter reads how long to wait from retry-after-ms (OpenAI) or
// Retry-After (seconds or an HTTP date). It returns 0 when neither is set.
func parseRetryAfter(header http.Header) time.Duration {
	if header == nil {
		return 0
	}
	if ms, err := strconv.ParseFloat(header.Get("Retry-After-Ms"), 64); err == nil && ms > 0 {
		return time.Duration(ms * float64(time.Millisecond))
	}
	value := header.Get("Retry-After")
	if seconds, err := strconv.ParseFloat(value, 64); err == nil && seconds > 0 {
		return time.Duration(seconds * float64(time.Second))
	}
	if at, err := http.ParseTime(value); err == nil {
		return max(time.Until(at), 0)
	}
	return 0
}

// RetryDelay is how long to wait before retry number attempt (from 0): the
// provider's hint up to a minute, else a backoff from half a second.
func RetryDelay(retryAfter time.Duration, attempt int) time.Duration {
	if retryAfter > 0 {
		return min(retryAfter, time.Minute)
	}
	return min(500*time.Millisecond<<attempt, 8*time.Second)
}

// Sleep waits for d or until ctx is done.
func Sleep(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
