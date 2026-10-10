package provider

import (
	"errors"
	"net/http"
	"testing"
	"time"

	"google.golang.org/genai"
)

func TestLimitErrorRetried(t *testing.T) {
	header := http.Header{"Retry-After": []string{"3"}}
	err := limitError(errors.New("429"), "anthropic", http.StatusTooManyRequests, header, []string{"rate_limit_error"}, "")
	var limit *LimitError
	if !errors.As(err, &limit) || !limit.Retried || limit.RetryAfter != 3*time.Second {
		t.Fatalf("HTTP limit from an SDK should be retried with Retry-After, got %#v", err)
	}

	err = bodyLimit(errors.New("stream"), []byte(`{"type":"error","error":{"type":"overloaded_error"}}`))
	if !errors.As(err, &limit) || limit.Retried {
		t.Fatalf("limit in a 200 body should not be retried, got %#v", err)
	}

	apiErr := genai.APIError{Code: http.StatusTooManyRequests, Status: "RESOURCE_EXHAUSTED", Details: []map[string]any{
		{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "30s"},
	}}
	err = geminiLimit(apiErr)
	if !errors.As(err, &limit) || limit.Retried || limit.RetryAfter != 30*time.Second || limit.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("Gemini HTTP limit should not be retried and carry RetryInfo, got %#v", err)
	}
}

func TestErrorFieldsStringError(t *testing.T) {
	status, codes, message := errorFields([]byte(`{"code":"permission-denied","error":"No credits left."}`))
	if status != 0 || message != "No credits left." {
		t.Fatalf("errorFields with a string error = %d, %v, %q", status, codes, message)
	}
	if len(codes) == 0 || codes[0] != "permission-denied" {
		t.Fatalf("top-level code should be kept, got %v", codes)
	}
}

func TestCreditMessages(t *testing.T) {
	const xaiCredits = "Your team abc has either used all available credits or reached its monthly spending limit."
	tests := []struct {
		name     string
		provider string
		status   int
		codes    []string
		message  string
		want     LimitKind
	}{
		{"xAI out of credits", "xai", http.StatusForbidden, []string{"permission-denied"}, xaiCredits, LimitInsufficientCredits},
		{"xAI never had credits", "xai", http.StatusForbidden, []string{"permission-denied"}, "Your team doesn't have any credits yet.", LimitInsufficientCredits},
		{"xAI credit text with another code", "xai", http.StatusForbidden, []string{"invalid-argument"}, xaiCredits, 0},
		{"xAI credit text on another status", "xai", http.StatusBadRequest, []string{"permission-denied"}, xaiCredits, 0},
		{"xAI credit text from another provider", "openai", http.StatusForbidden, []string{"permission-denied"}, xaiCredits, 0},
		{"Anthropic low balance", "anthropic", http.StatusBadRequest, []string{"invalid_request_error"}, "Your credit balance is too low to access the Anthropic API.", LimitInsufficientCredits},
		{"Gemini prepaid credits", "google", http.StatusTooManyRequests, []string{"RESOURCE_EXHAUSTED"}, "Your prepayment credits are depleted. Please go to AI Studio.", LimitInsufficientCredits},
		{"Gemini rate limit", "google", http.StatusTooManyRequests, []string{"RESOURCE_EXHAUSTED"}, "Resource has been exhausted (e.g. check quota).", LimitRateLimited},
		{"Gemini billing on another provider", "openai", http.StatusBadRequest, nil, "billing details are missing", 0},
		{"DeepSeek 402", "deepseek", http.StatusPaymentRequired, nil, "Insufficient Balance", LimitInsufficientCredits},
	}
	for _, tt := range tests {
		if got := classifyLimit(tt.provider, tt.status, tt.codes, tt.message); got != tt.want {
			t.Errorf("%s: classifyLimit = %v, want %v", tt.name, got, tt.want)
		}
	}

	apiErr := genai.APIError{Code: http.StatusForbidden, Status: "PERMISSION_DENIED", Message: "Permission denied.", Details: []map[string]any{
		{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "BILLING_DISABLED", "domain": "googleapis.com"},
	}}
	var limit *LimitError
	if err := geminiLimit(apiErr); !errors.As(err, &limit) || limit.Kind != LimitInsufficientCredits {
		t.Fatalf("Gemini BILLING_DISABLED should be insufficient credits, got %#v", err)
	}
}
