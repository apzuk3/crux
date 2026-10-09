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
	err := limitError(errors.New("429"), http.StatusTooManyRequests, header, []string{"rate_limit_error"}, "")
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
