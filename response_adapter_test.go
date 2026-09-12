package crux

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go/v3/responses"
)

func TestOpenRouterStepErrorTranslation(t *testing.T) {
	for _, tc := range []struct {
		name    string
		status  int
		body    string
		refusal bool
		failure bool
	}{
		{"http refusal", 403, `{"error_type":"refusal","error":{"code":"invalid_prompt","message":"Declined"}}`, true, true},
		{"generation refusal", 200, `{"status":"failed","error_type":"refusal","error":{"code":"invalid_prompt","message":"Declined"}}`, true, true},
		{"native refusal", 200, `{"status":"completed","output":[{"type":"message","content":[{"type":"refusal","refusal":"Declined"}]}]}`, true, true},
		{"validation", 400, `{"error":{"code":"invalid_prompt","message":"Invalid"}}`, false, true},
		{"generation failure", 200, `{"status":"failed","error_type":"authentication","error":{"code":"server_error","message":"Invalid credentials"}}`, false, true},
		{"success", 200, `{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"Hello"}]}]}`, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.status)
				fmt.Fprint(w, tc.body)
			}))
			defer server.Close()
			a := NewAgent("test-model", WithProvider(ProviderOpenrouter), WithAPIKey("test"), WithBaseURL(server.URL+"/"))
			text, err := a.Run(t.Context(), "input")
			_, refused := errors.AsType[*RefusalError](err)
			if refused != tc.refusal || (err != nil) != tc.failure {
				t.Fatalf("unexpected result: %q, %v", text, err)
			}
			if !tc.failure && text != "Hello" {
				t.Fatalf("unexpected output: %q", text)
			}
		})
	}
}

func TestOpenRouterWrappedResponseError(t *testing.T) {
	var response responses.Response
	if err := json.Unmarshal([]byte(`{"status":"failed","error_type":"refusal","error":{"code":"invalid_prompt","message":"Declined"}}`), &response); err != nil {
		t.Fatal(err)
	}
	a := &Agent{provider: ProviderOpenrouter, model: "test-model"}
	err := fmt.Errorf("wrapped: %w", &openAIResponseError{Provider: a.provider, Response: &response})
	refusal := a.openRouterRefusalFromError(err)
	if refusal == nil || refusal.Reason != "refusal" || refusal.Message != "Declined" {
		t.Fatalf("unexpected refusal: %v", refusal)
	}
}
