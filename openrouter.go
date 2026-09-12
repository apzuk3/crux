package crux

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"

	"github.com/openai/openai-go/v3"
)

// openrouterStep adds OpenRouter-specific error interpretation to the shared
// Responses step. Local tools are dispatched only after this step succeeds.
func (a *Agent) openrouterStep(ctx context.Context, log []Entry) ([]Entry, error) {
	if a.enableWebsearch {
		return nil, errors.New("native web search is not supported by this OpenRouter adapter")
	}
	entries, err := a.openAIstep(ctx, log)
	if err != nil {
		if refusal := a.openRouterRefusalFromError(err); refusal != nil {
			return nil, refusal
		}
	}
	return entries, err
}

// openRouterRefusalFromError handles HTTP errors and failed generation responses.
func (a *Agent) openRouterRefusalFromError(err error) *RefusalError {
	if refusal, ok := errors.AsType[*RefusalError](err); ok {
		return refusal
	}
	if responseErr, ok := errors.AsType[*openAIResponseError](err); ok {
		return a.openRouterRefusal([]byte(responseErr.Response.RawJSON()))
	}
	apiErr, ok := errors.AsType[*openai.Error](err)
	if !ok || apiErr.Response == nil || apiErr.Response.Body == nil {
		return nil
	}
	// The SDK decodes only the inner error. OpenRouter's canonical
	// error_type lives outside it, in the preserved response body.
	raw, readErr := io.ReadAll(apiErr.Response.Body)
	apiErr.Response.Body = io.NopCloser(bytes.NewReader(raw))
	if readErr != nil {
		return nil
	}
	return a.openRouterRefusal(raw)
}

// openRouterRefusal recognizes only documented policy codes, never HTTP status
// or message wording. The canonical type takes precedence over the lossy code.
func (a *Agent) openRouterRefusal(raw []byte) *RefusalError {
	var envelope struct {
		ErrorType string `json:"error_type"`
		Error     *struct {
			Code    json.RawMessage `json:"code"`
			Message string          `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &envelope) != nil || envelope.Error == nil {
		return nil
	}
	reason := envelope.ErrorType
	if reason == "" {
		var code string
		_ = json.Unmarshal(envelope.Error.Code, &code)
		if code == "image_content_policy_violation" {
			reason = code
		}
	}
	switch reason {
	case "refusal", "content_policy_violation", "image_content_policy_violation":
		return &RefusalError{Provider: a.provider, Model: a.model, Reason: reason, Message: envelope.Error.Message}
	default:
		return nil
	}
}
