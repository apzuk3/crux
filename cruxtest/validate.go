package cruxtest

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"

	"crux.foo"
)

// validateRequest checks a request body against rules the real provider APIs
// enforce, so a test fails on a request the provider would reject. It covers
// a few common mistakes, not the full API contract.
func validateRequest(provider crux.Provider, body []byte) error {
	if len(body) == 0 {
		return nil
	}
	switch provider {
	case providerDecisions:
		return validateDecisionsRequest(body)
	case crux.ProviderAnthropic:
		return validateAnthropicRequest(body)
	case crux.ProviderGoogle:
		return validateGeminiRequest(body)
	default:
		return validateOpenAIRequest(body)
	}
}

// invalidRequestResponse encodes a validation failure as an HTTP 400 in the
// provider's error format.
func invalidRequestResponse(provider crux.Provider, req *http.Request, err error) *http.Response {
	message := "cruxtest: invalid request: " + err.Error()
	var payload any
	switch provider {
	case providerDecisions:
		payload = map[string]any{"detail": []map[string]any{{"loc": []string{"body"}, "msg": message, "type": "value_error"}}}
	case crux.ProviderAnthropic:
		payload = map[string]any{"type": "error", "error": map[string]any{"type": "invalid_request_error", "message": message}}
	case crux.ProviderGoogle:
		payload = map[string]any{"error": map[string]any{"code": http.StatusBadRequest, "message": message, "status": "INVALID_ARGUMENT"}}
	default:
		payload = map[string]any{"error": map[string]any{"message": message, "type": "invalid_request_error", "code": "invalid_request"}}
	}
	body, _ := json.Marshal(payload)
	header := make(http.Header)
	header.Set("Content-Type", "application/json")
	return &http.Response{
		StatusCode:    http.StatusBadRequest,
		Status:        fmt.Sprintf("%d %s", http.StatusBadRequest, http.StatusText(http.StatusBadRequest)),
		Header:        header,
		Body:          io.NopCloser(bytes.NewReader(body)),
		ContentLength: int64(len(body)),
		Request:       req,
	}
}

type anthropicBlock struct {
	Type      string `json:"type"`
	Text      string `json:"text"`
	ID        string `json:"id"`
	ToolUseID string `json:"tool_use_id"`
}

func validateAnthropicRequest(body []byte) error {
	var request struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		return fmt.Errorf("decode anthropic request: %w", err)
	}
	if len(request.Messages) == 0 {
		return errors.New("anthropic: messages must not be empty")
	}
	// https://platform.claude.com/docs/en/build-with-claude/prompt-caching
	if n := bytes.Count(body, []byte(`"cache_control":`)); n > 4 {
		return fmt.Errorf("anthropic: a maximum of 4 blocks with cache_control may be provided, found %d", n)
	}
	var pending []string // tool_use ids awaiting results in the next message
	for i, message := range request.Messages {
		blocks, err := decodeAnthropicContent(i, message.Content)
		if err != nil {
			return err
		}
		uses, results, err := scanAnthropicBlocks(i, message.Role, blocks)
		if err != nil {
			return err
		}
		if err := checkToolPairing(i, pending, results); err != nil {
			return err
		}
		pending = uses
	}
	return nil
}

// decodeAnthropicContent decodes a message's content, which is either a
// string or a list of blocks.
func decodeAnthropicContent(i int, content json.RawMessage) ([]anthropicBlock, error) {
	var text string
	if err := json.Unmarshal(content, &text); err == nil {
		return []anthropicBlock{{Type: "text", Text: text}}, nil
	}
	var blocks []anthropicBlock
	if err := json.Unmarshal(content, &blocks); err != nil {
		return nil, fmt.Errorf("anthropic: messages.%d: decode content: %w", i, err)
	}
	if len(blocks) == 0 {
		return nil, fmt.Errorf("anthropic: messages.%d: content must not be empty", i)
	}
	return blocks, nil
}

// scanAnthropicBlocks checks the blocks of a message and returns the ids of
// its tool_use blocks and of the tool calls its tool_result blocks answer.
func scanAnthropicBlocks(i int, role string, blocks []anthropicBlock) (uses []string, results map[string]bool, err error) {
	results = make(map[string]bool)
	for j, b := range blocks {
		switch b.Type {
		case "text":
			if strings.TrimSpace(b.Text) == "" {
				return nil, nil, fmt.Errorf("anthropic: messages.%d.content.%d: text content blocks must contain non-whitespace text", i, j)
			}
		case "tool_use":
			uses = append(uses, b.ID)
		case "tool_result":
			if role != "user" {
				return nil, nil, fmt.Errorf("anthropic: messages.%d.content.%d: tool_result blocks belong in user messages", i, j)
			}
			if j > 0 && blocks[j-1].Type != "tool_result" {
				return nil, nil, fmt.Errorf("anthropic: messages.%d.content.%d: tool_result blocks must come first in a user message", i, j)
			}
			results[b.ToolUseID] = true
		}
	}
	return uses, results, nil
}

// checkToolPairing checks that the tool results of message i answer exactly
// the tool calls pending from the previous message.
func checkToolPairing(i int, pending []string, results map[string]bool) error {
	for id := range results {
		if !slices.Contains(pending, id) {
			return fmt.Errorf("anthropic: messages.%d: tool_result for %q has no tool_use in the previous message", i, id)
		}
	}
	for _, id := range pending {
		if !results[id] {
			return fmt.Errorf("anthropic: messages.%d: tool_use %q has no tool_result immediately after it", i, id)
		}
	}
	return nil
}

// geminiDataFields are the Part fields that carry data; a part must set one.
// https://ai.google.dev/api/caching#Part
var geminiDataFields = []string{
	"text", "inlineData", "fileData", "functionCall", "functionResponse",
	"executableCode", "codeExecutionResult", "toolCall", "toolResponse",
}

func validateGeminiRequest(body []byte) error {
	var request struct {
		Contents []struct {
			Role  string                       `json:"role"`
			Parts []map[string]json.RawMessage `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		return fmt.Errorf("decode gemini request: %w", err)
	}
	if len(request.Contents) == 0 {
		return errors.New("gemini: contents must not be empty")
	}
	for i, content := range request.Contents {
		if len(content.Parts) == 0 {
			return fmt.Errorf("gemini: contents.%d: parts must not be empty", i)
		}
		for j, part := range content.Parts {
			if !geminiPartHasData(part) {
				return fmt.Errorf("gemini: contents.%d.parts.%d: part must carry data (such as text or functionCall), not only metadata", i, j)
			}
		}
	}
	return nil
}

func geminiPartHasData(part map[string]json.RawMessage) bool {
	return slices.ContainsFunc(geminiDataFields, func(field string) bool {
		value, ok := part[field]
		return ok && string(value) != "null" && string(value) != `""`
	})
}

func validateOpenAIRequest(body []byte) error {
	var request struct {
		Input json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		return fmt.Errorf("decode openai request: %w", err)
	}
	var items []struct {
		Type   string `json:"type"`
		CallID string `json:"call_id"`
	}
	if err := json.Unmarshal(request.Input, &items); err != nil {
		return nil // plain string input
	}
	calls := make(map[string]bool)
	for i, item := range items {
		switch item.Type {
		case "function_call":
			calls[item.CallID] = true
		case "function_call_output":
			if !calls[item.CallID] {
				return fmt.Errorf("openai: input.%d: no tool call found for function call output with call_id %q", i, item.CallID)
			}
		}
	}
	return nil
}
