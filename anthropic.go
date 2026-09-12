package crux

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

const anthropicContentBlockOpaqueKey = "anthropic.message.content_block"

func (a *Agent) newAnthropicClient() *anthropic.Client {
	var opts []option.RequestOption
	if a.apikey != "" {
		opts = append(opts, option.WithAPIKey(a.apikey))
	}

	if a.baseURL != "" {
		opts = append(opts, option.WithBaseURL(a.baseURL))
	}

	client := anthropic.NewClient(opts...)
	return &client
}

// anthropicStep returns the model's ordered blocks with usage attached.
func (a *Agent) anthropicStep(ctx context.Context, log []Entry) ([]Entry, error) {
	messages, err := toAnthropicMessages(log)
	if err != nil {
		return nil, err
	}
	tools, err := anthropicTools(a.toolsRegistry, a.allowedTools)
	if err != nil {
		return nil, err
	}
	params := anthropic.MessageNewParams{
		Model: a.model, Messages: messages, Tools: tools,
		MaxTokens: 12000,
	}
	if a.instructions != "" {
		params.System = []anthropic.TextBlockParam{{Text: a.instructions}}
	}
	if a.outputSchema != nil {
		raw, err := json.Marshal(a.outputSchema)
		if err != nil {
			return nil, fmt.Errorf("marshal output schema: %w", err)
		}
		var schema map[string]any
		if err := json.Unmarshal(raw, &schema); err != nil {
			return nil, fmt.Errorf("decode output schema: %w", err)
		}
		params.OutputConfig = anthropic.OutputConfigParam{
			Format: anthropic.JSONOutputFormatParam{Schema: schema},
		}
	}
	response, err := a.anthropic.Messages.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("anthropic messages: %w", err)
	}
	switch response.StopReason {
	case anthropic.StopReasonEndTurn, anthropic.StopReasonStopSequence, anthropic.StopReasonToolUse:
	case anthropic.StopReasonRefusal:
		return nil, errors.New("anthropic refused the request")
	default:
		return nil, fmt.Errorf("anthropic response did not complete: stop reason %q", response.StopReason)
	}
	produced := make([]Entry, 0, len(response.Content))
	for _, block := range response.Content {
		entry, err := fromAnthropicContentBlock(block)
		if err != nil {
			return nil, err
		}
		produced = append(produced, entry)
	}
	if len(produced) == 0 {
		return nil, errors.New("anthropic returned no content")
	}
	produced[len(produced)-1].Usage = &Usage{
		InputTokens:  int(response.Usage.InputTokens + response.Usage.CacheCreationInputTokens + response.Usage.CacheReadInputTokens),
		OutputTokens: int(response.Usage.OutputTokens),
	}
	return produced, nil
}

func anthropicTools(registry *ToolsRegistry, allowedTools []string) ([]anthropic.ToolUnionParam, error) {
	selected := registry.selected(allowedTools)
	tools := make([]anthropic.ToolUnionParam, 0, len(selected))
	for _, tool := range selected {
		if tool.schema["type"] != "object" {
			return nil, fmt.Errorf("anthropic tool %q requires an object input schema", tool.name)
		}
		raw, err := json.Marshal(tool.schema)
		if err != nil {
			return nil, fmt.Errorf("marshal tool %q schema: %w", tool.name, err)
		}
		var schema anthropic.ToolInputSchemaParam
		if err := json.Unmarshal(raw, &schema); err != nil {
			return nil, fmt.Errorf("decode tool %q schema: %w", tool.name, err)
		}
		// The SDK's param decoder does not retain unknown schema keywords.
		schema.ExtraFields = make(map[string]any)
		for key, value := range tool.schema {
			if key != "type" && key != "properties" && key != "required" {
				schema.ExtraFields[key] = value
			}
		}
		tools = append(tools, anthropic.ToolUnionParam{OfTool: &anthropic.ToolParam{
			Name: tool.name, Description: anthropic.String(tool.description), InputSchema: schema,
		}})
	}
	return tools, nil
}

func fromAnthropicContentBlock(block anthropic.ContentBlockUnion) (Entry, error) {
	opaque := map[string][]byte{anthropicContentBlockOpaqueKey: []byte(block.RawJSON())}
	switch v := block.AsAny().(type) {
	case anthropic.TextBlock:
		return Entry{Kind: KindAssistant, Content: []ContentPart{{Kind: ContentKindText, Text: v.Text}}, Opaque: opaque}, nil
	case anthropic.ThinkingBlock:
		return Entry{Kind: KindReasoning, Reasoning: &Reasoning{Summary: v.Thinking}, Opaque: opaque}, nil
	case anthropic.RedactedThinkingBlock:
		return Entry{Kind: KindReasoning, Reasoning: &Reasoning{}, Opaque: opaque}, nil
	case anthropic.ToolUseBlock:
		return Entry{Kind: KindToolCall, ToolCall: &ToolCall{
			ID: v.ID, Name: v.Name, Args: v.Input,
		}, Opaque: opaque}, nil
	default:
		return Entry{}, fmt.Errorf("unsupported Anthropic content block %q", block.Type)
	}
}

// toAnthropicMessages groups consecutive blocks by role.
func toAnthropicMessages(log []Entry) ([]anthropic.MessageParam, error) {
	var messages []anthropic.MessageParam
	appendBlocks := func(role anthropic.MessageParamRole, blocks ...anthropic.ContentBlockParamUnion) {
		if len(blocks) == 0 {
			return
		}
		if len(messages) == 0 || messages[len(messages)-1].Role != role {
			messages = append(messages, anthropic.MessageParam{Role: role})
		}
		last := &messages[len(messages)-1]
		last.Content = append(last.Content, blocks...)
	}
	for _, e := range log {
		if e.Kind == KindStateDelta {
			continue
		}
		if e.Kind == KindReasoning && len(e.Opaque[anthropicContentBlockOpaqueKey]) == 0 {
			continue
		}
		role := anthropic.MessageParamRoleAssistant
		if e.Kind == KindUser || e.Kind == KindToolResult {
			role = anthropic.MessageParamRoleUser
		}
		blocks, err := toAnthropicContentBlockParamUnion(e)
		if err != nil {
			return nil, err
		}
		appendBlocks(role, blocks...)
	}
	return messages, nil
}

// toAnthropicContentBlockParamUnion renders an entry as content blocks.
func toAnthropicContentBlockParamUnion(e Entry) ([]anthropic.ContentBlockParamUnion, error) {
	if raw := e.Opaque[anthropicContentBlockOpaqueKey]; len(raw) > 0 && e.Kind != KindUser && e.Kind != KindToolResult {
		var block anthropic.ContentBlockUnion
		if err := json.Unmarshal(raw, &block); err != nil {
			return nil, fmt.Errorf("decode Anthropic content block: %w", err)
		}
		return []anthropic.ContentBlockParamUnion{block.ToParam()}, nil
	}

	var blocks []anthropic.ContentBlockParamUnion
	switch e.Kind {
	case KindUser, KindAssistant:
		for _, part := range e.Content {
			if part.Kind != ContentKindText && part.Kind != ContentKindRefusal {
				return nil, fmt.Errorf("unsupported content part kind %q", part.Kind)
			}
			blocks = append(blocks, anthropic.NewTextBlock(part.Text))
		}
	case KindReasoning:
		// Reasoning without signed provider data cannot be replayed.
	case KindToolCall:
		c := e.ToolCall
		if c == nil {
			return nil, errors.New("invalid or unsupported Anthropic tool call entry")
		}
		args := c.Args
		if len(args) == 0 {
			args = json.RawMessage(`{}`)
		}
		blocks = []anthropic.ContentBlockParamUnion{anthropic.NewToolUseBlock(c.ID, args, c.Name)}
	case KindToolResult:
		if e.ToolResult == nil {
			return nil, errors.New("tool result entry carries no tool result")
		}
		r := e.ToolResult
		output := r.Output
		if r.Error != "" {
			output = r.Error
		}
		blocks = []anthropic.ContentBlockParamUnion{anthropic.NewToolResultBlock(r.CallID, output, r.Error != "")}
	default:
		return nil, fmt.Errorf("unsupported entry kind %d for Anthropic content block", e.Kind)
	}

	return blocks, nil
}
