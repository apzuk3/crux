package crux

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

const anthropicContentBlockOpaqueKey = "anthropic.message.content_block"

func (a *Agent) newAnthropicClient(httpClient *http.Client) *anthropic.Client {
	var opts []option.RequestOption
	if a.apiKey != "" {
		opts = append(opts, option.WithAPIKey(a.apiKey))
	}

	if a.baseURL != "" {
		opts = append(opts, option.WithBaseURL(a.baseURL))
	}

	client := a.effectiveHTTPClient(httpClient)
	if client != nil {
		opts = append(opts, option.WithHTTPClient(client))
	}

	c := anthropic.NewClient(opts...)
	return &c
}

// defaultAnthropicMaxTokens is used when WithMaxTokens is unset, because the
// Messages API requires a cap. It stays below the SDK's non-streaming limit.
const defaultAnthropicMaxTokens = 16384

// anthropicStep returns ordered blocks with usage, resuming paused server turns
// internally until the model finishes or requests a local tool.
func (a *Agent) anthropicStep(ctx context.Context, log []Entry, httpClient *http.Client, emit chunkSink) ([]Entry, error) {
	messages, err := toAnthropicMessages(log)
	if err != nil {
		return nil, err
	}
	tools, err := anthropicTools(a.tools)
	if err != nil {
		return nil, err
	}
	params := anthropic.MessageNewParams{
		Model: a.model, Messages: messages, Tools: tools,
		MaxTokens: defaultAnthropicMaxTokens,
	}
	if a.maxTokens > 0 {
		params.MaxTokens = int64(a.maxTokens)
	}
	if a.temperature != nil {
		params.Temperature = anthropic.Float(*a.temperature)
	}
	if a.instructions != "" {
		params.System = []anthropic.TextBlockParam{{Text: a.instructions}}
	}
	if a.searchOptions != nil {
		tool := &anthropic.WebSearchTool20250305Param{}
		if location := a.searchOptions.UserLocation; location != nil {
			if location.Country != "" {
				tool.UserLocation.Country = anthropic.String(location.Country)
			}
			if location.City != "" {
				tool.UserLocation.City = anthropic.String(location.City)
			}
			if location.Region != "" {
				tool.UserLocation.Region = anthropic.String(location.Region)
			}
			if location.Timezone != "" {
				tool.UserLocation.Timezone = anthropic.String(location.Timezone)
			}
		}
		params.Tools = append(params.Tools, anthropic.ToolUnionParam{
			OfWebSearchTool20250305: tool,
		})
	}
	if a.outputSchema != nil {
		schema, err := wireSchemaFor(a.outputSchema, a.provider)
		if err != nil {
			return nil, err
		}
		params.OutputConfig = anthropic.OutputConfigParam{
			Format: anthropic.JSONOutputFormatParam{Schema: schema},
		}
	}
	client := a.newAnthropicClient(httpClient)
	now := time.Now().UTC()
	// Bound server-side continuation independently of the agent's tool turns.
	const maxContinuations = 10
	var (
		produced              []Entry
		totalInputTokens      int
		totalOutputTokens     int
		totalCacheReadTokens  int
		totalCacheWriteTokens int
	)

	// The SDK refuses non-streaming requests that may run past ten minutes,
	// judged from max_tokens, so those stream without emitting chunks.
	if emit == nil {
		if _, err := anthropic.CalculateNonStreamingTimeout(int(params.MaxTokens), params.Model, nil); err != nil {
			emit = func(Chunk) error { return nil }
		}
	}

	// anthropic has internal tool calling limitations. Once it's reach the maximum
	// it will pause and wait until the content is sent back to continue
	// https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons#pause-turn
	for continuations := 0; ; continuations++ {
		var response *anthropic.Message
		if emit == nil {
			response, err = client.Messages.New(ctx, params)
		} else {
			response, err = streamAnthropic(ctx, client, params, emit)
		}
		if err != nil {
			return nil, fmt.Errorf("anthropic messages: %w", err)
		}
		switch response.StopReason {
		case anthropic.StopReasonEndTurn, anthropic.StopReasonStopSequence, anthropic.StopReasonToolUse, anthropic.StopReasonPauseTurn:
		case anthropic.StopReasonRefusal:
			return nil, fmt.Errorf("anthropic: %w", ErrRefused)
		case anthropic.StopReasonMaxTokens:
			return nil, fmt.Errorf("anthropic response hit the %d output token limit; raise it with crux.WithMaxTokens", params.MaxTokens)
		default:
			return nil, fmt.Errorf("anthropic response did not complete: stop reason %q", response.StopReason)
		}
		if len(response.Content) == 0 {
			return nil, errors.New("anthropic returned no content")
		}
		for _, block := range response.Content {
			entry, err := fromAnthropicContentBlock(block)
			if err != nil {
				return nil, err
			}
			if entry.At.IsZero() {
				entry.At = now
			}
			produced = append(produced, entry)
		}
		totalInputTokens += int(response.Usage.InputTokens)
		totalOutputTokens += int(response.Usage.OutputTokens)
		totalCacheReadTokens += int(response.Usage.CacheReadInputTokens)
		totalCacheWriteTokens += int(response.Usage.CacheCreationInputTokens)

		if response.StopReason != anthropic.StopReasonPauseTurn {
			if len(produced) > 0 {
				produced[len(produced)-1].Usage = &Usage{
					InputTokens:      totalInputTokens,
					OutputTokens:     totalOutputTokens,
					CacheReadTokens:  totalCacheReadTokens,
					CacheWriteTokens: totalCacheWriteTokens,
				}
			}
			return produced, nil
		}
		if continuations == maxContinuations {
			return nil, errors.New("anthropic max paused-turn continuations reached")
		}
		// Replay all returned blocks unchanged, keeping the same tools and config.
		params.Messages = append(params.Messages, response.ToParam())
	}
}

func streamAnthropic(ctx context.Context, client *anthropic.Client, params anthropic.MessageNewParams, emit chunkSink) (*anthropic.Message, error) {
	stream := client.Messages.NewStreaming(ctx, params)
	defer stream.Close()
	var response anthropic.Message
	complete := false
	for stream.Next() {
		event := stream.Current()
		if err := response.Accumulate(event); err != nil {
			return nil, err
		}
		var err error
		switch event.Type {
		case "content_block_start":
			switch event.ContentBlock.Type {
			case "text":
				err = emitChunk(emit, ChunkText, event.ContentBlock.Text)
			case "thinking":
				err = emitChunk(emit, ChunkReasoning, event.ContentBlock.Thinking)
			}
		case "content_block_delta":
			switch event.Delta.Type {
			case "text_delta":
				err = emitChunk(emit, ChunkText, event.Delta.Text)
			case "thinking_delta":
				err = emitChunk(emit, ChunkReasoning, event.Delta.Thinking)
			}
		case "message_stop":
			complete = true
		case "error":
			return nil, fmt.Errorf("anthropic stream error: %s", event.RawJSON())
		}
		if err != nil {
			return nil, err
		}
	}
	if err := stream.Err(); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !complete {
		return nil, errors.New("anthropic stream ended without message_stop")
	}
	return &response, nil
}

func anthropicTools(selected []Tool) ([]anthropic.ToolUnionParam, error) {
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
	case anthropic.ServerToolUseBlock, anthropic.WebSearchToolResultBlock:
		return Entry{Kind: KindProviderTool, Opaque: opaque}, nil
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
		if e.Kind == KindProviderTool && len(e.Opaque[anthropicContentBlockOpaqueKey]) == 0 {
			continue
		}
		if e.HiddenFromModel() {
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
		args := c.objectArgs()
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
