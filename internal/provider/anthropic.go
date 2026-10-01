package provider

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

const anthropicContentBlockOpaqueKey = "anthropic.message.content_block"

// anthropicUnknownContentBlockOpaqueKey holds blocks the SDK cannot replay.
const anthropicUnknownContentBlockOpaqueKey = "anthropic.message.unknown_content_block"

func newAnthropicClient(req *Request) *anthropic.Client {
	var opts []option.RequestOption
	if req.APIKey != "" {
		opts = append(opts, option.WithAPIKey(req.APIKey))
	}

	if req.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(req.BaseURL))
	}

	client := req.HTTPClient
	if client == nil {
		// The SDK would fall back to http.DefaultClient, which never times out
		// a server that accepts a streaming request but does not answer.
		client = DefaultHTTPClient()
	}
	opts = append(opts, option.WithHTTPClient(client))

	c := anthropic.NewClient(opts...)
	return &c
}

// defaultAnthropicMaxTokens is used when WithMaxTokens is unset, because the
// Messages API requires a cap. It stays below the SDK's non-streaming limit.
const defaultAnthropicMaxTokens = 16384

// anthropicStep returns ordered blocks with usage, resuming paused server turns
// internally until the model finishes or requests a local tool.
func Anthropic(ctx context.Context, req *Request, emit Emit) ([]Item, error) {
	messages, err := toAnthropicMessages(req.Log)
	if err != nil {
		return nil, err
	}
	tools, err := anthropicTools(req.Tools)
	if err != nil {
		return nil, err
	}
	params := anthropic.MessageNewParams{
		Model: req.Model, Messages: messages, Tools: tools,
		MaxTokens: defaultAnthropicMaxTokens,
	}
	if req.MaxTokens > 0 {
		params.MaxTokens = int64(req.MaxTokens)
	}
	if req.Temperature != nil {
		params.Temperature = anthropic.Float(*req.Temperature)
	}
	if req.Instructions != "" {
		params.System = []anthropic.TextBlockParam{{Text: req.Instructions}}
	}
	if req.Search != nil {
		tool := &anthropic.WebSearchTool20250305Param{}
		if location := req.Search.Location; location != nil {
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
	if schema := req.OutputSchema; schema != nil {
		params.OutputConfig = anthropic.OutputConfigParam{
			Format: anthropic.JSONOutputFormatParam{Schema: schema},
		}
	}
	if err := setAnthropicReasoning(req, &params); err != nil {
		return nil, err
	}
	setAnthropicCacheBreakpoints(&params)
	client := newAnthropicClient(req)
	now := time.Now().UTC()
	// Bound server-side continuation independently of the agent's tool turns.
	const maxContinuations = 10
	var (
		produced              []Item
		totalInputTokens      int
		totalOutputTokens     int
		totalCacheReadTokens  int
		totalCacheWriteTokens int
	)

	// The SDK refuses non-streaming requests that may run past ten minutes,
	// judged from max_tokens, so those stream without emitting chunks.
	if emit == nil {
		if _, err := anthropic.CalculateNonStreamingTimeout(int(params.MaxTokens), params.Model, nil); err != nil {
			emit = func(ChunkKind, string) error { return nil }
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
			return nil, refused("anthropic", "")
		case anthropic.StopReasonMaxTokens:
			return nil, fmt.Errorf("anthropic response hit the %d output token limit; raise it with crux.WithMaxTokens", params.MaxTokens)
		default:
			return nil, fmt.Errorf("anthropic response did not complete: stop reason %q", response.StopReason)
		}
		if len(response.Content) == 0 && (response.StopReason == anthropic.StopReasonToolUse || response.StopReason == anthropic.StopReasonPauseTurn) {
			return nil, fmt.Errorf("anthropic returned no content with stop reason %q", response.StopReason)
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
		// Anthropic reports uncached input separately from cache reads and writes;
		// total input is their sum. See "Tracking cache performance":
		// https://platform.claude.com/docs/en/build-with-claude/prompt-caching
		totalInputTokens += int(response.Usage.InputTokens + response.Usage.CacheReadInputTokens + response.Usage.CacheCreationInputTokens)
		totalOutputTokens += int(response.Usage.OutputTokens)
		totalCacheReadTokens += int(response.Usage.CacheReadInputTokens)
		totalCacheWriteTokens += int(response.Usage.CacheCreationInputTokens)

		if response.StopReason != anthropic.StopReasonPauseTurn {
			if !hasAnswerOrCall(produced) {
				// The model may end its turn without saying anything; that is
				// still a final answer, and its usage must not be lost. See
				// "Empty responses with end_turn":
				// https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons
				produced = append(produced, Item{At: now, Kind: KindAssistant})
			}
			produced[len(produced)-1].Response = &ResponseInfo{ID: response.ID}
			produced[len(produced)-1].Usage = &Usage{
				InputTokens:      totalInputTokens,
				OutputTokens:     totalOutputTokens,
				CacheReadTokens:  totalCacheReadTokens,
				CacheWriteTokens: totalCacheWriteTokens,
			}
			return produced, nil
		}
		if continuations == maxContinuations {
			return nil, errors.New("anthropic max paused-turn continuations reached")
		}
		// Replay the returned blocks unchanged, keeping the same tools and
		// config, but without the empty text blocks the API rejects.
		paused := response.ToParam()
		paused.Content = slices.DeleteFunc(paused.Content, func(b anthropic.ContentBlockParamUnion) bool {
			return b.OfText != nil && strings.TrimSpace(b.OfText.Text) == ""
		})
		if len(paused.Content) > 0 {
			params.Messages = append(params.Messages, paused)
		}
	}
}

// anthropicThinkingBudgets are the thinking budgets of models without
// adaptive thinking. A budget must be at least 1024 and below max_tokens.
var anthropicThinkingBudgets = map[string]int64{
	ReasoningLow: 2048, ReasoningMedium: 8192, ReasoningHigh: 16384, ReasoningMax: 32768,
}

// setAnthropicReasoning maps WithReasoning onto the model's thinking settings:
// adaptive thinking with an effort from Claude 4.6, a thinking budget before
// it. Models that always think reject disabled thinking, so ReasoningOff uses
// their lowest effort. The thinking summary is requested so it can stream.
// https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
func setAnthropicReasoning(req *Request, params *anthropic.MessageNewParams) error {
	if req.Reasoning == "" {
		return nil
	}
	major, minor, family := claudeVersion(req.Model)
	budgetThinking := major > 0 && (major < 4 || major == 4 && minor < 6)
	alwaysThinks := family == "fable" || family == "mythos" || major == 5 && minor >= 5 || major > 5
	switch {
	case req.Reasoning == ReasoningOff && alwaysThinks:
		params.OutputConfig.Effort = anthropic.OutputConfigEffortLow
	case req.Reasoning == ReasoningOff:
		params.Thinking = anthropic.ThinkingConfigParamUnion{OfDisabled: &anthropic.ThinkingConfigDisabledParam{}}
	case budgetThinking:
		budget := anthropicThinkingBudgets[req.Reasoning]
		if req.MaxTokens == 0 {
			params.MaxTokens = budget + defaultAnthropicMaxTokens
		}
		budget = min(budget, params.MaxTokens-1)
		if budget < 1024 {
			return fmt.Errorf("anthropic thinking needs more than 1024 output tokens; raise crux.WithMaxTokens above %d", params.MaxTokens)
		}
		params.Thinking = anthropic.ThinkingConfigParamOfEnabled(budget)
	default:
		params.Thinking = anthropic.ThinkingConfigParamUnion{OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{
			Display: anthropic.ThinkingConfigAdaptiveDisplaySummarized,
		}}
		params.OutputConfig.Effort = anthropic.OutputConfigEffort(req.Reasoning)
	}
	return nil
}

// claudeVersion reads the family and version from a model ID such as
// claude-opus-4-6, claude-sonnet-4-5-20250929 or claude-3-5-haiku-20241022.
// major is 0 when the ID has no version.
func claudeVersion(model string) (major, minor int, family string) {
	parts := strings.Split(model, "-")
	if len(parts) < 2 || parts[0] != "claude" {
		return 0, 0, ""
	}
	var numbers []int
	for _, part := range parts[1:] {
		n, err := strconv.Atoi(part)
		switch {
		case err != nil:
			if family == "" {
				family = part
			}
		case len(part) <= 2 && len(numbers) < 2:
			numbers = append(numbers, n)
		}
	}
	if len(numbers) == 0 {
		return 0, 0, family
	}
	if len(numbers) == 2 {
		minor = numbers[1]
	}
	return numbers[0], minor, family
}

// setAnthropicCacheBreakpoints caches the prompt, because every turn of a run
// resends the whole conversation. The top-level marker moves to the last block
// of each request; the one on the tools and system prompt keeps that prefix
// cached when a turn adds more blocks than the cache lookback covers. Prompts
// below the model's minimum cacheable length are simply not cached.
// https://platform.claude.com/docs/en/build-with-claude/prompt-caching
func setAnthropicCacheBreakpoints(params *anthropic.MessageNewParams) {
	params.CacheControl = anthropic.NewCacheControlEphemeralParam()
	if n := len(params.System); n > 0 {
		params.System[n-1].CacheControl = anthropic.NewCacheControlEphemeralParam()
		return
	}
	if n := len(params.Tools); n > 0 {
		if cache := params.Tools[n-1].GetCacheControl(); cache != nil {
			*cache = anthropic.NewCacheControlEphemeralParam()
		}
	}
}

func streamAnthropic(ctx context.Context, client *anthropic.Client, params anthropic.MessageNewParams, emit Emit) (*anthropic.Message, error) {
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
		if tool.Schema["type"] != "object" {
			return nil, fmt.Errorf("anthropic tool %q requires an object input schema", tool.Name)
		}
		raw, err := json.Marshal(tool.Schema)
		if err != nil {
			return nil, fmt.Errorf("marshal tool %q schema: %w", tool.Name, err)
		}
		var schema anthropic.ToolInputSchemaParam
		if err := json.Unmarshal(raw, &schema); err != nil {
			return nil, fmt.Errorf("decode tool %q schema: %w", tool.Name, err)
		}
		// The SDK's param decoder does not retain unknown schema keywords.
		schema.ExtraFields = make(map[string]any)
		for key, value := range tool.Schema {
			if key != "type" && key != "properties" && key != "required" {
				schema.ExtraFields[key] = value
			}
		}
		tools = append(tools, anthropic.ToolUnionParam{OfTool: &anthropic.ToolParam{
			Name: tool.Name, Description: anthropic.String(tool.Description), InputSchema: schema,
		}})
	}
	return tools, nil
}

func fromAnthropicContentBlock(block anthropic.ContentBlockUnion) (Item, error) {
	opaque := map[string][]byte{anthropicContentBlockOpaqueKey: []byte(block.RawJSON())}
	switch v := block.AsAny().(type) {
	case anthropic.TextBlock:
		return Item{Kind: KindAssistant, Content: []ContentPart{{Kind: ContentKindText, Text: v.Text}}, Opaque: opaque}, nil
	case anthropic.ThinkingBlock:
		return Item{Kind: KindReasoning, Reasoning: &Reasoning{Summary: v.Thinking}, Opaque: opaque}, nil
	case anthropic.RedactedThinkingBlock:
		return Item{Kind: KindReasoning, Reasoning: &Reasoning{}, Opaque: opaque}, nil
	// Server tools run on Anthropic's side; their blocks are replayed as-is.
	// https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools
	case anthropic.ServerToolUseBlock, anthropic.WebSearchToolResultBlock, anthropic.WebFetchToolResultBlock,
		anthropic.CodeExecutionToolResultBlock, anthropic.BashCodeExecutionToolResultBlock,
		anthropic.TextEditorCodeExecutionToolResultBlock, anthropic.ToolSearchToolResultBlock, anthropic.ContainerUploadBlock:
		return Item{Kind: KindProviderTool, Opaque: opaque}, nil
	case anthropic.ToolUseBlock:
		return Item{Kind: KindToolCall, ToolCall: &ToolCall{
			ID: v.ID, Name: v.Name, Args: v.Input,
		}, Opaque: opaque}, nil
	default:
		// Keep blocks this SDK does not know for inspection; they are not replayed.
		return Item{Kind: KindProviderTool, Opaque: map[string][]byte{anthropicUnknownContentBlockOpaqueKey: []byte(block.RawJSON())}}, nil
	}
}

// toAnthropicMessages groups consecutive blocks by role.
func toAnthropicMessages(log []Item) ([]anthropic.MessageParam, error) {
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

// toAnthropicFileBlock renders a file as an image, a PDF document or text.
func toAnthropicFileBlock(part ContentPart) (anthropic.ContentBlockParamUnion, error) {
	if text, ok := part.InlineText(); ok {
		return anthropic.NewTextBlock(text), nil
	}
	switch {
	case part.IsImage() && part.URL != "":
		return anthropic.NewImageBlock(anthropic.URLImageSourceParam{URL: part.URL}), nil
	case slices.Contains([]string{"image/jpeg", "image/png", "image/gif", "image/webp"}, part.MIME):
		return anthropic.NewImageBlockBase64(part.MIME, base64.StdEncoding.EncodeToString(part.Data)), nil
	case part.MIME == "application/pdf":
		var block anthropic.ContentBlockParamUnion
		if part.URL != "" {
			block = anthropic.NewDocumentBlock(anthropic.URLPDFSourceParam{URL: part.URL})
		} else {
			block = anthropic.NewDocumentBlock(anthropic.Base64PDFSourceParam{Data: base64.StdEncoding.EncodeToString(part.Data)})
		}
		if part.Name != "" {
			block.OfDocument.Title = anthropic.String(part.Name)
		}
		return block, nil
	}
	return anthropic.ContentBlockParamUnion{}, unsupportedFile("anthropic", part)
}

// toAnthropicContentBlockParamUnion renders an entry as content blocks.
func toAnthropicContentBlockParamUnion(e Item) ([]anthropic.ContentBlockParamUnion, error) {
	if raw := e.Opaque[anthropicContentBlockOpaqueKey]; len(raw) > 0 && e.Kind != KindUser && e.Kind != KindToolResult {
		var block anthropic.ContentBlockUnion
		if err := json.Unmarshal(raw, &block); err != nil {
			return nil, fmt.Errorf("decode Anthropic content block: %w", err)
		}
		if block.Type == "text" && strings.TrimSpace(block.Text) == "" {
			// Claude often sends "\n\n" before a tool call, and a streamed text
			// block may stay empty, but the API rejects such blocks on replay.
			return nil, nil
		}
		return []anthropic.ContentBlockParamUnion{block.ToParam()}, nil
	}

	var blocks []anthropic.ContentBlockParamUnion
	switch e.Kind {
	case KindUser, KindAssistant:
		for _, part := range e.Content {
			if part.Kind == ContentKindFile {
				block, err := toAnthropicFileBlock(part)
				if err != nil {
					return nil, err
				}
				blocks = append(blocks, block)
				continue
			}
			if part.Kind != ContentKindText && part.Kind != ContentKindRefusal {
				return nil, fmt.Errorf("unsupported content part kind %q", part.Kind)
			}
			if strings.TrimSpace(part.Text) == "" {
				continue // Anthropic rejects empty text blocks.
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
		isError := r.Error != ""
		output := r.Output
		if isError {
			output = r.Error
		}
		// Anthropic rejects empty text blocks, so an empty result carries no
		// content, which the API allows ("content (optional)"):
		// https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls
		if strings.TrimSpace(output) == "" {
			block := anthropic.ToolResultBlockParam{ToolUseID: r.CallID, IsError: anthropic.Bool(isError)}
			blocks = []anthropic.ContentBlockParamUnion{{OfToolResult: &block}}
			break
		}
		blocks = []anthropic.ContentBlockParamUnion{anthropic.NewToolResultBlock(r.CallID, output, isError)}
	default:
		return nil, fmt.Errorf("unsupported entry kind %d for Anthropic content block", e.Kind)
	}

	return blocks, nil
}
