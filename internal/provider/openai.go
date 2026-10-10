package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

// Providers the OpenAI wire code treats specially, as crux.Provider spells them.
const (
	providerOpenAI     = "openai"
	providerOpenRouter = "openrouter"
	providerXAI        = "xai"
)

const openAIOutputItemOpaqueKey = "openai.response.output_item"

// openAIUnknownOutputItemOpaqueKey holds output items that cannot be replayed.
const openAIUnknownOutputItemOpaqueKey = "openai.response.unknown_output_item"

// openAIProviderOpaqueKey records which provider produced an output item.
// Every OpenAI-compatible provider shares the item keys, and one provider's
// items (such as encrypted reasoning) must not be replayed to another.
const openAIProviderOpaqueKey = "openai.response.provider"

// withoutForeignOpenAIItems drops the output items another provider produced,
// as Fork does when the provider changes, so a stored session can be resumed
// with an agent on a different OpenAI-compatible provider. Entries recorded
// without a provider are kept as they are.
func withoutForeignOpenAIItems(log []Item, provider string) []Item {
	foreign := func(e Item) bool {
		p, ok := e.Opaque[openAIProviderOpaqueKey]
		return ok && string(p) != provider
	}
	if !slices.ContainsFunc(log, foreign) {
		return log
	}
	out := make([]Item, 0, len(log))
	for _, e := range log {
		if foreign(e) {
			if e.Kind == KindProviderTool {
				continue
			}
			e.Opaque = nil
		}
		out = append(out, e)
	}
	return out
}

func newOpenAIClient(req *Request) *openai.Client {
	var opts []option.RequestOption
	if key := req.apiKey(); key != "" {
		opts = append(opts, option.WithAPIKey(key))
	}
	if req.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(req.BaseURL))
	}
	opts = append(opts, option.WithMaxRetries(req.Retries()), option.WithMiddleware(normalizeErrorBody))
	client := req.HTTPClient
	if client != nil {
		opts = append(opts, option.WithHTTPClient(client))
	} else if req.Provider != providerOpenAI {
		// Construct the Responses service directly to avoid inheriting OpenAI
		// credentials, organization, project, or custom headers from the environment.
		opts = append(opts, option.WithHTTPClient(DefaultHTTPClient()))
	}
	if req.Provider != providerOpenAI {
		return &openai.Client{Options: opts, Responses: responses.NewResponseService(opts...)}
	}

	c := openai.NewClient(opts...)
	return &c
}

// normalizeErrorBody is a client middleware for error responses whose "error"
// is a string, as xAI sends them ({"code":"invalid-argument","error":"..."}).
// The SDK decodes "error" as an object and, when that fails, reports only the
// decode error, dropping the status, headers and message. Rewriting the body
// into the object shape keeps them.
func normalizeErrorBody(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	resp, err := next(req)
	if err != nil || resp == nil || resp.StatusCode < http.StatusBadRequest || resp.Body == nil {
		return resp, err
	}
	body, readErr := io.ReadAll(resp.Body)
	resp.Body.Close()
	if readErr == nil {
		if fixed, ok := errorObjectBody(body); ok {
			body = fixed
		}
	}
	resp.Body = io.NopCloser(bytes.NewReader(body))
	resp.ContentLength = int64(len(body))
	return resp, nil
}

// errorObjectBody returns body with a string "error" field replaced by
// {"message": ..., "code": ...}, and false when the body is not such an object.
func errorObjectBody(body []byte) ([]byte, bool) {
	var fields map[string]json.RawMessage
	if json.Unmarshal(body, &fields) != nil {
		return nil, false
	}
	var message string
	if json.Unmarshal(fields["error"], &message) != nil {
		return nil, false
	}
	object := map[string]json.RawMessage{"message": fields["error"]}
	if code, ok := fields["code"]; ok {
		object["code"] = code
	}
	fields["error"], _ = json.Marshal(object)
	out, err := json.Marshal(fields)
	return out, err == nil
}

// OpenAI sends the log and returns the model's entries with usage attached.
func OpenAI(ctx context.Context, req *Request, emit Emit) ([]Item, error) {
	params, err := openAIParams(req)
	if err != nil {
		return nil, err
	}
	client := newOpenAIClient(req)
	var response *responses.Response
	if emit == nil {
		response, err = client.Responses.New(ctx, params)
	} else {
		response, err = streamOpenAI(ctx, client, params, emit)
	}
	if err != nil {
		return nil, fmt.Errorf("%s responses: %w", req.Provider, openAILimit(err))
	}
	// Scan all messages before converting items or executing any local tools.
	if err := openAIRefusal(req.Provider, response); err != nil {
		return nil, err
	}
	if err := openAIStatusError(req.Provider, response); err != nil {
		return nil, err
	}
	now := time.Now().UTC()
	produced, err := openAIItems(response, req.Provider, now)
	if err != nil {
		return nil, err
	}
	// InputTokens already includes cached tokens:
	// https://platform.openai.com/docs/guides/prompt-caching
	usage := response.Usage
	return finishTurn(produced, now, response.ID, &Usage{
		InputTokens:      int(usage.InputTokens),
		OutputTokens:     int(usage.OutputTokens),
		CacheReadTokens:  int(usage.InputTokensDetails.CachedTokens),
		CacheWriteTokens: int(usage.InputTokensDetails.CacheWriteTokens),
	}), nil
}

// openAIParams builds the request from req.
func openAIParams(req *Request) (responses.ResponseNewParams, error) {
	input, err := toOpenAIResponseInput(withoutForeignOpenAIItems(req.Log, req.Provider))
	if err != nil {
		return responses.ResponseNewParams{}, err
	}
	params := responses.ResponseNewParams{
		Model: openai.ResponsesModel(req.Model),
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: input},
		Tools: openAITools(req.Tools),
		Store: openai.Bool(false),
	}
	switch req.Provider {
	case providerOpenAI, providerOpenRouter, providerXAI:
		// Nothing is kept server side, so reasoning travels with the log.
		params.Include = []responses.ResponseIncludable{responses.ResponseIncludableReasoningEncryptedContent}
	}
	if req.Instructions != "" {
		params.Instructions = openai.String(req.Instructions)
	}
	setOpenAIToolChoice(req, &params)
	if req.MaxTokens > 0 {
		params.MaxOutputTokens = openai.Int(int64(req.MaxTokens))
	}
	if req.Temperature != nil {
		params.Temperature = openai.Float(*req.Temperature)
	}
	setOpenAIReasoning(req, &params)
	if req.Search != nil {
		params.Tools = append(params.Tools, openAISearchTool(req))
	}
	if schema := req.OutputSchema; schema != nil {
		params.Text = openAIOutputFormat(schema)
	}
	return params, nil
}

// setOpenAIToolChoice maps the tool choice and WithParallelToolCalls, which
// only mean something when the model has a tool to call.
func setOpenAIToolChoice(req *Request, params *responses.ResponseNewParams) {
	if len(req.Tools) == 0 && req.Search == nil {
		return
	}
	switch req.ToolChoice {
	case ToolChoiceAuto, ToolChoiceRequired, ToolChoiceNone:
		params.ToolChoice.OfToolChoiceMode = openai.Opt(responses.ToolChoiceOptions(req.ToolChoice))
	case ToolChoiceTool:
		params.ToolChoice.OfFunctionTool = &responses.ToolChoiceFunctionParam{Name: req.ToolName}
	}
	if req.Parallel != nil {
		params.ParallelToolCalls = openai.Bool(*req.Parallel)
	}
}

// setOpenAIReasoning maps WithReasoning onto the reasoning effort.
// https://platform.openai.com/docs/guides/reasoning
func setOpenAIReasoning(req *Request, params *responses.ResponseNewParams) {
	if req.Reasoning == "" {
		return
	}
	params.Reasoning = shared.ReasoningParam{Effort: shared.ReasoningEffort(req.Reasoning)}
	if req.Reasoning == ReasoningOff {
		params.Reasoning.Effort = shared.ReasoningEffortNone
	} else if req.Provider == providerOpenAI {
		// Without a summary, OpenAI returns no readable reasoning.
		params.Reasoning.Summary = shared.ReasoningSummaryAuto
	}
}

// openAISearchTool returns the web search tool. Only OpenAI itself takes a
// user location.
func openAISearchTool(req *Request) responses.ToolUnionParam {
	tool := responses.ToolParamOfWebSearch(responses.WebSearchToolTypeWebSearch)
	if req.Provider == providerOpenAI {
		openAIUserLocation(&tool.OfWebSearch.UserLocation, req.Search.Location)
	}
	return tool
}

func openAIUserLocation(dst *responses.WebSearchToolUserLocationParam, location *Location) {
	if location == nil {
		return
	}
	if location.Country != "" {
		dst.Country = openai.String(location.Country)
	}
	if location.City != "" {
		dst.City = openai.String(location.City)
	}
	if location.Region != "" {
		dst.Region = openai.String(location.Region)
	}
	if location.Timezone != "" {
		dst.Timezone = openai.String(location.Timezone)
	}
	if !location.empty() {
		dst.Type = "approximate"
	}
}

func openAIOutputFormat(schema map[string]any) responses.ResponseTextConfigParam {
	return responses.ResponseTextConfigParam{
		Format: responses.ResponseFormatTextConfigUnionParam{
			OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
				Name:   "output",
				Schema: schema,
				Strict: openai.Bool(true),
			},
		},
	}
}

// openAIRefusal returns the refusal in any message of the response.
func openAIRefusal(provider string, response *responses.Response) error {
	for _, item := range response.Output {
		message, ok := item.AsAny().(responses.ResponseOutputMessage)
		if !ok {
			continue
		}
		for _, part := range message.Content {
			if refusal, ok := part.AsAny().(responses.ResponseOutputRefusal); ok {
				return refused(provider, refusal.Refusal)
			}
		}
	}
	return nil
}

// openAIStatusError rejects a response that failed or did not complete.
func openAIStatusError(provider string, response *responses.Response) error {
	if response.Error.Code != "" || response.Status == responses.ResponseStatusFailed {
		err := fmt.Errorf("%s response failed: %s: %s", provider, response.Error.Code, response.Error.Message)
		// OpenRouter adds its canonical error_type next to the error.
		return bodyLimit(err, []byte(response.RawJSON()), string(response.Error.Code))
	}
	if response.Status == responses.ResponseStatusCompleted {
		return nil
	}
	// incomplete_details.reason: https://platform.openai.com/docs/api-reference/responses/object
	if response.IncompleteDetails.Reason == "content_filter" {
		return refused(provider, "content_filter")
	}
	if response.IncompleteDetails.Reason == "max_output_tokens" {
		return fmt.Errorf("%s response hit the output token limit; raise it with crux.WithMaxTokens", provider)
	}
	return fmt.Errorf("%s response did not complete: status %q, reason %q", provider, response.Status, response.IncompleteDetails.Reason)
}

// openAIItems converts the response's output items, in order, marking each
// with the provider that produced it.
func openAIItems(response *responses.Response, provider string, now time.Time) ([]Item, error) {
	produced := make([]Item, 0, len(response.Output))
	for _, item := range response.Output {
		entry, err := fromOpenAIResponseOutputItemUnion(item)
		if err != nil {
			return nil, err
		}
		if entry.At.IsZero() {
			entry.At = now
		}
		if entry.Opaque == nil {
			entry.Opaque = make(map[string][]byte)
		}
		entry.Opaque[openAIProviderOpaqueKey] = []byte(provider)
		produced = append(produced, entry)
	}
	return produced, nil
}

func streamOpenAI(ctx context.Context, client *openai.Client, params responses.ResponseNewParams, emit Emit) (*responses.Response, error) {
	stream := client.Responses.NewStreaming(ctx, params)
	defer stream.Close()
	var response *responses.Response
	commentary := commentaryTracker{ids: make(map[string]bool), indexes: make(map[int64]bool)}
	for stream.Next() {
		event := stream.Current()
		var err error
		switch event.Type {
		case "response.output_item.added":
			commentary.note(event)
		case "response.output_text.delta":
			err = emitChunk(emit, commentary.kind(event), event.Delta)
		case "response.reasoning_text.delta", "response.reasoning_summary_text.delta":
			err = emitChunk(emit, ChunkReasoning, event.Delta)
		case "response.completed", "response.failed", "response.incomplete":
			value := event.Response
			response = &value
		case "error", "response.error":
			return nil, bodyLimit(fmt.Errorf("provider stream error: %s", event.RawJSON()), []byte(event.RawJSON()))
		}
		if err != nil {
			return nil, err
		}
	}
	if err := streamEnd(ctx, stream.Err(), response != nil, "provider stream ended without a terminal response"); err != nil {
		return nil, err
	}
	return response, nil
}

// commentaryTracker remembers which streamed messages are commentary. Those
// become reasoning entries, so their text streams as reasoning too. Deltas
// name their item by ID and output index.
type commentaryTracker struct {
	ids     map[string]bool
	indexes map[int64]bool
}

// note records an added output item when it is a commentary message.
func (t commentaryTracker) note(event responses.ResponseStreamEventUnion) {
	item := event.Item
	if item.Type != "message" || item.Phase != responses.ResponseOutputMessagePhaseCommentary {
		return
	}
	if item.ID != "" {
		t.ids[item.ID] = true
	}
	t.indexes[event.OutputIndex] = true
}

// kind says how a text delta streams: as reasoning when its item is commentary.
func (t commentaryTracker) kind(event responses.ResponseStreamEventUnion) ChunkKind {
	if t.ids[event.ItemID] || (event.ItemID == "" && t.indexes[event.OutputIndex]) {
		return ChunkReasoning
	}
	return ChunkText
}

func openAITools(tools []Tool) []responses.ToolUnionParam {
	params := make([]responses.ToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		params = append(params, responses.ToolUnionParam{
			OfFunction: &responses.FunctionToolParam{
				Name:        tool.Name,
				Description: openai.String(tool.Description),
				Parameters:  tool.Schema,
				Strict:      openai.Bool(false),
			},
		})
	}
	return params
}

func fromOpenAIResponseOutputItemUnion(item responses.ResponseOutputItemUnion) (Item, error) {
	opaque := map[string][]byte{
		openAIOutputItemOpaqueKey: []byte(item.RawJSON()),
	}

	switch v := item.AsAny().(type) {
	case responses.ResponseOutputMessage:
		content := make([]ContentPart, 0, len(v.Content))
		for _, part := range v.Content {
			switch value := part.AsAny().(type) {
			case responses.ResponseOutputText:
				content = append(content, ContentPart{
					Kind: ContentKindText,
					Text: value.Text,
				})
			case responses.ResponseOutputRefusal:
				content = append(content, ContentPart{
					Kind: ContentKindRefusal,
					Text: value.Refusal,
				})
			}
		}
		if v.Phase == responses.ResponseOutputMessagePhaseCommentary && !slices.ContainsFunc(content, func(p ContentPart) bool { return p.Kind == ContentKindRefusal }) {
			// Commentary is what the model says while it works, not its
			// answer, so it must not become part of the final output. It is
			// still replayed as the original message.
			var text strings.Builder
			for _, part := range content {
				text.WriteString(part.Text)
			}
			return Item{Kind: KindReasoning, Reasoning: &Reasoning{Summary: text.String()}, Opaque: opaque}, nil
		}
		return Item{
			Kind:    KindAssistant,
			Content: content,
			Opaque:  opaque,
		}, nil
	case responses.ResponseReasoningItem:
		// A summary comes in parts, one per paragraph.
		// https://platform.openai.com/docs/guides/reasoning#reasoning-summaries
		summary := make([]string, 0, len(v.Summary))
		for _, part := range v.Summary {
			summary = append(summary, part.Text)
		}
		return Item{
			Kind: KindReasoning,
			Reasoning: &Reasoning{
				Summary: strings.Join(summary, "\n\n"),
			},
			Opaque: opaque,
		}, nil
	case responses.ResponseFunctionWebSearch:
		return Item{Kind: KindProviderTool, Opaque: opaque}, nil
	case responses.ResponseFunctionToolCall:
		return Item{
			Kind: KindToolCall,
			ToolCall: &ToolCall{
				ID:   v.CallID,
				Name: v.Name,
				Args: normalizeToolArgs(v.Arguments),
			},
			Opaque: opaque,
		}, nil
	default:
		// Keep items this SDK does not know for inspection; they are not replayed.
		return Item{Kind: KindProviderTool, Opaque: map[string][]byte{openAIUnknownOutputItemOpaqueKey: []byte(item.RawJSON())}}, nil
	}
}

// toOpenAIResponseInput renders a log as the input of the next request.
func toOpenAIResponseInput(log []Item) (responses.ResponseInputParam, error) {
	input := make(responses.ResponseInputParam, 0, len(log))

	for _, e := range log {
		if e.Kind == KindProviderTool && len(e.Opaque[openAIOutputItemOpaqueKey]) == 0 {
			continue
		}
		if e.Kind == KindReasoning && len(e.Opaque[openAIOutputItemOpaqueKey]) == 0 {
			continue // Foreign reasoning cannot be replayed without provider data.
		}
		if e.Kind == KindAssistant && len(e.Content) == 0 && len(e.Opaque[openAIOutputItemOpaqueKey]) == 0 {
			continue // An empty answer has nothing to replay.
		}

		item, err := toOpenAIResponseInputItemUnionParam(e)
		if err != nil {
			return nil, err
		}
		input = append(input, item)
	}

	return input, nil
}

// toOpenAIUserContent renders a user message: a string when it is only text,
// otherwise a list with images and files.
func toOpenAIUserContent(parts []ContentPart) (responses.EasyInputMessageContentUnionParam, error) {
	if !slices.ContainsFunc(parts, func(p ContentPart) bool { return p.Kind == ContentKindFile }) {
		var text strings.Builder
		for _, part := range parts {
			text.WriteString(part.Text)
		}
		return responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt(text.String())}, nil
	}
	list := make(responses.ResponseInputMessageContentListParam, 0, len(parts))
	for _, part := range parts {
		text, inline := part.InlineText()
		switch {
		case part.Kind == ContentKindText:
			text, inline = part.Text, true
		case part.Kind != ContentKindFile:
			return responses.EasyInputMessageContentUnionParam{}, fmt.Errorf("unsupported content part kind %q", part.Kind)
		}
		switch {
		case inline:
			list = append(list, responses.ResponseInputContentUnionParam{OfInputText: &responses.ResponseInputTextParam{Text: text}})
		case part.IsImage():
			list = append(list, openAIImagePart(part))
		default:
			list = append(list, openAIFilePart(part))
		}
	}
	return responses.EasyInputMessageContentUnionParam{OfInputItemContentList: list}, nil
}

func openAIImagePart(part ContentPart) responses.ResponseInputContentUnionParam {
	image := &responses.ResponseInputImageParam{Detail: responses.ResponseInputImageDetailAuto}
	if part.URL != "" {
		image.ImageURL = param.NewOpt(part.URL)
	} else {
		image.ImageURL = param.NewOpt(part.DataURL())
	}
	return responses.ResponseInputContentUnionParam{OfInputImage: image}
}

func openAIFilePart(part ContentPart) responses.ResponseInputContentUnionParam {
	file := &responses.ResponseInputFileParam{}
	if part.URL != "" {
		file.FileURL = param.NewOpt(part.URL)
	} else {
		file.FileData = param.NewOpt(part.DataURL())
		file.Filename = param.NewOpt(part.fileName())
	}
	return responses.ResponseInputContentUnionParam{OfInputFile: file}
}

// toOpenAIResponseInputItemUnionParam renders an entry as an input item.
func toOpenAIResponseInputItemUnionParam(e Item) (responses.ResponseInputItemUnionParam, error) {
	if raw, ok := e.Opaque[openAIOutputItemOpaqueKey]; ok && len(raw) > 0 {
		return openAIInputItemFromOutputItem(raw)
	}

	switch e.Kind {
	case KindUser:
		content, err := toOpenAIUserContent(e.Content)
		if err != nil {
			return responses.ResponseInputItemUnionParam{}, err
		}
		return responses.ResponseInputItemUnionParam{
			OfMessage: &responses.EasyInputMessageParam{
				Role:    responses.EasyInputMessageRoleUser,
				Content: content,
			},
		}, nil
	case KindAssistant:
		return openAIAssistantMessage(e.Content)
	case KindToolCall:
		if e.ToolCall == nil {
			return responses.ResponseInputItemUnionParam{}, errors.New("tool call entry carries no tool call")
		}
		return openAIFunctionCall(e.ToolCall), nil
	case KindToolResult:
		if e.ToolResult == nil {
			return responses.ResponseInputItemUnionParam{}, errors.New("tool result entry carries no tool result")
		}
		return openAIFunctionOutput(e.ToolResult), nil
	default:
		return responses.ResponseInputItemUnionParam{}, fmt.Errorf("unsupported entry kind %d for OpenAI input item", e.Kind)
	}
}

// openAIAssistantMessage renders an answer without provider data, such as
// one another provider gave.
func openAIAssistantMessage(parts []ContentPart) (responses.ResponseInputItemUnionParam, error) {
	content := make([]responses.ResponseOutputMessageContentUnionParam, 0, len(parts))
	for _, part := range parts {
		switch part.Kind {
		case ContentKindText:
			content = append(content, responses.ResponseOutputMessageContentUnionParam{
				OfOutputText: &responses.ResponseOutputTextParam{
					Text:        part.Text,
					Annotations: []responses.ResponseOutputTextAnnotationUnionParam{},
				},
			})
		case ContentKindRefusal:
			content = append(content, responses.ResponseOutputMessageContentUnionParam{
				OfRefusal: &responses.ResponseOutputRefusalParam{
					Refusal: part.Text,
				},
			})
		default:
			return responses.ResponseInputItemUnionParam{}, fmt.Errorf("unsupported content part kind %q", part.Kind)
		}
	}
	return responses.ResponseInputItemUnionParam{
		OfOutputMessage: &responses.ResponseOutputMessageParam{
			Content: content,
			Status:  responses.ResponseOutputMessageStatusCompleted,
		},
	}, nil
}

func openAIFunctionCall(c *ToolCall) responses.ResponseInputItemUnionParam {
	arguments := c.rawArgs()
	if arguments == "" {
		arguments = "{}"
	}
	return responses.ResponseInputItemUnionParam{
		OfFunctionCall: &responses.ResponseFunctionToolCallParam{
			CallID:    c.ID,
			Name:      c.Name,
			Arguments: arguments,
			Status:    responses.ResponseFunctionToolCallStatusCompleted,
		},
	}
}

// openAIFunctionOutput renders a tool result. A failed or declined call is
// still owed an output.
func openAIFunctionOutput(r *ToolResult) responses.ResponseInputItemUnionParam {
	output, _ := r.text()
	return responses.ResponseInputItemUnionParam{
		OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
			CallID: param.NewOpt(r.CallID),
			Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
				OfString: param.NewOpt(output),
			},
		},
	}
}

// openAIInputItemFromOutputItem replays a stored output item. It goes through
// the variant's own param form because an assistant message decoded straight
// into the input union matches EasyInputMessage first, losing its id, status
// and phase.
func openAIInputItemFromOutputItem(raw []byte) (responses.ResponseInputItemUnionParam, error) {
	var item responses.ResponseOutputItemUnion
	if err := json.Unmarshal(raw, &item); err != nil {
		return responses.ResponseInputItemUnionParam{}, fmt.Errorf("decode OpenAI output item: %w", err)
	}

	switch v := item.AsAny().(type) {
	case responses.ResponseOutputMessage:
		p := v.ToParam()
		return responses.ResponseInputItemUnionParam{OfOutputMessage: &p}, nil
	case responses.ResponseReasoningItem:
		p := v.ToParam()
		return responses.ResponseInputItemUnionParam{OfReasoning: &p}, nil
	case responses.ResponseFunctionToolCall:
		p := v.ToParam()
		return responses.ResponseInputItemUnionParam{OfFunctionCall: &p}, nil
	case responses.ResponseFunctionWebSearch:
		p := v.ToParam()
		return responses.ResponseInputItemUnionParam{OfWebSearchCall: &p}, nil
	default:
		return responses.ResponseInputItemUnionParam{}, fmt.Errorf("unsupported OpenAI output item type %q", item.Type)
	}
}

// openAILimit returns err as a *LimitError when the server reported a limit:
// as an HTTP error, or as a stream event the SDK stopped at because it has an
// "error" field (OpenRouter sends those as "error" and "response.error").
func openAILimit(err error) error {
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		_, codes, message := errorFields([]byte(apiErr.RawJSON()))
		var header http.Header
		if apiErr.Response != nil {
			header = apiErr.Response.Header
		}
		return limitError(err, apiErr.StatusCode, header, codes, message)
	}
	var streamErr *ssestream.StreamError
	if errors.As(err, &streamErr) {
		return bodyLimit(err, streamErr.Event.Data)
	}
	return err
}
