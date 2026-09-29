package crux

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
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
func withoutForeignOpenAIItems(log []Entry, provider Provider) []Entry {
	foreign := func(e Entry) bool {
		p, ok := e.Opaque[openAIProviderOpaqueKey]
		return ok && Provider(p) != provider
	}
	if !slices.ContainsFunc(log, foreign) {
		return log
	}
	out := make([]Entry, 0, len(log))
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

func (a *Agent) newOpenAIClient(httpClient *http.Client) *openai.Client {
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
	} else if a.provider != ProviderOpenAI {
		// Construct the Responses service directly to avoid inheriting OpenAI
		// credentials, organization, project, or custom headers from the environment.
		opts = append(opts, option.WithHTTPClient(defaultHTTPClient()))
	}
	if a.provider != ProviderOpenAI {
		return &openai.Client{Options: opts, Responses: responses.NewResponseService(opts...)}
	}

	c := openai.NewClient(opts...)
	return &c
}

// openAIstep sends the log and returns the model's entries with usage attached.
func (a *Agent) openAIstep(ctx context.Context, log []Entry, httpClient *http.Client, emit chunkSink) ([]Entry, error) {
	input, err := toOpenAIResponseInput(withoutForeignOpenAIItems(log, a.provider))
	if err != nil {
		return nil, err
	}

	params := responses.ResponseNewParams{
		Model: openai.ResponsesModel(a.model),
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: input},
		Tools: openAITools(a.tools),
		Store: openai.Bool(false),
	}

	switch a.provider {
	case ProviderOpenAI, ProviderOpenrouter, ProviderXAI:
		// Nothing is kept server side, so reasoning travels with the log.
		params.Include = []responses.ResponseIncludable{responses.ResponseIncludableReasoningEncryptedContent}
	}
	if a.instructions != "" {
		params.Instructions = openai.String(a.instructions)
	}
	if a.maxTokens > 0 {
		params.MaxOutputTokens = openai.Int(int64(a.maxTokens))
	}
	if a.temperature != nil {
		params.Temperature = openai.Float(*a.temperature)
	}
	if a.searchOptions != nil {
		tool := responses.ToolParamOfWebSearch(responses.WebSearchToolTypeWebSearch)
		if location := a.searchOptions.UserLocation; location != nil && a.provider == ProviderOpenAI {
			if location.Country != "" {
				tool.OfWebSearch.UserLocation.Country = openai.String(location.Country)
			}
			if location.City != "" {
				tool.OfWebSearch.UserLocation.City = openai.String(location.City)
			}
			if location.Region != "" {
				tool.OfWebSearch.UserLocation.Region = openai.String(location.Region)
			}
			if location.Timezone != "" {
				tool.OfWebSearch.UserLocation.Timezone = openai.String(location.Timezone)
			}
			if location.Country != "" || location.City != "" || location.Region != "" || location.Timezone != "" {
				tool.OfWebSearch.UserLocation.Type = "approximate"
			}
		}
		params.Tools = append(params.Tools, tool)
	}
	if a.outputSchema != nil {
		schema, err := wireSchemaFor(a.outputSchema, a.provider)
		if err != nil {
			return nil, err
		}

		params.Text = responses.ResponseTextConfigParam{
			Format: responses.ResponseFormatTextConfigUnionParam{
				OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
					Name:   "output",
					Schema: schema,
					Strict: openai.Bool(true),
				},
			},
		}
	}

	client := a.newOpenAIClient(httpClient)
	var response *responses.Response
	if emit == nil {
		response, err = client.Responses.New(ctx, params)
	} else {
		response, err = streamOpenAI(ctx, client, params, emit)
	}
	if err != nil {
		return nil, fmt.Errorf("%s responses: %w", a.provider, err)
	}
	// Scan all messages before converting items or executing any local tools.
	for _, item := range response.Output {
		if message, ok := item.AsAny().(responses.ResponseOutputMessage); ok {
			for _, part := range message.Content {
				if refusal, ok := part.AsAny().(responses.ResponseOutputRefusal); ok {
					if refusal.Refusal != "" {
						return nil, fmt.Errorf("%s: %w: %s", a.provider, ErrRefused, refusal.Refusal)
					}
					return nil, fmt.Errorf("%s: %w", a.provider, ErrRefused)
				}
			}
		}
	}
	if response.Error.Code != "" || response.Status == responses.ResponseStatusFailed {
		return nil, fmt.Errorf("%s response failed: %s: %s", a.provider, response.Error.Code, response.Error.Message)
	}
	if response.Status != responses.ResponseStatusCompleted {
		// incomplete_details.reason: https://platform.openai.com/docs/api-reference/responses/object
		if response.IncompleteDetails.Reason == "content_filter" {
			return nil, fmt.Errorf("%s: %w: content_filter", a.provider, ErrRefused)
		}
		if response.IncompleteDetails.Reason == "max_output_tokens" {
			return nil, fmt.Errorf("%s response hit the output token limit; raise it with crux.WithMaxTokens", a.provider)
		}
		return nil, fmt.Errorf("%s response did not complete: status %q, reason %q", a.provider, response.Status, response.IncompleteDetails.Reason)
	}

	now := time.Now().UTC()
	produced := make([]Entry, 0, len(response.Output))
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
		entry.Opaque[openAIProviderOpaqueKey] = []byte(a.provider)
		produced = append(produced, entry)
	}

	if !hasAnswerOrCall(produced) {
		// A completed response with no output is an empty final answer; its
		// usage must not be lost. https://platform.openai.com/docs/api-reference/responses/object
		produced = append(produced, Entry{At: now, Kind: KindAssistant})
	}

	// Usage is reported per response, so it hangs on the last thing the model
	// produced rather than being spread over the entries. InputTokens already
	// includes cached tokens: https://platform.openai.com/docs/guides/prompt-caching
	usage := response.Usage
	produced[len(produced)-1].Usage = &Usage{
		InputTokens:      int(usage.InputTokens),
		OutputTokens:     int(usage.OutputTokens),
		CacheReadTokens:  int(usage.InputTokensDetails.CachedTokens),
		CacheWriteTokens: int(usage.InputTokensDetails.CacheWriteTokens),
	}

	return produced, nil
}

func streamOpenAI(ctx context.Context, client *openai.Client, params responses.ResponseNewParams, emit chunkSink) (*responses.Response, error) {
	stream := client.Responses.NewStreaming(ctx, params)
	defer stream.Close()
	var response *responses.Response
	for stream.Next() {
		event := stream.Current()
		var err error
		switch event.Type {
		case "response.output_text.delta":
			err = emitChunk(emit, ChunkText, event.Delta)
		case "response.reasoning_text.delta", "response.reasoning_summary_text.delta":
			err = emitChunk(emit, ChunkReasoning, event.Delta)
		case "response.completed", "response.failed", "response.incomplete":
			value := event.Response
			response = &value
		case "error":
			return nil, fmt.Errorf("provider stream error: %s", event.RawJSON())
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
	if response == nil {
		return nil, errors.New("provider stream ended without a terminal response")
	}
	return response, nil
}

func openAITools(tools []Tool) []responses.ToolUnionParam {
	params := make([]responses.ToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		params = append(params, responses.ToolUnionParam{
			OfFunction: &responses.FunctionToolParam{
				Name:        tool.name,
				Description: openai.String(tool.description),
				Parameters:  tool.schema,
				Strict:      openai.Bool(false),
			},
		})
	}
	return params
}

func fromOpenAIResponseOutputItemUnion(item responses.ResponseOutputItemUnion) (Entry, error) {
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
		return Entry{
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
		return Entry{
			Kind: KindReasoning,
			Reasoning: &Reasoning{
				Summary: strings.Join(summary, "\n\n"),
			},
			Opaque: opaque,
		}, nil
	case responses.ResponseFunctionWebSearch:
		return Entry{Kind: KindProviderTool, Opaque: opaque}, nil
	case responses.ResponseFunctionToolCall:
		return Entry{
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
		return Entry{Kind: KindProviderTool, Opaque: map[string][]byte{openAIUnknownOutputItemOpaqueKey: []byte(item.RawJSON())}}, nil
	}
}

// toOpenAIResponseInput renders a log as the input of the next request.
func toOpenAIResponseInput(log []Entry) (responses.ResponseInputParam, error) {
	input := make(responses.ResponseInputParam, 0, len(log))

	for _, e := range log {
		if e.Kind == KindProviderTool && len(e.Opaque[openAIOutputItemOpaqueKey]) == 0 {
			continue
		}
		if e.HiddenFromModel() {
			continue // tool-written state or approval event, never shown to the model
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

// toOpenAIResponseInputItemUnionParam renders an entry as an input item.
func toOpenAIResponseInputItemUnionParam(e Entry) (responses.ResponseInputItemUnionParam, error) {
	if raw, ok := e.Opaque[openAIOutputItemOpaqueKey]; ok && len(raw) > 0 {
		return openAIInputItemFromOutputItem(raw)
	}

	switch e.Kind {
	case KindUser:
		return responses.ResponseInputItemUnionParam{
			OfMessage: &responses.EasyInputMessageParam{
				Role: responses.EasyInputMessageRoleUser,
				Content: responses.EasyInputMessageContentUnionParam{
					OfString: param.NewOpt(e.Text()),
				},
			},
		}, nil
	case KindAssistant:
		content := make([]responses.ResponseOutputMessageContentUnionParam, 0, len(e.Content))
		for _, part := range e.Content {
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
	case KindToolCall:
		if e.ToolCall == nil {
			return responses.ResponseInputItemUnionParam{}, errors.New("tool call entry carries no tool call")
		}
		arguments := e.ToolCall.rawArgs()
		if arguments == "" {
			arguments = "{}"
		}
		return responses.ResponseInputItemUnionParam{
			OfFunctionCall: &responses.ResponseFunctionToolCallParam{
				CallID:    e.ToolCall.ID,
				Name:      e.ToolCall.Name,
				Arguments: arguments,
				Status:    responses.ResponseFunctionToolCallStatusCompleted,
			},
		}, nil
	case KindToolResult:
		if e.ToolResult == nil {
			return responses.ResponseInputItemUnionParam{}, errors.New("tool result entry carries no tool result")
		}
		// A failed or declined call is still owed an output.
		output := e.ToolResult.Output
		if e.ToolResult.Error != "" {
			output = e.ToolResult.Error
		}
		return responses.ResponseInputItemUnionParam{
			OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
				CallID: param.NewOpt(e.ToolResult.CallID),
				Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
					OfString: param.NewOpt(output),
				},
			},
		}, nil
	default:
		return responses.ResponseInputItemUnionParam{}, fmt.Errorf("unsupported entry kind %d for OpenAI input item", e.Kind)
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
