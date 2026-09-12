package crux

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

const openAIOutputItemOpaqueKey = "openai.response.output_item"

// openAIResponseError retains unsuccessful response metadata for adapters that
// interpret provider-specific fields after the shared step returns.
type openAIResponseError struct {
	Provider Provider
	Response *responses.Response
}

func (e *openAIResponseError) Error() string {
	response := e.Response
	if response.Error.Code != "" || response.Status == responses.ResponseStatusFailed {
		return fmt.Sprintf("%s response failed: %s: %s", e.Provider, response.Error.Code, response.Error.Message)
	}
	return fmt.Sprintf("%s response did not complete: status %q, reason %q", e.Provider, response.Status, response.IncompleteDetails.Reason)
}

func (a *Agent) newOpenAIClient() *openai.Client {
	var opts []option.RequestOption
	if a.apikey != "" {
		opts = append(opts, option.WithAPIKey(a.apikey))
	}
	if a.baseURL != "" {
		opts = append(opts, option.WithBaseURL(a.baseURL))
	}

	client := openai.NewClient(opts...)
	return &client
}

// openAIstep sends the log and returns the model's entries with usage attached.
func (a *Agent) openAIstep(ctx context.Context, log []Entry) ([]Entry, error) {
	input, err := toOpenAIResponseInput(log)
	if err != nil {
		return nil, err
	}

	params := responses.ResponseNewParams{
		Model: openai.ResponsesModel(a.model),
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: input},
		Tools: openAITools(a.toolsRegistry, a.allowedTools),
		// Nothing is kept server side, so reasoning has to travel with the log.
		Store:   openai.Bool(false),
		Include: []responses.ResponseIncludable{responses.ResponseIncludableReasoningEncryptedContent},
	}
	if a.instructions != "" {
		params.Instructions = openai.String(a.instructions)
	}
	if a.enableWebsearch {
		params.Tools = append(params.Tools, responses.ToolParamOfWebSearch(responses.WebSearchToolTypeWebSearch))
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

	response, err := a.openai.Responses.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("openai responses: %w", err)
	}
	// Scan all messages before converting items or executing any local tools.
	for _, item := range response.Output {
		if message, ok := item.AsAny().(responses.ResponseOutputMessage); ok {
			for _, part := range message.Content {
				if refusal, ok := part.AsAny().(responses.ResponseOutputRefusal); ok {
					return nil, &RefusalError{
						Provider: a.provider, Model: a.model,
						Reason: "refusal", Message: refusal.Refusal,
					}
				}
			}
		}
	}
	if response.Status == responses.ResponseStatusIncomplete && response.IncompleteDetails.Reason == "content_filter" {
		return nil, &RefusalError{Provider: a.provider, Model: a.model, Reason: "content_filter"}
	}
	if response.Error.Code != "" || response.Status != responses.ResponseStatusCompleted {
		return nil, &openAIResponseError{Provider: a.provider, Response: response}
	}

	produced := make([]Entry, 0, len(response.Output))
	for _, item := range response.Output {
		entry, err := fromOpenAIResponseOutputItemUnion(item)
		if err != nil {
			return nil, err
		}
		produced = append(produced, entry)
	}

	// Usage is reported per response, so it hangs on the last thing the model
	// produced rather than being spread over the entries.
	if len(produced) > 0 {
		produced[len(produced)-1].Usage = &Usage{
			InputTokens:  int(response.Usage.InputTokens),
			OutputTokens: int(response.Usage.OutputTokens),
		}
	}

	return produced, nil
}

func openAITools(registry *ToolsRegistry, allowedTools []string) []responses.ToolUnionParam {
	tools := registry.selected(allowedTools)

	params := make([]responses.ToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		params = append(params, responses.ToolUnionParam{
			OfFunction: &responses.FunctionToolParam{
				Name:        tool.name,
				Description: openai.String(tool.description),
				Parameters:  tool.schema,
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
		var summary strings.Builder
		for _, part := range v.Summary {
			summary.WriteString(part.Text)
		}
		return Entry{
			Kind: KindReasoning,
			Reasoning: &Reasoning{
				Summary: summary.String(),
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
				Args: json.RawMessage(v.Arguments),
			},
			Opaque: opaque,
		}, nil
	default:
		return Entry{}, fmt.Errorf("unsupported OpenAI output item type %T", v)
	}

}

// toOpenAIResponseInput renders a log as the input of the next request.
func toOpenAIResponseInput(log []Entry) (responses.ResponseInputParam, error) {
	input := make(responses.ResponseInputParam, 0, len(log))

	for _, e := range log {
		if e.Kind == KindProviderTool && len(e.Opaque[openAIOutputItemOpaqueKey]) == 0 {
			continue
		}
		if e.Kind == KindStateDelta {
			continue // tool-written state, never shown to the model
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
	case KindReasoning:
		item := responses.ResponseReasoningItemParam{
			Status: responses.ResponseReasoningItemStatusCompleted,
		}
		if e.Reasoning != nil && e.Reasoning.Summary != "" {
			item.Summary = []responses.ResponseReasoningItemSummaryParam{
				{Text: e.Reasoning.Summary},
			}
		}
		return responses.ResponseInputItemUnionParam{OfReasoning: &item}, nil
	case KindToolCall:
		if e.ToolCall == nil {
			return responses.ResponseInputItemUnionParam{}, errors.New("tool call entry carries no tool call")
		}
		arguments := string(e.ToolCall.Args)
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
		return responses.ResponseInputItemUnionParam{}, fmt.Errorf("unsupported OpenAI output item type %T", v)
	}
}
