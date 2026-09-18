package crux

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"google.golang.org/genai"
)

const geminiPartOpaqueKey = "gemini.content.part"

// Candidate metadata is retained for sources/display, not replayed as a part.
const geminiGroundingMetadataOpaqueKey = "gemini.candidate.grounding_metadata"

func (a *Agent) newGeminiClient(ctx context.Context) (*genai.Client, error) {
	config := &genai.ClientConfig{Backend: genai.BackendGeminiAPI}
	if a.apikey != "" {
		config.APIKey = a.apikey
	}
	if a.baseURL != "" {
		config.HTTPOptions.BaseURL = a.baseURL
	}
	return genai.NewClient(ctx, config)
}

// geminiStep returns the model's ordered parts with usage attached.
func (a *Agent) geminiStep(ctx context.Context, log []Entry) ([]Entry, error) {
	contents, err := toGeminiContents(log)
	if err != nil {
		return nil, err
	}
	tools, err := geminiTools(a.tools)
	if err != nil {
		return nil, err
	}
	config := &genai.GenerateContentConfig{Tools: tools}
	if a.searchOptions != nil {
		config.Tools = append(config.Tools, &genai.Tool{GoogleSearch: &genai.GoogleSearch{}})
		if location := a.searchOptions.UserLocation; location != nil && location.Latitude != nil && location.Longitude != nil {
			config.ToolConfig = &genai.ToolConfig{
				RetrievalConfig: &genai.RetrievalConfig{
					LatLng: &genai.LatLng{Latitude: location.Latitude, Longitude: location.Longitude},
				},
			}
		}
	}
	if a.instructions != "" {
		config.SystemInstruction = &genai.Content{Parts: []*genai.Part{genai.NewPartFromText(a.instructions)}}
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
		config.ResponseMIMEType = "application/json"
		config.ResponseJsonSchema = schema
	}
	// The SDK constructor can fail, so initialization errors flow through Run.
	client, err := a.newGeminiClient(ctx)
	if err != nil {
		return nil, fmt.Errorf("gemini client: %w", err)
	}
	response, err := client.Models.GenerateContent(ctx, a.model, contents, config)
	if err != nil {
		return nil, fmt.Errorf("gemini generate content: %w", err)
	}
	if len(response.Candidates) == 0 {
		return nil, errors.New("gemini returned no candidates")
	}
	candidate := response.Candidates[0]
	if candidate.FinishReason != genai.FinishReasonStop {
		return nil, fmt.Errorf("gemini response did not complete: finish reason %q", candidate.FinishReason)
	}
	if candidate.Content == nil || len(candidate.Content.Parts) == 0 {
		return nil, errors.New("gemini returned no content")
	}
	produced := make([]Entry, 0, len(candidate.Content.Parts))
	for i, part := range candidate.Content.Parts {
		entry, err := fromGeminiPart(part)
		if err != nil {
			return nil, err
		}
		// Gemini can omit call IDs. Keep a local reference without changing
		// the original part replayed to the API.
		if entry.ToolCall != nil && entry.ToolCall.ID == "" {
			entry.ToolCall.ID = fmt.Sprintf("gemini-call-%d-%d", len(log), i)
		}
		produced = append(produced, entry)
	}
	if candidate.GroundingMetadata != nil {
		raw, err := json.Marshal(candidate.GroundingMetadata)
		if err != nil {
			return nil, fmt.Errorf("marshal Gemini grounding metadata: %w", err)
		}
		produced[len(produced)-1].Opaque[geminiGroundingMetadataOpaqueKey] = raw
	}
	if usage := response.UsageMetadata; usage != nil {
		produced[len(produced)-1].Usage = &Usage{
			InputTokens:  int(usage.PromptTokenCount),
			OutputTokens: int(usage.CandidatesTokenCount + usage.ThoughtsTokenCount),
		}
	}
	return produced, nil
}

func geminiTools(selected []Tool) ([]*genai.Tool, error) {
	if len(selected) == 0 {
		return nil, nil
	}
	declarations := make([]*genai.FunctionDeclaration, 0, len(selected))
	for _, tool := range selected {
		if tool.schema["type"] != "object" {
			return nil, fmt.Errorf("gemini function %q requires an object parameter schema", tool.name)
		}
		declarations = append(declarations, &genai.FunctionDeclaration{
			Name: tool.name, Description: tool.description, ParametersJsonSchema: tool.schema,
		})
	}
	return []*genai.Tool{{FunctionDeclarations: declarations}}, nil
}

func fromGeminiPart(part *genai.Part) (Entry, error) {
	if part == nil {
		return Entry{}, errors.New("gemini returned a nil part")
	}
	raw, err := json.Marshal(part)
	if err != nil {
		return Entry{}, fmt.Errorf("marshal Gemini part: %w", err)
	}
	opaque := map[string][]byte{geminiPartOpaqueKey: raw}
	switch {
	case part.FunctionCall != nil:
		call := part.FunctionCall
		args := json.RawMessage(`{}`)
		if call.Args != nil {
			args, err = json.Marshal(call.Args)
			if err != nil {
				return Entry{}, fmt.Errorf("marshal Gemini function arguments: %w", err)
			}
		}
		return Entry{Kind: KindToolCall, ToolCall: &ToolCall{ID: call.ID, Name: call.Name, Args: args}, Opaque: opaque}, nil
	case part.Thought:
		return Entry{Kind: KindReasoning, Reasoning: &Reasoning{Summary: part.Text}, Opaque: opaque}, nil
	case part.Text != "":
		return Entry{Kind: KindAssistant, Content: []ContentPart{{Kind: ContentKindText, Text: part.Text}}, Opaque: opaque}, nil
	case len(part.ThoughtSignature) > 0:
		return Entry{Kind: KindReasoning, Reasoning: &Reasoning{}, Opaque: opaque}, nil
	default:
		return Entry{}, errors.New("unsupported Gemini part")
	}
}

// toGeminiContents groups consecutive parts by role.
func toGeminiContents(log []Entry) ([]*genai.Content, error) {
	var contents []*genai.Content
	calls := make(map[string]*genai.FunctionCall)
	for _, e := range log {
		if e.Kind == KindProviderTool && len(e.Opaque[geminiPartOpaqueKey]) == 0 {
			continue
		}
		if e.HiddenFromModel() {
			continue
		}
		parts, err := toGeminiParts(e, calls)
		if err != nil {
			return nil, err
		}
		if len(parts) == 0 {
			continue
		}
		if e.Kind == KindToolCall && e.ToolCall != nil {
			calls[e.ToolCall.ID] = parts[0].FunctionCall
		}
		role := "model"
		if e.Kind == KindUser || e.Kind == KindToolResult {
			role = "user"
		}
		if len(contents) == 0 || contents[len(contents)-1].Role != role {
			contents = append(contents, &genai.Content{Role: role})
		}
		last := contents[len(contents)-1]
		last.Parts = append(last.Parts, parts...)
	}
	return contents, nil
}

// toGeminiParts renders an entry as content parts.
func toGeminiParts(e Entry, calls map[string]*genai.FunctionCall) ([]*genai.Part, error) {
	if raw := e.Opaque[geminiPartOpaqueKey]; len(raw) > 0 && e.Kind != KindUser && e.Kind != KindToolResult {
		var part genai.Part
		if err := json.Unmarshal(raw, &part); err != nil {
			return nil, fmt.Errorf("decode Gemini part: %w", err)
		}
		return []*genai.Part{&part}, nil
	}
	var parts []*genai.Part
	switch e.Kind {
	case KindUser, KindAssistant:
		for _, part := range e.Content {
			if part.Kind != ContentKindText && part.Kind != ContentKindRefusal {
				return nil, fmt.Errorf("unsupported content part kind %q", part.Kind)
			}
			parts = append(parts, genai.NewPartFromText(part.Text))
		}
	case KindReasoning:
		// Reasoning without signed provider data cannot be replayed.
	case KindToolCall:
		if e.ToolCall == nil {
			return nil, errors.New("tool call entry carries no tool call")
		}
		call := e.ToolCall
		args := make(map[string]any)
		if len(call.Args) > 0 {
			if err := json.Unmarshal(call.Args, &args); err != nil {
				return nil, fmt.Errorf("decode Gemini function arguments: %w", err)
			}
		}
		parts = []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: call.ID, Name: call.Name, Args: args}}}
	case KindToolResult:
		if e.ToolResult == nil {
			return nil, errors.New("tool result entry carries no tool result")
		}
		r := e.ToolResult
		call := calls[r.CallID]
		if call == nil {
			return nil, fmt.Errorf("Gemini function response has no matching call %q", r.CallID)
		}
		response := map[string]any{"output": r.Output}
		if r.Error != "" {
			response = map[string]any{"error": r.Error}
		}
		parts = []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: call.ID, Name: call.Name, Response: response}}}
	default:
		return nil, fmt.Errorf("unsupported entry kind %d for Gemini part", e.Kind)
	}
	return parts, nil
}
