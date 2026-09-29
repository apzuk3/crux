package crux

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strings"
	"time"

	"google.golang.org/genai"
)

const geminiPartOpaqueKey = "gemini.content.part"

// Candidate metadata is retained for sources/display, not replayed as a part.
const geminiGroundingMetadataOpaqueKey = "gemini.candidate.grounding_metadata"

func (a *Agent) newGeminiClient(ctx context.Context, httpClient *http.Client) (*genai.Client, error) {
	config := &genai.ClientConfig{Backend: genai.BackendGeminiAPI}
	if a.apiKey != "" {
		config.APIKey = a.apiKey
	}
	if a.baseURL != "" {
		config.HTTPOptions.BaseURL = a.baseURL
	}
	client := a.effectiveHTTPClient(httpClient)
	if client == nil {
		// genai's own default client has no timeout at all.
		client = defaultHTTPClient()
	}
	config.HTTPClient = client
	return genai.NewClient(ctx, config)
}

// geminiStep returns the model's ordered parts with usage attached.
func (a *Agent) geminiStep(ctx context.Context, log []Entry, httpClient *http.Client, emit chunkSink) ([]Entry, error) {
	contents, err := toGeminiContents(log)
	if err != nil {
		return nil, err
	}
	tools, err := geminiTools(a.tools)
	if err != nil {
		return nil, err
	}
	config := &genai.GenerateContentConfig{Tools: tools}
	if a.maxTokens > 0 {
		config.MaxOutputTokens = int32(min(a.maxTokens, math.MaxInt32))
	}
	if a.temperature != nil {
		temperature := float32(*a.temperature)
		config.Temperature = &temperature
	}
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
		schema, err := wireSchemaFor(a.outputSchema, a.provider)
		if err != nil {
			return nil, err
		}
		config.ResponseMIMEType = "application/json"
		config.ResponseJsonSchema = schema
	}
	// The SDK constructor can fail, so initialization errors flow through Run.
	client, err := a.newGeminiClient(ctx, httpClient)
	if err != nil {
		return nil, fmt.Errorf("gemini client: %w", err)
	}
	var response *genai.GenerateContentResponse
	if emit == nil {
		response, err = client.Models.GenerateContent(ctx, a.model, contents, config)
	} else {
		response, err = streamGemini(ctx, client, a.model, contents, config, emit)
	}
	if err != nil {
		return nil, fmt.Errorf("gemini generate content: %w", err)
	}
	if len(response.Candidates) == 0 {
		// A blocked prompt has no candidates, only prompt feedback:
		// https://ai.google.dev/api/generate-content#BlockReason
		if feedback := response.PromptFeedback; feedback != nil && feedback.BlockReason != "" {
			return nil, fmt.Errorf("gemini: %w: prompt blocked: %s", ErrRefused, feedback.BlockReason)
		}
		return nil, errors.New("gemini returned no candidates")
	}
	candidate := response.Candidates[0]
	// Finish reasons: https://ai.google.dev/api/generate-content#FinishReason
	switch candidate.FinishReason {
	case genai.FinishReasonStop:
	case genai.FinishReasonMaxTokens:
		return nil, errors.New("gemini response hit the output token limit; raise it with crux.WithMaxTokens")
	case genai.FinishReasonSafety, genai.FinishReasonProhibitedContent, genai.FinishReasonBlocklist,
		genai.FinishReasonSPII, genai.FinishReasonRecitation, genai.FinishReasonImageSafety,
		genai.FinishReasonImageProhibitedContent, genai.FinishReasonImageRecitation:
		return nil, fmt.Errorf("gemini: %w: finish reason %s", ErrRefused, candidate.FinishReason)
	default:
		return nil, fmt.Errorf("gemini response did not complete: finish reason %q", candidate.FinishReason)
	}
	now := time.Now().UTC()
	var parts []*genai.Part
	if candidate.Content != nil {
		parts = candidate.Content.Parts
	}
	produced := make([]Entry, 0, len(parts)+1)
	for i, part := range parts {
		if geminiPartEmpty(part) {
			continue
		}
		entry, err := fromGeminiPart(part)
		if err != nil {
			return nil, err
		}
		if entry.At.IsZero() {
			entry.At = now
		}
		// Gemini can omit call IDs. Keep a local reference without changing
		// the original part replayed to the API.
		if entry.ToolCall != nil && entry.ToolCall.ID == "" {
			entry.ToolCall.ID = fmt.Sprintf("gemini-call-%d-%d", len(log), i)
		}
		produced = append(produced, entry)
	}
	if !hasAnswerOrCall(produced) {
		// The model may stop (finish reason STOP) without saying anything;
		// that is still a final answer, and its usage must not be lost.
		// https://ai.google.dev/api/generate-content#FinishReason
		produced = append(produced, Entry{At: now, Kind: KindAssistant})
	}
	if candidate.GroundingMetadata != nil {
		raw, err := json.Marshal(candidate.GroundingMetadata)
		if err != nil {
			return nil, fmt.Errorf("marshal Gemini grounding metadata: %w", err)
		}
		last := &produced[len(produced)-1]
		if last.Opaque == nil {
			last.Opaque = make(map[string][]byte)
		}
		last.Opaque[geminiGroundingMetadataOpaqueKey] = raw
	}
	if usage := response.UsageMetadata; usage != nil {
		var cacheRead int
		if usage.CachedContentTokenCount > 0 {
			cacheRead = int(usage.CachedContentTokenCount)
		}
		// PromptTokenCount already includes cached content; tool-use prompt
		// tokens (such as search results) are counted separately.
		// https://ai.google.dev/api/generate-content#UsageMetadata
		produced[len(produced)-1].Usage = &Usage{
			InputTokens:     int(usage.PromptTokenCount + usage.ToolUsePromptTokenCount),
			OutputTokens:    int(usage.CandidatesTokenCount + usage.ThoughtsTokenCount),
			CacheReadTokens: cacheRead,
		}
	}
	return produced, nil
}

func streamGemini(ctx context.Context, client *genai.Client, model string, contents []*genai.Content, config *genai.GenerateContentConfig, emit chunkSink) (*genai.GenerateContentResponse, error) {
	// Preserve streamed parts verbatim, including signature-only parts and tool
	// calls. Joining their text for display must not discard replay metadata.
	candidate := &genai.Candidate{Content: &genai.Content{Role: "model"}}
	response := &genai.GenerateContentResponse{Candidates: []*genai.Candidate{candidate}}
	sawCandidate := false
	for chunk, err := range client.Models.GenerateContentStream(ctx, model, contents, config) {
		if err != nil {
			return nil, err
		}
		if chunk == nil {
			return nil, errors.New("gemini returned a nil stream chunk")
		}
		if chunk.UsageMetadata != nil {
			response.UsageMetadata = chunk.UsageMetadata
		}
		if chunk.PromptFeedback != nil {
			response.PromptFeedback = chunk.PromptFeedback
		}
		if len(chunk.Candidates) == 0 {
			continue
		}
		sawCandidate = true
		current := chunk.Candidates[0]
		if current.FinishReason != "" {
			candidate.FinishReason = current.FinishReason
		}
		if current.GroundingMetadata != nil {
			candidate.GroundingMetadata = current.GroundingMetadata
		}
		if current.Content == nil {
			continue
		}
		for _, part := range current.Content.Parts {
			if part == nil {
				return nil, errors.New("gemini returned a nil part")
			}
			if geminiPartEmpty(part) {
				continue
			}
			candidate.Content.Parts = append(candidate.Content.Parts, part)
			kind := ChunkText
			if part.Thought {
				kind = ChunkReasoning
			}
			if err := emitChunk(emit, kind, part.Text); err != nil {
				return nil, err
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !sawCandidate {
		// A blocked prompt yields only prompt feedback.
		// https://ai.google.dev/api/generate-content#BlockReason
		response.Candidates = nil
	}
	return response, nil
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
		// Other data, such as executable code or inline media, is kept and
		// replayed verbatim. Part types: https://ai.google.dev/api/caching#Part
		return Entry{Kind: KindProviderTool, Opaque: opaque}, nil
	}
}

// geminiPartEmpty reports whether a part carries nothing at all, like the
// empty text parts Gemini sometimes sends alongside real content (often the
// last streamed chunk, which only carries the finish reason and usage).
// https://ai.google.dev/api/caching#Part
func geminiPartEmpty(part *genai.Part) bool {
	if part == nil {
		return false
	}
	raw, err := json.Marshal(part)
	if err != nil {
		return false
	}
	empty, _ := json.Marshal(&genai.Part{})
	return bytes.Equal(raw, empty)
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

// geminiStandInSignature is the documented "skip_thought_signature_validator"
// value. The field is bytes sent as base64, so it holds that string decoded;
// Go encodes it with the standard alphabet ('/' for '_'), which the API
// decodes to the same bytes.
var geminiStandInSignature, _ = base64.URLEncoding.DecodeString("skip_thought_signature_validator")

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
			if strings.TrimSpace(part.Text) == "" {
				continue // Gemini rejects empty text parts.
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
			if err := json.Unmarshal(call.objectArgs(), &args); err != nil {
				return nil, fmt.Errorf("decode Gemini function arguments: %w", err)
			}
		}
		// History from another provider has no thought signature, which
		// Gemini 3 requires on function calls; this is the documented stand-in:
		// https://ai.google.dev/gemini-api/docs/thought-signatures
		parts = []*genai.Part{{
			FunctionCall:     &genai.FunctionCall{ID: call.ID, Name: call.Name, Args: args},
			ThoughtSignature: geminiStandInSignature,
		}}
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
