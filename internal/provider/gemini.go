package provider

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	"google.golang.org/genai"
)

const geminiPartOpaqueKey = "gemini.content.part"

// Candidate metadata is retained for sources/display, not replayed as a part.
const geminiGroundingMetadataOpaqueKey = "gemini.candidate.grounding_metadata"

func newGeminiClient(ctx context.Context, req *Request) (*genai.Client, error) {
	config := &genai.ClientConfig{Backend: genai.BackendGeminiAPI}
	if key := req.apiKey(); key != "" {
		config.APIKey = key
	}
	if req.BaseURL != "" {
		config.HTTPOptions.BaseURL = req.BaseURL
	}
	client := req.HTTPClient
	if client == nil {
		// genai's own default client has no timeout at all.
		client = DefaultHTTPClient()
	}
	config.HTTPClient = client
	// genai retries only when asked to, and its backoff ignores the delay the
	// server asks for. It retries connection errors and plain server errors
	// here, as the OpenAI and Anthropic SDKs do; rate limits and overloads
	// (429, 503) come back at once and are retried by the caller, which waits
	// for the delay in the error's RetryInfo.
	config.HTTPOptions.RetryOptions = &genai.HTTPRetryOptions{
		Attempts:        genai.Ptr(int32(req.Retries() + 1)),
		InitialDelay:    genai.Ptr(0.5),
		MaxDelay:        genai.Ptr(8.0),
		HTTPStatusCodes: []int32{http.StatusRequestTimeout, http.StatusInternalServerError, http.StatusBadGateway, http.StatusGatewayTimeout},
	}
	return genai.NewClient(ctx, config)
}

// geminiFunctionCalling maps the tool choice onto the function calling mode.
// Gemini has no setting for parallel calls; crux rejects one when the agent
// is created.
// geminiToolConfig returns the request's tool config, creating it once.
func geminiToolConfig(config *genai.GenerateContentConfig) *genai.ToolConfig {
	if config.ToolConfig == nil {
		config.ToolConfig = &genai.ToolConfig{}
	}
	return config.ToolConfig
}

func geminiFunctionCalling(req *Request) *genai.FunctionCallingConfig {
	if len(req.Tools) == 0 {
		return nil
	}
	switch req.ToolChoice {
	case ToolChoiceAuto:
		return &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAuto}
	case ToolChoiceRequired:
		return &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAny}
	case ToolChoiceNone:
		return &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeNone}
	case ToolChoiceTool:
		return &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAny, AllowedFunctionNames: []string{req.ToolName}}
	}
	return nil
}

// Gemini returns the model's ordered parts with usage attached.
func Gemini(ctx context.Context, req *Request, emit Emit) ([]Item, error) {
	contents, err := toGeminiContents(req.Log)
	if err != nil {
		return nil, err
	}
	config, err := geminiConfig(req)
	if err != nil {
		return nil, err
	}
	// The SDK constructor can fail, so initialization errors flow through Run.
	client, err := newGeminiClient(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("gemini client: %w", err)
	}
	var response *genai.GenerateContentResponse
	if emit == nil {
		response, err = client.Models.GenerateContent(ctx, req.Model, contents, config)
	} else {
		response, err = streamGemini(ctx, client, req.Model, contents, config, emit)
	}
	if err != nil {
		return nil, fmt.Errorf("gemini generate content: %w", geminiLimit(err))
	}
	candidate, err := geminiCandidate(response)
	if err != nil {
		return nil, err
	}
	now := time.Now().UTC()
	produced, err := geminiItems(candidate, req.Seq, now)
	if err != nil {
		return nil, err
	}
	produced = ensureAnswer(produced, now)
	last := &produced[len(produced)-1]
	if err := attachGrounding(last, candidate.GroundingMetadata); err != nil {
		return nil, err
	}
	if response.ResponseID != "" {
		last.Response = &ResponseInfo{ID: response.ResponseID}
	}
	if usage := response.UsageMetadata; usage != nil {
		last.Usage = geminiUsage(usage)
	}
	return produced, nil
}

// geminiConfig builds the generation config from req.
func geminiConfig(req *Request) (*genai.GenerateContentConfig, error) {
	tools, err := geminiTools(req.Tools)
	if err != nil {
		return nil, err
	}
	config := &genai.GenerateContentConfig{Tools: tools}
	if req.MaxTokens > 0 {
		config.MaxOutputTokens = int32(min(req.MaxTokens, math.MaxInt32))
	}
	if req.Temperature != nil {
		temperature := float32(*req.Temperature)
		config.Temperature = &temperature
	}
	if req.Reasoning != "" {
		config.ThinkingConfig = geminiThinking(req.Model, req.Reasoning)
	}
	if req.Search != nil {
		setGeminiSearch(config, req)
	}
	if req.Instructions != "" {
		config.SystemInstruction = &genai.Content{Parts: []*genai.Part{genai.NewPartFromText(req.Instructions)}}
	}
	if calling := geminiFunctionCalling(req); calling != nil {
		geminiToolConfig(config).FunctionCallingConfig = calling
	}
	if err := applyGeminiOutputSchema(config, req); err != nil {
		return nil, err
	}
	return config, nil
}

// setGeminiSearch adds Google Search. With function tools too, the API needs
// server-side tool invocations in the response (the search calls and results
// come back as parts, kept in Opaque and replayed) or it refuses the request.
func setGeminiSearch(config *genai.GenerateContentConfig, req *Request) {
	config.Tools = append(config.Tools, &genai.Tool{GoogleSearch: &genai.GoogleSearch{}})
	if location := req.Search.Location; location != nil && location.Latitude != nil && location.Longitude != nil {
		geminiToolConfig(config).RetrievalConfig = &genai.RetrievalConfig{
			LatLng: &genai.LatLng{Latitude: location.Latitude, Longitude: location.Longitude},
		}
	}
	if len(req.Tools) > 0 {
		include := true
		geminiToolConfig(config).IncludeServerSideToolInvocations = &include
	}
}

// applyGeminiOutputSchema asks for structured output. Gemini before 3 rejects
// it combined with tools, so the schema then goes into the instructions
// instead; crux validates the answer either way.
// https://ai.google.dev/gemini-api/docs/structured-output
func applyGeminiOutputSchema(config *genai.GenerateContentConfig, req *Request) error {
	schema := req.OutputSchema
	if schema == nil {
		return nil
	}
	if len(config.Tools) == 0 || geminiStructuredOutputWithTools(req.Model) {
		config.ResponseMIMEType = "application/json"
		config.ResponseJsonSchema = schema
		return nil
	}
	raw, err := json.Marshal(schema)
	if err != nil {
		return fmt.Errorf("marshal output schema: %w", err)
	}
	if config.SystemInstruction == nil {
		config.SystemInstruction = &genai.Content{}
	}
	config.SystemInstruction.Parts = append(config.SystemInstruction.Parts,
		genai.NewPartFromText("Your final answer must be only JSON matching this JSON schema: "+string(raw)))
	return nil
}

// geminiCandidate returns the response's candidate when it completed.
func geminiCandidate(response *genai.GenerateContentResponse) (*genai.Candidate, error) {
	if len(response.Candidates) == 0 {
		// A blocked prompt has no candidates, only prompt feedback:
		// https://ai.google.dev/api/generate-content#BlockReason
		if feedback := response.PromptFeedback; feedback != nil && feedback.BlockReason != "" {
			return nil, refused("gemini", fmt.Sprintf("prompt blocked: %s", feedback.BlockReason))
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
		return nil, refused("gemini", fmt.Sprintf("finish reason %s", candidate.FinishReason))
	default:
		return nil, fmt.Errorf("gemini response did not complete: finish reason %q", candidate.FinishReason)
	}
	return candidate, nil
}

// geminiItems converts the candidate's parts, in order.
func geminiItems(candidate *genai.Candidate, seq int, now time.Time) ([]Item, error) {
	var parts []*genai.Part
	if candidate.Content != nil {
		parts = foldGeminiSignatures(candidate.Content.Parts)
	}
	produced := make([]Item, 0, len(parts)+1)
	for i, part := range parts {
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
			entry.ToolCall.ID = fmt.Sprintf("gemini-call-%d-%d", seq, i)
		}
		produced = append(produced, entry)
	}
	return produced, nil
}

// attachGrounding keeps the grounding metadata on the last item.
func attachGrounding(last *Item, metadata *genai.GroundingMetadata) error {
	if metadata == nil {
		return nil
	}
	raw, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("marshal Gemini grounding metadata: %w", err)
	}
	if last.Opaque == nil {
		last.Opaque = make(map[string][]byte)
	}
	last.Opaque[geminiGroundingMetadataOpaqueKey] = raw
	return nil
}

// geminiUsage converts the usage. PromptTokenCount already includes cached
// content; tool-use prompt tokens (such as search results) are counted
// separately. https://ai.google.dev/api/generate-content#UsageMetadata
func geminiUsage(usage *genai.GenerateContentResponseUsageMetadata) *Usage {
	var cacheRead int
	if usage.CachedContentTokenCount > 0 {
		cacheRead = int(usage.CachedContentTokenCount)
	}
	return &Usage{
		InputTokens:     int(usage.PromptTokenCount + usage.ToolUsePromptTokenCount),
		OutputTokens:    int(usage.CandidatesTokenCount + usage.ThoughtsTokenCount),
		CacheReadTokens: cacheRead,
	}
}

// geminiStructuredOutputWithTools reports whether the model accepts a response
// schema together with tools, which Gemini supports from version 3.
func geminiStructuredOutputWithTools(model string) bool {
	return geminiMajorVersion(model) >= 3
}

// geminiMajorVersion returns the major version of a model ID such as
// gemini-2.5-flash, or 0 when the ID has none.
func geminiMajorVersion(model string) int {
	version, ok := strings.CutPrefix(strings.TrimPrefix(model, "models/"), "gemini-")
	if !ok {
		return 0
	}
	end := strings.IndexFunc(version, func(r rune) bool { return r < '0' || r > '9' })
	if end == -1 {
		end = len(version)
	}
	major, _ := strconv.Atoi(version[:end])
	return major
}

// geminiThinking maps WithReasoning onto a thinking level from Gemini 3, or a
// thinking budget on Gemini 2.5, which has no levels. Thought summaries are
// requested so they can stream.
// https://ai.google.dev/gemini-api/docs/thinking
func geminiThinking(model, effort string) *genai.ThinkingConfig {
	if major := geminiMajorVersion(model); major > 0 && major < 3 {
		budgets := map[string]int32{
			ReasoningOff: 0, ReasoningLow: 1024, ReasoningMedium: 8192, ReasoningHigh: 24576, ReasoningMax: 24576,
		}
		return &genai.ThinkingConfig{ThinkingBudget: genai.Ptr(budgets[effort]), IncludeThoughts: effort != ReasoningOff}
	}
	levels := map[string]genai.ThinkingLevel{
		ReasoningOff:    genai.ThinkingLevelMinimal,
		ReasoningLow:    genai.ThinkingLevelLow,
		ReasoningMedium: genai.ThinkingLevelMedium,
		ReasoningHigh:   genai.ThinkingLevelHigh,
		ReasoningMax:    genai.ThinkingLevelHigh,
	}
	return &genai.ThinkingConfig{ThinkingLevel: levels[effort], IncludeThoughts: effort != ReasoningOff}
}

func streamGemini(ctx context.Context, client *genai.Client, model string, contents []*genai.Content, config *genai.GenerateContentConfig, emit Emit) (*genai.GenerateContentResponse, error) {
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
		mergeGeminiChunkMeta(response, chunk)
		if len(chunk.Candidates) == 0 {
			continue
		}
		sawCandidate = true
		if err := mergeGeminiCandidate(candidate, chunk.Candidates[0], emit); err != nil {
			return nil, err
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

// mergeGeminiChunkMeta keeps the response-level fields of a streamed chunk.
func mergeGeminiChunkMeta(response, chunk *genai.GenerateContentResponse) {
	if chunk.UsageMetadata != nil {
		response.UsageMetadata = chunk.UsageMetadata
	}
	if chunk.ResponseID != "" {
		response.ResponseID = chunk.ResponseID
	}
	if chunk.PromptFeedback != nil {
		response.PromptFeedback = chunk.PromptFeedback
	}
}

// mergeGeminiCandidate folds a streamed candidate into the accumulated one
// and emits its parts.
func mergeGeminiCandidate(candidate, current *genai.Candidate, emit Emit) error {
	if current.FinishReason != "" {
		candidate.FinishReason = current.FinishReason
	}
	if current.GroundingMetadata != nil {
		candidate.GroundingMetadata = current.GroundingMetadata
	}
	if current.Content == nil {
		return nil
	}
	for _, part := range current.Content.Parts {
		if part == nil {
			return errors.New("gemini returned a nil part")
		}
		if geminiPartEmpty(part) {
			continue
		}
		appendGeminiPart(candidate.Content, part)
		if err := emitGeminiPart(emit, part); err != nil {
			return err
		}
	}
	return nil
}

// appendGeminiPart adds a streamed part to content. A streamed answer arrives
// as many text parts; they are kept as the one part a non-streamed response
// has, signed by the last.
func appendGeminiPart(content *genai.Content, part *genai.Part) {
	parts := content.Parts
	if n := len(parts); n > 0 && canMergeGeminiText(parts[n-1], part) {
		parts[n-1].Text += part.Text
		parts[n-1].ThoughtSignature = part.ThoughtSignature
		return
	}
	merged := *part
	content.Parts = append(parts, &merged)
}

func emitGeminiPart(emit Emit, part *genai.Part) error {
	kind := ChunkText
	if part.Thought {
		kind = ChunkReasoning
	}
	return emitChunk(emit, kind, part.Text)
}

// foldGeminiSignatures drops empty parts and moves the signature of a part
// that carries no data onto the part before it. The last chunk of a streamed
// answer is often such a part ({"text": "", "thoughtSignature": ...}), and
// Gemini rejects a replayed part without data.
func foldGeminiSignatures(parts []*genai.Part) []*genai.Part {
	folded := make([]*genai.Part, 0, len(parts))
	for _, part := range parts {
		if geminiPartEmpty(part) {
			continue
		}
		if n := len(folded); n > 0 && geminiSignatureOnly(part) && len(folded[n-1].ThoughtSignature) == 0 {
			folded[n-1].ThoughtSignature = part.ThoughtSignature
			continue
		}
		folded = append(folded, part)
	}
	return folded
}

// canMergeGeminiText reports whether next continues the text of prev: both
// carry only text of the same kind, and prev is not yet signed.
func canMergeGeminiText(prev, next *genai.Part) bool {
	if prev.Thought != next.Thought || len(prev.ThoughtSignature) > 0 || prev.Text == "" || next.Text == "" {
		return false
	}
	return geminiTextOnly(prev) && geminiTextOnly(next)
}

func geminiTextOnly(part *genai.Part) bool {
	rest := *part
	rest.Text, rest.Thought, rest.ThoughtSignature = "", false, nil
	return geminiPartEmpty(&rest)
}

func geminiTools(selected []Tool) ([]*genai.Tool, error) {
	if len(selected) == 0 {
		return nil, nil
	}
	declarations := make([]*genai.FunctionDeclaration, 0, len(selected))
	for _, tool := range selected {
		if tool.Schema["type"] != "object" {
			return nil, fmt.Errorf("gemini function %q requires an object parameter schema", tool.Name)
		}
		declarations = append(declarations, &genai.FunctionDeclaration{
			Name: tool.Name, Description: tool.Description, ParametersJsonSchema: tool.Schema,
		})
	}
	return []*genai.Tool{{FunctionDeclarations: declarations}}, nil
}

func fromGeminiPart(part *genai.Part) (Item, error) {
	if part == nil {
		return Item{}, errors.New("gemini returned a nil part")
	}
	raw, err := json.Marshal(part)
	if err != nil {
		return Item{}, fmt.Errorf("marshal Gemini part: %w", err)
	}
	opaque := map[string][]byte{geminiPartOpaqueKey: raw}
	switch {
	case part.FunctionCall != nil:
		call := part.FunctionCall
		args := json.RawMessage(`{}`)
		if call.Args != nil {
			args, err = json.Marshal(call.Args)
			if err != nil {
				return Item{}, fmt.Errorf("marshal Gemini function arguments: %w", err)
			}
		}
		return Item{Kind: KindToolCall, ToolCall: &ToolCall{ID: call.ID, Name: call.Name, Args: args}, Opaque: opaque}, nil
	case part.Thought:
		return Item{Kind: KindReasoning, Reasoning: &Reasoning{Summary: part.Text}, Opaque: opaque}, nil
	case part.Text != "":
		return Item{Kind: KindAssistant, Content: []ContentPart{{Kind: ContentKindText, Text: part.Text}}, Opaque: opaque}, nil
	case len(part.ThoughtSignature) > 0:
		return Item{Kind: KindReasoning, Reasoning: &Reasoning{}, Opaque: opaque}, nil
	default:
		// Other data, such as executable code or inline media, is kept and
		// replayed verbatim. Part types: https://ai.google.dev/api/caching#Part
		return Item{Kind: KindProviderTool, Opaque: opaque}, nil
	}
}

// geminiPartEmpty reports whether a part carries nothing at all, like the
// empty text parts Gemini sometimes sends alongside real content (often the
// last streamed chunk, which only carries the finish reason and usage). A bare
// thought flag carries nothing either.
// https://ai.google.dev/api/caching#Part
func geminiPartEmpty(part *genai.Part) bool {
	if part == nil {
		return false
	}
	rest := *part
	rest.Thought = false
	raw, err := json.Marshal(&rest)
	if err != nil {
		return false
	}
	empty, _ := json.Marshal(&genai.Part{})
	return bytes.Equal(raw, empty)
}

// geminiSignatureOnly reports whether a part carries a thought signature but
// no data. Gemini requires every part to carry data, so such a signature must
// travel on a neighbouring part.
func geminiSignatureOnly(part *genai.Part) bool {
	if part == nil || len(part.ThoughtSignature) == 0 {
		return false
	}
	rest := *part
	rest.ThoughtSignature = nil
	return geminiPartEmpty(&rest)
}

// toGeminiContents groups consecutive parts by role.
func toGeminiContents(log []Item) ([]*genai.Content, error) {
	var contents []*genai.Content
	calls := make(map[string]*genai.FunctionCall)
	for _, e := range log {
		if e.Kind == KindProviderTool && len(e.Opaque[geminiPartOpaqueKey]) == 0 {
			continue
		}
		parts, err := toGeminiParts(e, calls)
		if err != nil {
			return nil, err
		}
		if len(parts) == 0 {
			continue
		}
		if len(parts) == 1 && geminiSignatureOnly(parts[0]) {
			attachOrphanSignature(contents, parts[0].ThoughtSignature)
			continue
		}
		if e.Kind == KindToolCall && e.ToolCall != nil {
			calls[e.ToolCall.ID] = parts[0].FunctionCall
		}
		contents = appendGeminiContent(contents, geminiRole(e.Kind), parts)
	}
	return contents, nil
}

// attachOrphanSignature moves a signature on its own onto the last model
// part. Sessions recorded before signatures were folded into the part they
// sign may hold one on its own.
func attachOrphanSignature(contents []*genai.Content, signature []byte) {
	n := len(contents)
	if n == 0 || contents[n-1].Role != "model" {
		return
	}
	parts := contents[n-1].Parts
	if prev := parts[len(parts)-1]; len(prev.ThoughtSignature) == 0 {
		prev.ThoughtSignature = signature
	}
}

func geminiRole(kind Kind) string {
	if kind == KindUser || kind == KindToolResult {
		return "user"
	}
	return "model"
}

// appendGeminiContent adds parts to the last content when it has the same
// role, or starts a new one.
func appendGeminiContent(contents []*genai.Content, role string, parts []*genai.Part) []*genai.Content {
	if len(contents) == 0 || contents[len(contents)-1].Role != role {
		contents = append(contents, &genai.Content{Role: role})
	}
	last := contents[len(contents)-1]
	last.Parts = append(last.Parts, parts...)
	return contents
}

// geminiStandInSignature is the documented "skip_thought_signature_validator"
// value. The field is bytes sent as base64, so it holds that string decoded;
// Go encodes it with the standard alphabet ('/' for '_'), which the API
// decodes to the same bytes.
var geminiStandInSignature, _ = base64.URLEncoding.DecodeString("skip_thought_signature_validator")

// toGeminiParts renders an entry as content parts.
func toGeminiParts(e Item, calls map[string]*genai.FunctionCall) ([]*genai.Part, error) {
	if raw := e.Opaque[geminiPartOpaqueKey]; len(raw) > 0 && e.Kind != KindUser && e.Kind != KindToolResult {
		var part genai.Part
		if err := json.Unmarshal(raw, &part); err != nil {
			return nil, fmt.Errorf("decode Gemini part: %w", err)
		}
		return []*genai.Part{&part}, nil
	}
	switch e.Kind {
	case KindUser, KindAssistant:
		return geminiContentParts(e.Content)
	case KindReasoning:
		// Reasoning without signed provider data cannot be replayed.
		return nil, nil
	case KindToolCall:
		if e.ToolCall == nil {
			return nil, errors.New("tool call entry carries no tool call")
		}
		return geminiFunctionCallPart(e.ToolCall)
	case KindToolResult:
		if e.ToolResult == nil {
			return nil, errors.New("tool result entry carries no tool result")
		}
		return geminiFunctionResponsePart(e.ToolResult, calls)
	default:
		return nil, fmt.Errorf("unsupported entry kind %d for Gemini part", e.Kind)
	}
}

// geminiContentParts renders a message's parts as text and file parts.
func geminiContentParts(content []ContentPart) ([]*genai.Part, error) {
	var parts []*genai.Part
	for _, part := range content {
		if text, ok := part.InlineText(); ok {
			parts = append(parts, genai.NewPartFromText(text))
			continue
		}
		if part.Kind == ContentKindFile {
			parts = append(parts, geminiFilePart(part))
			continue
		}
		if part.Kind != ContentKindText && part.Kind != ContentKindRefusal {
			return nil, fmt.Errorf("unsupported content part kind %q", part.Kind)
		}
		if strings.TrimSpace(part.Text) == "" {
			continue // Gemini rejects empty text parts.
		}
		parts = append(parts, genai.NewPartFromText(part.Text))
	}
	return parts, nil
}

func geminiFilePart(part ContentPart) *genai.Part {
	if part.URL != "" {
		return genai.NewPartFromURI(part.URL, part.MIME)
	}
	return genai.NewPartFromBytes(part.Data, part.MIME)
}

// geminiFunctionCallPart renders a tool call. History from another provider
// has no thought signature, which Gemini 3 requires on function calls; the
// part carries the documented stand-in:
// https://ai.google.dev/gemini-api/docs/thought-signatures
func geminiFunctionCallPart(call *ToolCall) ([]*genai.Part, error) {
	args := make(map[string]any)
	if len(call.Args) > 0 {
		if err := json.Unmarshal(call.objectArgs(), &args); err != nil {
			return nil, fmt.Errorf("decode Gemini function arguments: %w", err)
		}
	}
	return []*genai.Part{{
		FunctionCall:     &genai.FunctionCall{ID: call.ID, Name: call.Name, Args: args},
		ThoughtSignature: geminiStandInSignature,
	}}, nil
}

// geminiFunctionResponsePart renders a tool result, named after the call it
// answers.
func geminiFunctionResponsePart(r *ToolResult, calls map[string]*genai.FunctionCall) ([]*genai.Part, error) {
	call := calls[r.CallID]
	if call == nil {
		return nil, fmt.Errorf("Gemini function response has no matching call %q", r.CallID)
	}
	text, isError := r.text()
	key := "output"
	if isError {
		key = "error"
	}
	response := map[string]any{key: text}
	return []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: call.ID, Name: call.Name, Response: response}}}, nil
}

// geminiLimit returns err as a *LimitError when Gemini reported a limit. Its
// delay comes from the google.rpc.RetryInfo detail, as there are no headers.
// The genai client is told not to retry limits, so the error is not retried.
func geminiLimit(err error) error {
	var apiErr genai.APIError
	if !errors.As(err, &apiErr) {
		return err
	}
	limit, ok := limitError(err, apiErr.Code, nil, []string{apiErr.Status}, apiErr.Message).(*LimitError)
	if !ok {
		return err
	}
	limit.Retried = false
	for _, detail := range apiErr.Details {
		if kind, _ := detail["@type"].(string); strings.HasSuffix(kind, "google.rpc.RetryInfo") {
			if delay, _ := detail["retryDelay"].(string); delay != "" {
				limit.RetryAfter, _ = time.ParseDuration(delay)
			}
		}
	}
	return limit
}
