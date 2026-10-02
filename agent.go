package crux

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"slices"

	"github.com/apzuk3/crux/internal/schema"
	"github.com/google/uuid"
	"github.com/invopop/jsonschema"
)

// Agent is the immutable blueprint defining an agent's identity,
// capabilities, instructions, and provider connection.
// It is completely stateless and safe for concurrent use.
type Agent struct {
	id           uuid.UUID
	name         string
	model        string
	instructions string
	provider     Provider
	apiKey       string
	baseURL      string
	httpClient   *http.Client

	// Capabilities & Schemas
	tools         []Tool
	searchOptions *SearchOptions // nil disables web search
	outputSchema  *jsonschema.Schema

	// Execution Policies & Limits
	maxTurns    int
	maxRepairs  int
	maxTokens   int             // 0 uses the provider default
	temperature *float64        // nil uses the provider default
	reasoning   ReasoningEffort // "" uses the provider default

	// Context
	compaction    CompactionOptions
	contextWindow int // 0 uses the model's known window
}

// SearchOptions configures provider-executed web search.
type SearchOptions struct {
	UserLocation *UserLocation // nil means no location is supplied
}

type SearchOption func(*SearchOptions)

func New(name, model string, opts ...AgentOption) (*Agent, error) {
	agent := &Agent{
		name:     name,
		model:    model,
		maxTurns: 10,
		tools:    make([]Tool, 0),
	}
	for _, opt := range opts {
		if err := opt(agent); err != nil {
			return nil, err
		}
	}

	if agent.provider == "" {
		agent.provider = inferProvider(agent.model)
	}

	seen := make(map[string]bool, len(agent.tools))
	for _, tool := range agent.tools {
		if seen[tool.name] {
			return nil, fmt.Errorf("agent %q has two tools named %q", agent.name, tool.name)
		}
		seen[tool.name] = true
	}

	if agent.provider == "" {
		return nil, fmt.Errorf("cannot infer provider from model %q, please pass through crux.WithProvider", agent.model)
	}

	spec, ok := providerSpecs[agent.provider]
	if !ok {
		return nil, fmt.Errorf("unsupported provider %q", agent.provider)
	}
	if agent.apiKey == "" {
		agent.apiKey = firstEnv(spec.envVars)
	}
	if agent.baseURL == "" {
		agent.baseURL = spec.baseURL
	}
	if spec.prepare != nil {
		if err := spec.prepare(agent); err != nil {
			return nil, err
		}
	}

	agent.id = uuid.NewSHA1(agentNamespace, agent.canonicalData())

	return agent, nil
}

func Must(agent *Agent, err error) *Agent {
	if err != nil {
		panic(err)
	}

	return agent
}

// agentNamespace is the base UUID namespace for computing deterministic agent IDs.
var agentNamespace = uuid.MustParse("e0f4f9a0-6f91-4c74-9844-3bfa3eb238b1")

// canonicalData describes the agent's configuration. The agent's ID derives
// from it, and GORMStore keeps it in crux_agents. Each tool is identified by a
// hash of its definition, so the ID changes when a tool's description or schema
// does.
func (a *Agent) canonicalData() []byte {
	var toolNames []string
	if len(a.tools) > 0 {
		toolNames = make([]string, len(a.tools))
		for i, t := range a.tools {
			toolNames[i] = t.name
		}
	}

	data := map[string]any{
		"name":           a.name,
		"model":          a.model,
		"provider":       a.provider,
		"instructions":   a.instructions,
		"max_turns":      a.maxTurns,
		"max_repairs":    a.maxRepairs,
		"max_tokens":     a.maxTokens,
		"temperature":    a.temperature,
		"tools":          toolNames,
		"tool_hashes":    hashTools(a.tools),
		"search_options": a.searchOptions,
		"output_schema":  a.outputSchema,
		"reasoning":      a.reasoning,
	}
	// Added only when set, so agents that keep the defaults keep their IDs.
	if a.compaction != (CompactionOptions{}) {
		data["compaction"] = a.compaction
	}
	if a.contextWindow != 0 {
		data["context_window"] = a.contextWindow
	}
	raw, _ := json.Marshal(data)
	return raw
}

func (a *Agent) ID() uuid.UUID {
	return a.id
}

// Provider returns the provider the agent sends requests to.
func (a *Agent) Provider() Provider {
	return a.provider
}

func (a *Agent) Name() string {
	return a.name
}

func (a *Agent) Model() string {
	return a.model
}

func (a *Agent) Instructions() string {
	return a.instructions
}

func (a *Agent) MaxTurns() int {
	return a.maxTurns
}

func (a *Agent) ToolNames() []string {
	names := make([]string, len(a.tools))
	for i, t := range a.tools {
		names[i] = t.name
	}
	return names
}

func (a *Agent) toolRequiresApproval(name string) bool {
	index := slices.IndexFunc(a.tools, func(tool Tool) bool { return tool.name == name })
	if index < 0 {
		return false
	}
	return a.tools[index].approvalNeeded
}

// AgentOption configures an Agent in New.
type AgentOption func(*Agent) error

// WithTools replaces existing registry tools with the selected ones, preserving subagents.
func WithTools(tools []string) AgentOption {
	return func(a *Agent) error {
		selected, err := defaultToolsRegistry.selected(tools)
		if err != nil {
			return err
		}

		a.tools = slices.DeleteFunc(a.tools, func(t Tool) bool {
			return t.kind == toolKindTool
		})
		a.tools = append(a.tools, selected...)

		return nil
	}
}

// WithoutTools removes all registry tools, preserving subagents.
func WithoutTools() AgentOption {
	return func(a *Agent) error {
		a.tools = slices.DeleteFunc(a.tools, func(t Tool) bool {
			return t.kind == toolKindTool
		})
		return nil
	}
}

// WithoutSubagents removes all subagents, preserving registry tools.
func WithoutSubagents() AgentOption {
	return func(a *Agent) error {
		a.tools = slices.DeleteFunc(a.tools, func(t Tool) bool {
			return t.kind == toolKindSubagent
		})
		return nil
	}
}

// WithToolsRegistry replaces existing registry tools using a custom registry, preserving subagents.
func WithToolsRegistry(tools []string, registry ToolsRegistry) AgentOption {
	return func(a *Agent) error {
		selected, err := registry.selected(tools)
		if err != nil {
			return err
		}

		a.tools = slices.DeleteFunc(a.tools, func(t Tool) bool {
			return t.kind == toolKindTool
		})
		a.tools = append(a.tools, selected...)

		return nil
	}
}

// WithToolsets adds every tool labelled with one of the toolset names (see
// WithToolset) from the default registry. WithTools replaces the agent's tool
// list, so put WithToolsets after it when using both.
func WithToolsets(names ...string) AgentOption {
	return WithToolsetsRegistry(defaultToolsRegistry, names...)
}

// WithToolsetsRegistry adds every tool labelled with one of the toolset names
// from a custom registry.
func WithToolsetsRegistry(registry ToolsRegistry, names ...string) AgentOption {
	return func(a *Agent) error {
		selected, err := registry.inToolsets(names)
		if err != nil {
			return err
		}

		for _, tool := range selected {
			if !slices.ContainsFunc(a.tools, func(t Tool) bool { return t.name == tool.name }) {
				a.tools = append(a.tools, tool)
			}
		}

		return nil
	}
}

func WithInstructions(instructions string) AgentOption {
	return func(a *Agent) error { a.instructions = instructions; return nil }
}

// WithUserLocation supplies geographic context using the fields supported by
// the provider. Gemini uses paired coordinates; OpenAI and Anthropic use the
// named location fields. Providers without location support ignore it.
func WithUserLocation(location UserLocation) SearchOption {
	return func(opts *SearchOptions) {
		opts.UserLocation = cloneUserLocation(&location)
	}
}

// WithWebSearch enables provider-executed search on supported OpenAI,
// Anthropic, Gemini, and xAI models. The model decides when to search.
// Each call replaces the search configuration; no options means no location.
func WithWebSearch(opts ...SearchOption) AgentOption {
	return func(a *Agent) error {
		search := &SearchOptions{}
		for _, opt := range opts {
			opt(search)
		}
		a.searchOptions = search

		return nil
	}
}

// WithProvider sets the provider explicitly. It is required when the model
// is not one of the known models for a provider.
func WithProvider(provider Provider) AgentOption {
	return func(a *Agent) error { a.provider = provider; return nil }
}

// WithModel changes the model. When forking across providers, also set
// WithProvider and the destination's connection settings.
func WithModel(model string) AgentOption {
	return func(a *Agent) error { a.model = model; return nil }
}

// WithMaxTurns limits how many model requests one Run may make. The default is 10.
// Requests that repair invalid output (WithMaxRepairs) are not counted.
func WithMaxTurns(turns int) AgentOption {
	return func(a *Agent) error {
		if turns < 1 {
			return fmt.Errorf("max turns must be at least 1, got %d", turns)
		}
		a.maxTurns = turns
		return nil
	}
}

func WithBaseURL(url string) AgentOption {
	return func(a *Agent) error { a.baseURL = url; return nil }
}

// WithHTTPClient configures a custom HTTP client for API requests across all providers.
func WithHTTPClient(client *http.Client) AgentOption {
	return func(a *Agent) error { a.httpClient = client; return nil }
}

func WithAPIKey(apikey string) AgentOption {
	return func(a *Agent) error { a.apiKey = apikey; return nil }
}

// WithOutputSchema sets the response schema. Crux automatically adapts the schema
// for each provider's wire requirements (strict object closure, property nullability,
// and constraint placement).
func WithOutputSchema(schema *jsonschema.Schema) AgentOption {
	return func(a *Agent) error { a.outputSchema = schema; return nil }
}

// WithMaxTokens caps the tokens the model may generate per request. Zero uses
// the provider default; Anthropic requires a cap and defaults to 16384.
func WithMaxTokens(tokens int) AgentOption {
	return func(a *Agent) error {
		if tokens < 0 {
			return fmt.Errorf("max tokens cannot be negative, got %d", tokens)
		}
		a.maxTokens = tokens
		return nil
	}
}

// WithTemperature sets the sampling temperature. Unset uses the provider
// default. Some reasoning models reject a temperature.
func WithTemperature(temperature float64) AgentOption {
	return func(a *Agent) error {
		if temperature < 0 {
			return fmt.Errorf("temperature cannot be negative, got %v", temperature)
		}
		a.temperature = &temperature
		return nil
	}
}

// ReasoningEffort sets how much a model reasons before it answers.
type ReasoningEffort string

const (
	// ReasoningOff reasons as little as the model allows. Models that always
	// reason, such as Claude Opus 5.5, use their lowest effort instead.
	ReasoningOff    ReasoningEffort = "off"
	ReasoningLow    ReasoningEffort = "low"
	ReasoningMedium ReasoningEffort = "medium"
	ReasoningHigh   ReasoningEffort = "high"
	ReasoningMax    ReasoningEffort = "max"
)

// WithReasoning sets how much the model reasons. Unset keeps the provider
// default. Readable reasoning is requested where the provider offers it, so
// Stream yields ChunkReasoning chunks. Each provider maps the level to its
// own setting, and a model that does not support a level rejects the
// request. On Anthropic, reasoning other than ReasoningOff cannot be combined
// with WithTemperature.
func WithReasoning(effort ReasoningEffort) AgentOption {
	return func(a *Agent) error {
		switch effort {
		case ReasoningOff, ReasoningLow, ReasoningMedium, ReasoningHigh, ReasoningMax:
		default:
			return fmt.Errorf("unknown reasoning effort %q", effort)
		}
		a.reasoning = effort
		return nil
	}
}

// WithMaxRepairs sets the number of attempts the agent will make
// to ask the model to repair its response if output validation fails.
// Each Run or Resume has this many repairs, on top of WithMaxTurns; Resume
// also repairs a stored final answer that fails validation.
func WithMaxRepairs(repairs int) AgentOption {
	return func(a *Agent) error {
		if repairs < 0 {
			return fmt.Errorf("max repairs cannot be negative, got %d", repairs)
		}
		a.maxRepairs = repairs
		return nil
	}
}

type subAgentInput struct {
	Task string `json:"task"`
}

var subAgentInputSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"task": map[string]any{
			"type":        "string",
			"description": "Everything the subagent needs to do the work: the task and any input it applies to.",
		},
	},
	"required": []string{"task"},
}

var outputReflector = &jsonschema.Reflector{
	Anonymous:      true,
	ExpandedStruct: true,
}

func makeOptionalNullable(schema *jsonschema.Schema) {
	if schema == nil {
		return
	}
	seen := make(map[*jsonschema.Schema]bool)
	var visit func(*jsonschema.Schema)
	visit = func(s *jsonschema.Schema) {
		if s == nil || seen[s] {
			return
		}
		seen[s] = true
		if s.Properties != nil {
			reqSet := make(map[string]bool, len(s.Required))
			for _, r := range s.Required {
				reqSet[r] = true
			}
			for pair := s.Properties.Oldest(); pair != nil; pair = pair.Next() {
				visit(pair.Value)
				if !reqSet[pair.Key] {
					s.Properties.Set(pair.Key, &jsonschema.Schema{
						AnyOf: []*jsonschema.Schema{pair.Value, {Type: "null"}},
					})
				}
			}
		}
		if s.Definitions != nil {
			for _, def := range s.Definitions {
				visit(def)
			}
		}
		if s.Items != nil {
			visit(s.Items)
		}
		for _, anyOf := range s.AnyOf {
			visit(anyOf)
		}
		for _, oneOf := range s.OneOf {
			visit(oneOf)
		}
		for _, allOf := range s.AllOf {
			visit(allOf)
		}
	}
	visit(schema)
}

// WithOutputSchemaFrom reflects T into a response schema, or disables structured
// output for string and any. Optional properties also permit null.
func WithOutputSchemaFrom[T any]() AgentOption {
	return func(a *Agent) error {
		var zero T
		switch any(&zero).(type) {
		case *string, *any:
			a.outputSchema = nil
		default:
			schema := outputReflector.ReflectFromType(reflect.TypeFor[T]())
			schema.Version = ""
			makeOptionalNullable(schema)
			a.outputSchema = schema
		}

		return nil
	}
}

// WithSubAgent exposes subAgent as a tool named "agent_<name>". The parent
// passes a task as text; each call runs in a fresh session of subAgent, and its
// final output, shaped by subAgent's output schema if it has one, becomes the
// tool result. description tells the parent model what the subagent does.
func WithSubAgent(subAgent *Agent, description string) AgentOption {
	return func(parent *Agent) error {
		if subAgent == nil {
			return errors.New("subagent cannot be nil")
		}
		name := "agent_" + subAgent.name
		if err := schema.ValidateToolName(name); err != nil {
			return fmt.Errorf("subagent %q: %w", subAgent.name, err)
		}

		tool := Tool{
			name:        name,
			description: description,
			schema:      subAgentInputSchema,
			kind:        toolKindSubagent,
			subAgent:    subAgent,
		}

		parent.tools = append(parent.tools, tool)

		return nil
	}
}

// turnInfo describes the agent a provider request is made with.
func (a *Agent) turnInfo() *TurnInfo {
	return &TurnInfo{
		AgentID:  a.id,
		Provider: a.provider,
		Model:    a.model,
	}
}

// hashTools hashes each tool's name, description and input schema.
func hashTools(tools []Tool) map[string]string {
	if len(tools) == 0 {
		return nil
	}
	hashes := make(map[string]string, len(tools))
	for _, tool := range tools {
		raw, _ := json.Marshal(map[string]any{
			"name":        tool.name,
			"description": tool.description,
			"schema":      tool.schema,
		})
		sum := sha256.Sum256(raw)
		hashes[tool.name] = hex.EncodeToString(sum[:])
	}
	return hashes
}

func (a *Agent) clone(opts ...AgentOption) (*Agent, error) {
	allOpts := make([]AgentOption, 0, len(opts)+2)

	allOpts = append(allOpts, func(fork *Agent) error {
		fork.maxTurns = a.maxTurns
		fork.instructions = a.instructions
		fork.provider = "" // set after opts, so an explicit WithProvider is detectable
		fork.baseURL = a.baseURL
		fork.httpClient = a.httpClient
		fork.outputSchema = a.outputSchema
		fork.maxRepairs = a.maxRepairs
		fork.maxTokens = a.maxTokens
		fork.temperature = a.temperature
		fork.reasoning = a.reasoning
		fork.apiKey = a.apiKey
		fork.tools = slices.Clone(a.tools)
		fork.searchOptions = cloneSearchOptions(a.searchOptions)
		fork.compaction = a.compaction
		fork.contextWindow = a.contextWindow
		return nil
	})

	allOpts = append(allOpts, opts...)

	allOpts = append(allOpts, func(fork *Agent) error {
		if fork.provider == "" {
			fork.provider = a.provider
			if fork.model != a.model {
				if inferred := inferProvider(fork.model); inferred != "" {
					fork.provider = inferred
				}
			}
		}

		if fork.model != a.model && fork.contextWindow == a.contextWindow {
			fork.contextWindow = 0 // set for the old model
		}
		if fork.provider != a.provider {
			if fork.apiKey == a.apiKey {
				fork.apiKey = ""
			}
			if fork.baseURL == a.baseURL {
				fork.baseURL = ""
			}
		}
		return nil
	})

	return New(a.name, a.model, allOpts...)
}
