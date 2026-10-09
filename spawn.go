package crux

import (
	"cmp"
	"errors"
	"fmt"
	"slices"
	"strings"

	"crux.foo/internal/schema"
)

// spawnToolName is the tool WithAgentSpawning adds.
const spawnToolName = "spawn_agent"

const spawnDescription = "Create an agent for a task and run it; its final answer is the result. " +
	"The agent does not see this conversation: write its instructions and put everything it needs in the task. " +
	"Give it only the tools the task needs."

// SpawnOption configures WithAgentSpawning.
type SpawnOption func(*spawnConfig) error

type spawnConfig struct {
	models   []string // nil allows only the parent's model
	tools    []Tool   // nil allows the parent's tools
	maxTurns int      // 0 uses the parent's
	parallel int      // 0 allows as many as other tool calls
}

// WithSpawnModels sets the models a spawned agent may run on. The first is
// used when the model does not choose. The default is the parent's model.
func WithSpawnModels(models ...string) SpawnOption {
	return func(c *spawnConfig) error {
		if len(models) == 0 {
			return errors.New("spawn models cannot be empty")
		}
		c.models = slices.Compact(slices.Clone(models))
		return nil
	}
}

// WithSpawnTools sets the tools from the default registry that a spawned
// agent may be given. The default is the parent's tools.
func WithSpawnTools(names ...string) SpawnOption {
	return WithSpawnToolsRegistry(defaultToolsRegistry, names...)
}

// WithSpawnToolsRegistry sets the tools from a custom registry that a spawned
// agent may be given.
func WithSpawnToolsRegistry(registry ToolsRegistry, names ...string) SpawnOption {
	return func(c *spawnConfig) error {
		tools, err := registry.selected(names)
		if err != nil {
			return fmt.Errorf("spawn tools: %w", err)
		}
		c.tools = tools
		return nil
	}
}

// WithSpawnMaxTurns limits how many model requests a spawned agent may make
// in one call. The default is the parent's limit.
func WithSpawnMaxTurns(turns int) SpawnOption {
	return func(c *spawnConfig) error {
		if turns < 1 {
			return fmt.Errorf("spawn max turns must be at least 1, got %d", turns)
		}
		c.maxTurns = turns
		return nil
	}
}

// WithSpawnConcurrency limits how many spawned agents run at the same time.
// When the model spawns more in one turn, the rest wait for a running one to
// finish. By default as many run at once as other tool calls (8).
func WithSpawnConcurrency(agents int) SpawnOption {
	return func(c *spawnConfig) error {
		if agents < 1 {
			return fmt.Errorf("spawn concurrency must be at least 1, got %d", agents)
		}
		c.parallel = agents
		return nil
	}
}

// spawnConcurrency reports whether name is the spawn tool and, if so, how
// many of its calls may run at once.
func (a *Agent) spawnConcurrency(name string) (int, bool) {
	index := slices.IndexFunc(a.tools, func(t Tool) bool { return t.name == name && t.spawn != nil })
	if index < 0 {
		return 0, false
	}
	return cmp.Or(a.tools[index].spawn.parallel, maxConcurrentToolCalls), true
}

// WithAgentSpawning adds a tool named "spawn_agent" with which the model
// creates an agent and runs it on a task, choosing its name, instructions,
// tools and model within the limits the options set. By default a spawned
// agent may use the parent's tools and model and has the parent's turn limit
// and connection settings. A spawned agent cannot spawn agents, has no output
// schema, and runs like a subagent: in a session derived from the call, with
// approvals its tools need surfacing on the parent.
func WithAgentSpawning(opts ...SpawnOption) AgentOption {
	return func(a *Agent) error {
		config := &spawnConfig{}
		for _, opt := range opts {
			if err := opt(config); err != nil {
				return err
			}
		}
		a.tools = append(a.tools, Tool{
			name:  spawnToolName,
			kind:  toolKindSubagent,
			spawn: config,
		})
		return nil
	}
}

// spawnLimits returns the models and tools a spawned agent may have.
func (a *Agent) spawnLimits(config *spawnConfig) (models []string, tools []Tool, err error) {
	models = config.models
	if models == nil {
		models = []string{a.model}
	}
	for _, model := range models {
		if model != a.model && inferProvider(model) == "" {
			return nil, nil, fmt.Errorf("spawn model %q: cannot infer its provider", model)
		}
	}

	tools = config.tools
	if tools == nil {
		tools = slices.DeleteFunc(slices.Clone(a.tools), func(t Tool) bool { return t.spawn != nil })
	}
	return models, tools, nil
}

// prepareSpawning gives the spawn tool its description and input schema,
// which list the tools and models the agent allows, so the model knows what
// it can give the agents it creates.
func (a *Agent) prepareSpawning() error {
	index := slices.IndexFunc(a.tools, func(t Tool) bool { return t.spawn != nil })
	if index < 0 {
		return nil
	}
	models, tools, err := a.spawnLimits(a.tools[index].spawn)
	if err != nil {
		return fmt.Errorf("agent %q: %w", a.name, err)
	}
	a.tools[index].description = spawnToolDescription(tools)
	a.tools[index].schema = spawnSchema(models, tools)
	return nil
}

func spawnToolDescription(tools []Tool) string {
	if len(tools) == 0 {
		return spawnDescription + " It can have no tools."
	}
	var b strings.Builder
	b.WriteString(spawnDescription)
	b.WriteString(" Tools you can give it:")
	for _, t := range tools {
		fmt.Fprintf(&b, "\n- %s", t.name)
		if t.description != "" {
			fmt.Fprintf(&b, ": %s", t.description)
		}
	}
	return b.String()
}

func spawnSchema(models []string, tools []Tool) map[string]any {
	properties := map[string]any{
		"name": map[string]any{
			"type":        "string",
			"description": "A short name for the agent, such as researcher.",
		},
		"instructions": map[string]any{
			"type":        "string",
			"description": "The agent's instructions: its role, how to work and what to answer with.",
		},
		"task": map[string]any{
			"type":        "string",
			"description": "Everything the agent needs to do the work: the task and any input it applies to.",
		},
	}
	if len(tools) > 0 {
		names := make([]string, len(tools))
		for i, t := range tools {
			names[i] = t.name
		}
		properties["tools"] = map[string]any{
			"type":        "array",
			"items":       map[string]any{"type": "string", "enum": names},
			"description": "The tools the agent may use. Omit for none.",
		}
	}
	if len(models) > 1 {
		properties["model"] = map[string]any{
			"type":        "string",
			"enum":        slices.Clone(models),
			"description": fmt.Sprintf("The model the agent runs on. Defaults to %s.", models[0]),
		}
	}
	return map[string]any{
		"type":       "object",
		"properties": properties,
		"required":   []string{"name", "instructions", "task"},
	}
}

type spawnInput struct {
	Name         string   `json:"name"`
	Instructions string   `json:"instructions"`
	Task         string   `json:"task"`
	Tools        []string `json:"tools"`
	Model        string   `json:"model"`
}

// spawnedAgent builds the agent a spawn call asks for. It depends only on
// the parent and the call's arguments, so a call that runs again gets the
// same agent and continues its session.
func (a *Agent) spawnedAgent(tool Tool, args []byte) (*Agent, string, error) {
	validator, err := schema.Compile(tool.schema)
	if err != nil {
		return nil, "", err
	}
	in, err := schema.DecodeArgs[spawnInput](tool.name, args, validator)
	if err != nil {
		return nil, "", err
	}
	name := strings.TrimSpace(in.Name)
	switch {
	case name == "" || len(name) > 64:
		return nil, "", errors.New("the agent needs a name of 1 to 64 characters")
	case strings.TrimSpace(in.Instructions) == "":
		return nil, "", errors.New("the agent needs instructions")
	case strings.TrimSpace(in.Task) == "":
		return nil, "", errors.New("the agent needs a non-empty task")
	}

	models, allowed, err := a.spawnLimits(tool.spawn)
	if err != nil {
		return nil, "", err
	}
	model := models[0]
	if in.Model != "" {
		if !slices.Contains(models, in.Model) {
			return nil, "", fmt.Errorf("model %q is not allowed", in.Model)
		}
		model = in.Model
	}
	var tools []Tool
	for _, t := range allowed {
		if slices.Contains(in.Tools, t.name) {
			tools = append(tools, t)
		}
	}
	for _, requested := range in.Tools {
		if !slices.ContainsFunc(tools, func(t Tool) bool { return t.name == requested }) {
			return nil, "", fmt.Errorf("tool %q is not allowed", requested)
		}
	}

	maxTurns := cmp.Or(tool.spawn.maxTurns, a.maxTurns)
	agent, err := a.clone(func(c *Agent) error {
		c.name = name
		c.model = model
		c.instructions = in.Instructions
		c.tools = tools
		c.maxTurns = maxTurns
		c.outputSchema = nil
		c.toolChoice = ""
		return nil
	})
	if err != nil {
		return nil, "", fmt.Errorf("spawn agent %q: %w", name, err)
	}
	return agent, in.Task, nil
}
