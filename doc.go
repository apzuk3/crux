// Package crux is a small toolkit for building LLM agents in Go that run
// the same way on OpenAI, Anthropic, Google Gemini, xAI, DeepSeek,
// OpenRouter and Ollama. Import it as "crux.foo".
//
// There are three pieces:
//
//   - Tools are plain Go functions, registered once anywhere in your program
//     with RegisterTool. Their JSON schema comes from the input type.
//   - An Agent is an immutable blueprint: a model, instructions and the names
//     of the tools it may use. Create one with New.
//   - A Session is one conversation with an agent. Run sends input, executes
//     tool calls until the model answers, and returns the final text.
//
// A minimal agent with one tool:
//
//	type WeatherArgs struct {
//		City string `json:"city" description:"City name"`
//	}
//
//	func init() {
//		crux.RegisterTool("get_weather", "Get the current weather",
//			func(ctx context.Context, in WeatherArgs) (string, error) {
//				return "Sunny, 22°C in " + in.City, nil
//			})
//	}
//
//	agent := crux.Must(crux.New("assistant", crux.ClaudeHaiku4_5,
//		crux.WithTools([]string{"get_weather"})))
//	session := crux.MustSession(crux.NewSession(ctx, agent))
//	answer, err := session.Run(ctx, "What's the weather in Paris?")
//
// API keys are read from the provider's usual environment variable (for
// example ANTHROPIC_API_KEY) unless WithAPIKey is given.
//
// Sessions are kept in memory by default. Pass WithStore to persist them, for
// example with NewGORMStore, and WithSessionID to continue a stored
// conversation.
//
// A session's log is the full record of what happened: besides the
// conversation it holds when each run started and how it ended, each provider
// request (with the agent, model and tool definitions it used) and each tool
// that started. WithEntryHandler reports every entry as it is stored, which is
// the way to follow a session live, subagents included.
//
// # Tools
//
// A tool is registered once, by name, in a registry: the default one with
// RegisterTool, or one from NewToolsRegistry with RegisterToolWithRegistry.
// Agents select tools by name with WithTools, or by group with WithToolsets,
// WithMCPs (MCP servers, see ConfigureMCP) and WithSkills (see AddSkills).
// There is no way to define a tool on an agent. An invalid or repeated tool
// name panics. A tool's error, panic or timeout is sent to the model as its
// result and never fails the run. WithApprovalNeeded makes a call wait for a
// human, WithToolTimeout bounds it, WithSequential keeps calls in order.
//
// # Sessions
//
// A Session is not safe for concurrent use; create one per conversation and
// share the Agent instead. RunInto(ctx, &target, inputs...) decodes the answer
// into target, which comes before the inputs; WithOutputSchemaFrom makes the
// model answer in that shape. Stream yields deltas. After an error, call
// Resume: calling Run again would send the input twice.
//
// # Approvals
//
// When a tool needs approval, Run returns ErrApprovalNeeded. Decide each call
// from PendingApprovals with Approve or Reject, then call Resume. Calls made by
// subagents (WithSubAgent, WithAgentSpawning) surface on the parent.
//
// # Decisions
//
// For classification, routing and yes/no questions, Decide answers the
// questions a struct describes, with calibrated probabilities on decision
// models such as Jev. Create the Decider with NewDecider.
//
// # Errors
//
// Check errors with errors.Is against ErrApprovalNeeded, ErrMaxTurns,
// ErrRefused, ErrOutputValidation, ErrContextTooLong, ErrToolNotFound,
// ErrSessionNotFound and ErrSessionConflict.
//
// # Testing
//
// The cruxtest package provides a mock HTTP transport for testing agents
// without calling a real provider: queue responses with Mock.Expect and pass
// Mock.AgentOptions to New or NewDecider.
//
// # More documentation
//
// The skills/crux directory of this module is a guide for coding agents: how
// crux is structured, the rules that matter and a page per topic. Read
// skills/crux/SKILL.md in the module (under $(go env GOMODCACHE)/crux.foo@<version>),
// or install the directory as an Agent Skill in your project.
package crux
