// Package crux is a small toolkit for building LLM agents in Go that run
// the same way on OpenAI, Anthropic, Google Gemini, xAI, DeepSeek,
// OpenRouter and Ollama.
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
// The cruxtest package provides a mock HTTP transport for testing agents
// without calling a real provider.
package crux
