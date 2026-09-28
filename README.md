# crux

[![Go Reference](https://pkg.go.dev/badge/github.com/apzuk3/crux.svg)](https://pkg.go.dev/github.com/apzuk3/crux)

A small Go toolkit for building LLM agents that run the same way on every major provider.

- **Tools are plain Go functions.** The JSON schema comes from the input type.
- **One agent API for every provider:** OpenAI, Anthropic, Google Gemini, xAI, DeepSeek, OpenRouter and Ollama.
- **Sessions are replayable logs.** You can resume them, fork them (even onto another provider) and persist them.
- **One import.** Everything lives in the `crux` package. Pure Go, no cgo, and no database required.

> Status: `v0.0.x`. The API may still change between releases.

## Install

```sh
go get github.com/apzuk3/crux
```

Requires Go 1.25+.

## Quickstart

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/apzuk3/crux"
)

type WeatherArgs struct {
	City string `json:"city" description:"City name, e.g. Paris"`
}

func getWeather(ctx context.Context, in WeatherArgs) (string, error) {
	return "Sunny, 22°C in " + in.City, nil
}

func init() {
	// Register once, anywhere in your program. Any agent can then use it by name.
	crux.RegisterTool("get_weather", "Get the current weather for a city", getWeather)
}

func main() {
	agent := crux.Must(crux.New("assistant", crux.ClaudeHaiku4_5,
		crux.WithInstructions("You are a helpful assistant."),
		crux.WithTools([]string{"get_weather"}),
	))

	ctx := context.Background()
	session := crux.MustSession(crux.NewSession(ctx, agent))

	answer, err := session.Run(ctx, "What's the weather in Paris?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(answer)
}
```

`Run` sends the input, executes every tool the model calls, sends the results back, and repeats until the model gives a final answer or `WithMaxTurns` (default 10) is reached.

## Concepts

| | |
|---|---|
| **Tool** | A Go function `func(ctx, In) (Out, error)` registered with `RegisterTool`. Tools live in a registry (the default one or your own `NewToolsRegistry`), not inside an agent, so any package can contribute tools. |
| **Agent** | An immutable blueprint: model, instructions, and the names of the tools it may use. It is safe to share across goroutines. |
| **Session** | One conversation with an agent. It holds an append-only log of entries (user input, assistant text, tool calls and results). It must not be used from multiple goroutines at once. |
| **Store** | Where a session's log is persisted. The default is in memory. |

## Features

### Structured output

```go
type Forecast struct {
	City    string `json:"city"`
	Summary string `json:"summary"`
}

agent := crux.Must(crux.New("forecaster", crux.ChatModelGPT5_4,
	crux.WithOutputSchemaFrom[Forecast](),
	crux.WithMaxRepairs(1), // ask the model to fix invalid output once
))

var forecast Forecast
err := session.RunInto(ctx, "Describe a spring day in Paris.", &forecast)
```

### Streaming

```go
for chunk, err := range session.Stream(ctx, "Write a haiku about Go.") {
	if err != nil {
		log.Fatal(err)
	}
	fmt.Print(chunk.Delta)
}
```

### Toolsets

A toolset registers a group of related tools in one call. Implement `Register`, label each tool with `WithToolset`, and add it with `AddToolset` (default registry) or `AddToolsetWithRegistry`. An agent then gets every tool in the set with `WithToolsets`:

```go
type mathTools struct{}

func (mathTools) Register(reg crux.ToolsRegistry) error {
	crux.RegisterToolWithRegistry(reg, "add", "Add two numbers", add, crux.WithToolset("math"))
	crux.RegisterToolWithRegistry(reg, "multiply", "Multiply two numbers", multiply, crux.WithToolset("math"))
	return nil
}

if err := crux.AddToolset(mathTools{}); err != nil {
	log.Fatal(err)
}
agent := crux.Must(crux.New("calc", crux.ChatModelGPT5_4,
	crux.WithTools([]string{"get_weather"}), // single tools
	crux.WithToolsets("math"),               // plus every tool in the "math" toolset
))
```

`WithTools` replaces the agent's tool list, so put `WithToolsets` after it. Use `WithToolsetsRegistry` with a custom registry.

### Filesystem tools

`Filesystem(root)` is a built-in toolset for reading and editing files under `root`. Nothing outside `root` can be reached (not through `..`, absolute paths or symlinks), paths use `/` on every OS, and it works the same on Linux, macOS and Windows.

```go
if err := crux.AddToolset(crux.Filesystem("./workspace")); err != nil {
	log.Fatal(err)
}
agent := crux.Must(crux.New("coder", crux.ChatModelGPT5_4, crux.WithToolsets("filesystem")))
```

| Tool | What it does | Approval |
|---|---|---|
| `read_file` | read a text file, optionally a line range | no |
| `read_multiple_files` | read several files at once | no |
| `list_directory` | list a directory | no |
| `directory_tree` | recursive tree, optional `max_depth` | no |
| `glob` | find files by pattern, e.g. `**/*.go` | no |
| `search_files_content` | text or regex search across files | no |
| `write_file` | create or overwrite a file | yes |
| `edit_file` | replace exact, unique text in a file | yes |
| `create_directory` | create directories | yes |
| `remove_directory` | remove empty directories | yes |

### Tool approvals

```go
crux.RegisterTool("delete_file", "Delete a file", deleteFile, crux.WithApprovalNeeded(true))

_, err := session.Run(ctx, "Delete /tmp/cache.txt")
if errors.Is(err, crux.ErrApprovalNeeded) {
	for _, call := range session.PendingApprovals() {
		session.Approve(ctx, call.ID) // or session.Reject(ctx, call.ID, "reason")
	}
	answer, err = session.Resume(ctx)
}
```

### Subagents

```go
researcher := crux.Must(crux.New("researcher", crux.ChatModelGPT5_4, crux.WithTools([]string{"search"})))
coordinator := crux.Must(crux.New("coordinator", crux.ClaudeSonnet5,
	crux.WithSubAgent(researcher, "Researches a topic and returns a brief"),
))
```

The coordinator sees a tool named `agent_researcher` that takes a `task` string.

### Persisting and resuming sessions

Sessions are kept in memory by default. To keep them, use any `crux.Store`. `crux.NewGORMStore` works with any GORM driver:

```go
import (
	"github.com/glebarez/sqlite" // pure Go; or gorm.io/driver/sqlite, postgres, ...
	"gorm.io/gorm"
)

db, _ := gorm.Open(sqlite.Open("crux.db"), &gorm.Config{})
store, _ := crux.NewGORMStore(db)

session, _ := crux.NewSession(ctx, agent, crux.WithStore(store))

// Later, possibly in another process: the same ID continues the conversation.
session, _ = crux.NewSession(ctx, agent, crux.WithStore(store), crux.WithSessionID(id))
```

### Forking

```go
// Continue the same conversation on another provider.
forked, err := session.Fork(ctx, crux.WithModel(crux.Gemini3_8Flash), crux.WithProvider(crux.ProviderGoogle))
```

### Errors

`Run` returns sentinel errors you can check with `errors.Is`: `ErrApprovalNeeded`, `ErrMaxTurns`, `ErrRefused`, `ErrOutputValidation`. A tool that returns an error or panics does not stop the run. The error is sent to the model as the tool result so it can recover.

## Providers

The provider is inferred from known model constants (`crux.ClaudeSonnet5`, `crux.ChatModelGPT5_4`, `crux.Gemini3_8Flash`, …). For any other model ID, pass `crux.WithProvider`.

| Provider | `Provider` | API key environment variable | Web search |
|---|---|---|---|
| OpenAI | `ProviderOpenAI` | `OPENAI_API_KEY` | ✓ |
| Anthropic | `ProviderAnthropic` | `ANTHROPIC_API_KEY` | ✓ |
| Google Gemini | `ProviderGoogle` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | ✓ |
| xAI | `ProviderXAI` | `XAI_API_KEY` | ✓ |
| DeepSeek | `ProviderDeepSeek` | `DEEPSEEK_API_KEY` | – |
| OpenRouter | `ProviderOpenrouter` | `OPENROUTER_API_KEY` | – |
| Ollama | `ProviderOllama` | none needed locally (`http://localhost:11434/v1`) | with `OLLAMA_API_KEY` |

Common agent options: `WithInstructions`, `WithTools`, `WithToolsets`, `WithMaxTurns`, `WithMaxTokens`, `WithTemperature`, `WithOutputSchemaFrom`, `WithWebSearch`, `WithAPIKey`, `WithBaseURL`, `WithHTTPClient`.

## Testing your agents

`cruxtest` is a mock HTTP transport that speaks each provider's wire format, so you can test agent logic without network calls:

```go
mock := cruxtest.NewMock()
mock.Expect().ReturnToolCall("get_weather", map[string]any{"city": "Paris"})
mock.Expect().ReturnText("It's sunny in Paris.")

agent := crux.Must(crux.New("assistant", crux.ClaudeHaiku4_5,
	append(mock.AgentOptions(), crux.WithTools([]string{"get_weather"}))...))
```

## Examples

See [`examples/`](examples): `basic` (multi-tool planner with structured output), `stream`, `store`, `fork`, `subagents`, `websearch`, `filesystem`.

## Development

```sh
go test ./...                      # unit tests, no network
go test -tags evals ./evals/...    # live evaluations; needs provider API keys
```

## License

[Apache 2.0](LICENSE)
