---
name: crux
description: How to build LLM agents in Go with crux (crux.foo). Use whenever code imports or adds crux.foo, or when creating agents, tools, sessions, MCP servers, approvals, structured output or typed decisions with it.
---

# crux

crux is a Go toolkit for LLM agents that run the same way on OpenAI,
Anthropic, Google Gemini, xAI, DeepSeek, OpenRouter, Ollama and decision models
such as TypeSafe's Jev. It is young and changes between versions, so trust this
skill and `go doc crux.foo` over anything you remember.

## How crux is structured

Everything comes from one package: `import "crux.foo"`. Never
import `crux.foo/internal/...`. `crux.foo/cruxtest`
is the only other package, for tests.

There are three pieces, kept apart on purpose:

1. **Tools** are plain Go functions registered in a **registry**, the default
   one (`crux.RegisterTool`) or your own (`crux.NewToolsRegistry`). Any package
   can register tools, usually in `init`. The input struct is the JSON schema.
2. **Agents** are immutable blueprints: a name, a model, instructions and the
   **names** of the tools they may use (`crux.WithTools([]string{...})`).
   Agents never hold tool functions, and are safe to share between goroutines.
3. **Sessions** are one conversation with an agent, stored as an append-only
   log of entries. `Run` sends input and loops (model → tool calls → model …)
   until the model answers. A session is not safe for concurrent use.

Toolsets (`crux.Filesystem`, `crux.Network`, MCP servers, skills) are groups of
tools registered the same way and selected by toolset name.

## Minimal program

```go
package main

import (
	"context"
	"fmt"
	"log"

	"crux.foo"
)

type weatherArgs struct {
	City string `json:"city" description:"City name"`
}

func init() {
	crux.RegisterTool("get_weather", "Get the current weather",
		func(ctx context.Context, in weatherArgs) (string, error) {
			return "Sunny, 22°C in " + in.City, nil
		})
}

func main() {
	ctx := context.Background()
	agent := crux.Must(crux.New("assistant", crux.ClaudeHaiku4_5,
		crux.WithInstructions("Answer briefly."),
		crux.WithTools([]string{"get_weather"})))
	session := crux.MustSession(crux.NewSession(ctx, agent))
	answer, err := session.Run(ctx, "What's the weather in Paris?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(answer)
}
```

The provider is inferred from the model constant, and the API key comes from
its usual environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …). No
other setup is needed.

## Rules that matter most

- Register a tool once, then reference it by name. Don't look for a way to pass
  functions to `crux.New`; there isn't one.
- Registering an invalid or duplicate tool name panics. Names match
  `^[a-zA-Z0-9_-]{1,64}$`.
- A tool's error or panic is sent to the model as the result; it never fails
  the run.
- One `Session` per conversation, used from one goroutine at a time. To continue
  a conversation later, persist it with `WithStore` and reopen it with
  `WithSessionID`.
- `RunInto(ctx, &target, inputs...)` takes the target **before** the inputs.
- After a failed `Run`, call `Resume`; calling `Run` again repeats the input.
- Check errors with `errors.Is` against the `crux.Err...` sentinels.
- The `crux` package is pure Go (`CGO_ENABLED=0`); use a pure-Go GORM driver
  such as `github.com/glebarez/sqlite`, not `mattn/go-sqlite3`.
- Constructors return errors; `crux.Must`, `crux.MustSession` and
  `crux.MustDecider` panic instead, for `main` and examples.

## Reference

Read the page for what you are doing:

- [references/tools.md](references/tools.md): registering tools, schemas, tool options (approval, timeouts, ordering, runtime schemas), session state.
- [references/toolsets.md](references/toolsets.md): grouping tools, the built-in filesystem and network toolsets.
- [references/mcp.md](references/mcp.md): using MCP servers' tools, local or remote, with OAuth.
- [references/skills.md](references/skills.md): giving agents Agent Skills.
- [references/sessions.md](references/sessions.md): running, streaming, inputs and files, persistence, resuming, the log, forking, long conversations.
- [references/structured-output.md](references/structured-output.md): typed answers from an agent.
- [references/decisions.md](references/decisions.md): classification, routing and yes/no with `Decide[T]`.
- [references/approvals.md](references/approvals.md): tools that wait for a human.
- [references/multi-agent.md](references/multi-agent.md): subagents and agents that spawn agents.
- [references/providers.md](references/providers.md): models, providers, keys, reasoning, tool choice, web search.
- [references/errors.md](references/errors.md): the sentinel errors.
- [references/testing.md](references/testing.md): testing agents without API keys.
