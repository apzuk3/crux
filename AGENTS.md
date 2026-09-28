# AGENTS.md

Context for coding agents working on crux. Read it before changing anything.

## What crux is

A cross-platform Go agent development kit. **The primary goal is developer experience:** keep things simple, make the common path obvious, and never make users wire many pieces together.

**Current focus:** the foundation of *creating* agents and *running* sessions. That means the loop that sends input, executes tools, and returns a result, working the same across providers. RAG, telemetry, workflows and similar features come later, once this core is solid. Don't start them unless asked.

## Principles (non-negotiable)

- **One flat package.** Everything a user needs comes from `import "github.com/apzuk3/crux"`. Do not create subpackages for features (no `crux/store/...`, `crux/tools/...`). `cruxtest` (test helpers), `examples/` and `evals/` are the only other directories.
- **Pure Go, no cgo.** The `crux` package must build and run with `CGO_ENABLED=0`. Never import a cgo SQLite driver (such as `mattn/go-sqlite3`) or any library that needs cgo or loads native libraries. GORM itself is fine because it's pure Go. Users bring their own GORM driver; tests use the pure-Go `github.com/glebarez/sqlite`. CI enforces this.
- **Small public API.** Don't export something unless users need it. Internal mechanisms stay unexported (for example, session parent tracking). Removing an export later is a breaking change.
- **Works with zero configuration.** `crux.New(name, model)` + `crux.NewSession(ctx, agent)` + `session.Run(ctx, input)` must work with only an API key in the environment.
- **Discuss before redesigning public APIs.** The owner wants to talk through design changes (especially tools, sessions and stores) before code is written. Bug fixes and internal changes can go ahead.

## Core concepts and intentional design decisions

- **Tools are decoupled from agents, on purpose.** Tools are registered in a registry: the default one via `RegisterTool`, or a custom `NewToolsRegistry()` passed with `WithToolsRegistry`. Any package in an application can contribute tools independently of where agents are defined, and each agent references the tools it may use **by name** (`WithTools([]string{...})`). This keeps large LLM apps from turning into a tangle. **Do not replace this with tools defined inline on agents.** Improvements that were discussed and not yet decided:
  - duplicate-name registration should panic (today it silently overwrites);
  - `RegisterTool` could return a reference value that agents can pass instead of a string;
  - a simpler custom-registry option, `WithRegistry(reg)`.
- **Agent** (`agent.go`): an immutable, stateless blueprint (model, instructions, tool names, limits), safe for concurrent use. Its ID is derived from `CanonicalData()` unless `WithAgentID` is given.
- **Session** (`agent.go`): one conversation, stored as an append-only log of `Entry` values (`types.go`). Not safe for concurrent use. `Run`/`RunInto`/`Stream` drive the loop:
  1. run any pending tool calls;
  2. call the provider (`step`);
  3. append what the model produced;
  4. stop on approvals, refusals or a final answer; otherwise repeat, up to `maxTurns`.
- **Tool failures never stop a run.** Tool errors, and tool panics (recovered in `invokeTool`), are sent to the model as `ToolResult.Error`.
- **Approvals:** tools registered with `WithApprovalNeeded(true)` make `Run` return `ErrApprovalNeeded`. The caller then uses `Approve`/`Reject` and `Resume`.
- **Stores** (`store.go`, `store_gorm.go`): `Store.Append(ctx, *Session, entries...)` and `Store.Get(ctx, id)`.
  - `MemoryStore` is the default.
  - `GORMStore` (`NewGORMStore(db)`) persists to the `crux_agents`, `crux_sessions` and `crux_session_logs` tables.
  - `NewSession` with `WithSessionID(id)` loads an existing session from the store, or starts fresh if there isn't one. There is no separate resume function.
  - History seeded with `WithSessionLogs` (as forks do) is written to the store.
- **Parent/child sessions are hidden and only for tracing.**
  - `dispatch` puts the running session into the tool's context under an unexported `sessionContextKey`.
  - `NewSession` reads it, records the parent's ID, and reuses the parent's store.
  - `GORMStore` saves the link in `crux_sessions.parent_id`.
  - There is no public API for this, and none should be added without discussion.
- **Subagents** (`WithSubAgent`): exposed to the parent as a tool named `agent_<name>` that takes `{"task": string}`. Each call runs a fresh child session, and its output is the tool result.
- **Fork** (`fork.go`): copies history into a new session, optionally switching model or provider. Provider-specific data (`Opaque`) is dropped when the provider changes.
- **Errors** (`errors.go`): sentinel errors for `errors.Is` checks: `ErrApprovalNeeded`, `ErrMaxTurns`, `ErrRefused`, `ErrOutputValidation`, `ErrSessionNotFound`, `ErrToolNotFound`.
- **`Kind` values are persisted.** Never renumber them. The value after `KindStateDelta` is a reserved blank (`_`), kept for future compaction.

## Providers

There are only three wire implementations. Every provider maps onto one of them:

- `openai.go`: the OpenAI Responses API. It also serves xAI, DeepSeek, OpenRouter and Ollama (see `xai.go`, `deepseek.go`, `openrouter.go`, `ollama.go`).
- `anthropic.go`: Anthropic Messages. `max_tokens` is required and defaults to `defaultAnthropicMaxTokens` (16384).
- `gemini.go`: Google GenAI.

`*_provider.go` files hold the model constants and register known models, so `New` can infer the provider from the model name (`provider.go`). API keys come from environment variables (`discoverAPIKey`) unless `WithAPIKey` is given. A new agent option that affects requests (like `WithMaxTokens`/`WithTemperature`) must be wired into all three wire files, added to `CanonicalData()`, and copied in `Agent.clone` (`fork.go`).

## Conventions

- Functional options: `AgentOption func(*Agent) error` and `SessionOption func(*Session) error`. Validate inside the option and return an error rather than panicking. `Must`/`MustSession` exist for examples and main functions.
- Match the surrounding style: short doc comments on exported identifiers, few inline comments, errors wrapped with `%w` and context.
- Minimum Go version is 1.25 (set by `openai-go`). Don't use newer standard-library APIs; `go vet` checks this.
- Pre-v0.0.1: breaking changes are acceptable when they improve developer experience. Say so in the PR description.

## Testing

```sh
go test ./...                      # unit tests; no network, no API keys
CGO_ENABLED=0 go test ./...        # must also pass
go vet ./... && go vet -tags evals ./evals/...
gofmt -l .                         # must print nothing
go test -tags evals ./evals/...    # live provider evals; needs API keys, don't run by default
```

- Test agent behaviour end to end with `cruxtest` (a mock HTTP transport that speaks each provider's wire format): `mock.Expect().ReturnText/ReturnToolCall/ReturnRefusal/WithUsage`, `mock.AgentOptions()`, `mock.Requests()`. Tests that need `cruxtest` go in `cruxtest/*_test.go` (package `cruxtest_test`), because the root package can't import it.
- Tests that need unexported fields go in the root package (see `approval_test.go`, `store_test.go`). You can drive tools without a provider by seeding tool calls with `WithSessionLogs` and calling `executeUnexecutedToolCalls`.
- Live evals in `evals/` carry `//go:build evals`. Keep `evals/doc.go` untagged so `go test ./...` still finds the package.
- CI (`.github/workflows/ci.yml`) runs gofmt, vet, `go test -race`, and a `CGO_ENABLED=0` test on Go 1.25 and stable. It also fails if the `crux` package depends on a SQLite driver.

## Layout

| Path | Contents |
|---|---|
| `agent.go` | `Agent`, `Session`, `New`, `NewSession`, the run loop, tool dispatch, approvals |
| `agent_option.go` | agent and session options, `WithSubAgent` |
| `tools.go` | tool registry and `RegisterTool*`; tool input schemas come from Go types (`json` + `description` tags) |
| `schema.go` | output schema validation and per-provider schema adaptation |
| `state.go` | state deltas and `StateSnapshot`/`StateFromContext` |
| `stream.go` | `Session.Stream` (text and reasoning chunks) |
| `fork.go` | `Fork`/`ForkFrom`, `cloneEntries` |
| `store.go`, `store_gorm.go` | `Store`, `MemoryStore`, `GORMStore` |
| `types.go`, `errors.go` | log entry types, sentinel errors |
| `openai.go`, `anthropic.go`, `gemini.go` + adapters | provider wire code |
| `cruxtest/` | mock transport for tests |
| `examples/` | runnable examples (need real API keys) |
| `evals/` | live evals (`-tags evals`) |
