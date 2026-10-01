# AGENTS.md

Context for coding agents working on crux. Read it before changing anything.

## What crux is

A cross-platform Go agent development kit. **The primary goal is developer experience:** keep things simple, make the common path obvious, and never make users wire many pieces together.

**Current focus:** the foundation of *creating* agents and *running* sessions. That means the loop that sends input, executes tools, and returns a result, working the same across providers. RAG, telemetry, workflows and similar features come later, once this core is solid. Don't start them unless asked.

## Principles (non-negotiable)

- **One flat package.** Everything a user needs comes from `import "github.com/apzuk3/crux"`. Do not create subpackages for features (no `crux/store/...`, `crux/tools/...`). `cruxtest` (test helpers), `internal/` (implementation details users never import), `examples/` and `evals/` are the only other directories.
- **Root holds the API; `internal/` holds the machinery.** The root package keeps every exported identifier and the code bound to `Agent`/`Session` internals (run loop, registry, stores, options). Code that doesn't need them lives in `internal/<pkg>`: provider wire code, the schema engine, toolset implementations, MCP OAuth. Internal packages never import `crux` (it would be a cycle) and define their own plain types; a thin root file adapts (`provider.go` for providers, `cli.go` for the TUI, `toolset.go`/`mcp*.go` for toolsets). Public types stay defined in root, not aliased from `internal`, so `go doc` shows them whole. Moving a built-in tool's input struct must not change its schema, because agent IDs hash tool definitions.
- **Pure Go, no cgo.** The `crux` package must build and run with `CGO_ENABLED=0`. Never import a cgo SQLite driver (such as `mattn/go-sqlite3`) or any library that needs cgo or loads native libraries. GORM itself is fine because it's pure Go. Users bring their own GORM driver; tests use the pure-Go `github.com/glebarez/sqlite`. CI enforces this.
- **Small public API.** Don't export something unless users need it. Internal mechanisms stay unexported (for example, session parent tracking). Removing an export later is a breaking change.
- **Works with zero configuration.** `crux.New(name, model)` + `crux.NewSession(ctx, agent)` + `session.Run(ctx, input)` must work with only an API key in the environment.
- **Discuss before redesigning public APIs.** The owner wants to talk through design changes (especially tools, sessions and stores) before code is written. Bug fixes and internal changes can go ahead.

## Core concepts and intentional design decisions

- **Tools are decoupled from agents, on purpose.** Tools are registered in a registry: the default one via `RegisterTool`, or a custom `NewToolsRegistry()` passed with `WithToolsRegistry`. Any package in an application can contribute tools independently of where agents are defined, and each agent references the tools it may use **by name** (`WithTools([]string{...})`). This keeps large LLM apps from turning into a tangle. **Do not replace this with tools defined inline on agents.** Tool names must match `^[a-zA-Z0-9_-]{1,64}$`; an invalid or already registered name panics (`AddToolset*` returns it as an error), and `New` rejects an agent with two tools of the same name. Improvements that were discussed and not yet decided:
  - `RegisterTool` could return a reference value that agents can pass instead of a string;
  - a simpler custom-registry option, `WithRegistry(reg)`.
- **Runtime schemas:** `WithInputSchema(schema)` replaces the schema generated from the input type, for tools whose arguments are only known at run time (such as MCP tools). The input type may then be `json.RawMessage`.
- **MCP** (`mcp.go`, `internal/mcpclient`): `ConfigureMCP(ctx, name, transport)` connects to an MCP server through the official Go SDK (`github.com/modelcontextprotocol/go-sdk`, kept out of the public API), lists its tools once and registers each one as `<name>_<tool>` with `WithInputSchema`, `WithToolset(name)` and approval unless the server marks it read-only. Agents select them with `WithMCPs(name)` (sugar over `WithToolsets`). All names, schemas and conflicts are checked before anything is registered. Results become text: `IsError` is a tool error, structured content is JSON when there is no text. Remote servers authorize with OAuth on a 401 (`mcpclient.NewOAuthHandler`: protected resource metadata, auth server metadata, dynamic client registration or a configured client, PKCE with a loopback redirect, token cached in a `TokenStore` and refreshed). Not supported yet: `list_changed`, resources, prompts, sampling, elicitation, per-user identities.
- **Network toolset** (`toolset.go`, `internal/network`): `Network(opts...)` returns a `*NetworkToolset` (a `Toolset` with `Close`) whose tools are `dns_lookup`, `whois`, `http_get`, `http_request`, `net_connect`, `net_send`, `net_read`, `net_list`, `net_close`, `net_listen` and `http_serve`. Pure Go on every OS: DNS with `golang.org/x/net/dns/dnsmessage` (system servers from `/etc/resolv.conf`, or `GetAdaptersAddresses` on Windows), WebSockets with `github.com/coder/websocket`. Long-lived connections, listeners and servers are handles in the toolset (shared by its sessions), each with a background reader filling a capped `netBuffer` that `net_read` drains. Every dial goes through `network.Tools.dial`, whose `net.Dialer.Control` checks the resolved IP against `netPolicy` (`WithNetworkPrivate`, `WithNetworkHosts`); `http_get` always applies the public-only rule, which is what lets it run without approval. Approval defaults are `netApprovalDefaults`: lookups, reads and handle management are free, anything that sends data or opens a port needs approval. Approval is static per tool; a per-call approval hook has been discussed and deferred.
- **Toolsets** group tools. A `Toolset` registers its tools in `Register(registry)` and labels each one with the `WithToolset(name)` tool option; `AddToolset`/`AddToolsetWithRegistry` call `Register`. Agents select a whole set with `WithToolsets`/`WithToolsetsRegistry`, which add to the tool list (unlike `WithTools`, which replaces it).
- **Agent** (`agent.go`): an immutable, stateless blueprint (model, instructions, tool names, limits), safe for concurrent use. Its ID is derived from `canonicalData()`, which includes a SHA-256 of each tool definition, so a changed tool description or schema changes the ID.
- **Session** (`session.go`): one conversation, stored as an append-only log of `Entry` values (`types.go`). Not safe for concurrent use. `Run`/`RunInto`/`Stream` drive the loop:
  1. run any pending tool calls (the calls from one model turn run concurrently, at most 8 at a time; results keep the model's order);
  2. record `KindTurnStarted` and call the provider (`step`);
  3. append what the model produced;
  4. stop on approvals, refusals or a final answer; otherwise repeat, up to `maxTurns`. Output repair requests (`WithMaxRepairs`) are extra and don't count as turns.
- **Tool failures never stop a run.** Tool errors, and tool panics (recovered in `invokeTool`), are sent to the model as `ToolResult.Error`. Arguments are validated against the tool's input schema first; repeated keys, keys that match a field only case-insensitively and keys of a struct input that match no field are rejected (maps and nested types with their own decoder accept any key), so an approval shown from the raw arguments matches what the tool receives.
- **Approvals:** tools registered with `WithApprovalNeeded(true)` make `Run` return `ErrApprovalNeeded`. The caller then uses `Approve`/`Reject` and `Resume`. Approvals needed inside a subagent surface on the parent: `PendingApprovals` includes them (`ToolCall.Agent` names the agent that made each call), and `Approve`/`Reject` record the decision in the subagent's session.
- **Stores** (`store.go`): `Store.Append(ctx, *Session, entries...)` and `Store.Get(ctx, id)`.
  - `MemoryStore` is the default.
  - `GORMStore` (`NewGORMStore(db)`) persists to the `crux_agents`, `crux_sessions` and `crux_session_logs` tables.
  - `NewSession` with `WithSessionID(id)` loads an existing session from the store, or starts fresh if there isn't one. There is no separate resume function.
  - History seeded with `WithSessionLogs` (as forks do) is written to the store.
  - The store is the source of truth. `Session.logs` is only a cache: `appendLogs` adds entries to it after `Store.Append` succeeds. After a failed write the session is unchanged, so tool calls whose results weren't stored run again on the next `Run`/`Resume`.
- **Parent/child sessions are hidden and only for tracing.**
  - `dispatch` puts the running session into the tool's context under an unexported `sessionContextKey`.
  - `NewSession` reads it, records the parent's ID, and reuses the parent's store.
  - `GORMStore` saves the link in `crux_sessions.parent_id`.
  - There is no public API for this, and none should be added without discussion.
- **Subagents** (`WithSubAgent`): exposed to the parent as a tool named `agent_<name>` that takes `{"task": string}`, validated as strictly as other tool arguments. Each call runs a child session whose ID is derived from the parent session and the call (`childSessionID`), and its output is the tool result. Subagent tools have no `invoke`: `dispatch` runs them through `Session.runSubAgent`. When the child stops for approval, `dispatch` returns it as waiting: the call gets no result, the child is kept in `Session.children`, and the parent returns `ErrApprovalNeeded`. Running the call again loads the child by its ID and resumes it. `NewSession` reloads waiting children when it loads a session from the store.
- **Fork** (`session.go`): copies history into a new session, optionally switching model or provider. Provider-specific data (`Opaque`) is dropped when the provider changes.
- **Errors** (`types.go`): sentinel errors for `errors.Is` checks: `ErrApprovalNeeded`, `ErrMaxTurns`, `ErrRefused`, `ErrOutputValidation`, `ErrSessionNotFound`, `ErrSessionConflict`, `ErrToolNotFound`.
- **The log is the absolute source of truth, including the run's lifecycle.** `run` records `KindRunStarted` before it does any work and `KindRunFinished` (`Entry.Run`: outcome and error text) when it returns, even after a cancel. `KindTurnStarted` precedes each provider request (`Entry.Turn`: agent ID, provider and model; the agent ID pins the tool definitions). `KindToolStarted` is written, in one batch, before the tools of a turn run. A start without a result means the tool may have run; a start without a finish means the process died. The last entry of a model response carries `Entry.Response` (provider response ID, time to first streamed token). These kinds are `HiddenFromModel`, and `FinalOutput` skips them rather than treating them as a turn boundary. Only streamed deltas are not stored: they are previews of an entry that is stored whole. Don't add lifecycle events that bypass the log.
- **Terminal chat** (`cli.go`, `internal/tui`): `CLI` is a `WithEntryHandler` listener plus `Stream` for deltas. It shows lifecycle entries (`KindRunStarted`, `KindTurnStarted`, `KindToolStarted`, …) as a progress pipeline, so new lifecycle facts belong in the log and get shown from there. The handler only forwards events; approvals, runs and model switches happen off the UI goroutine. Switching models forks the session with the unexported `forkWith`, which also copies the session's entry handlers (public `Fork` does not).
- **Listening:** `WithEntryHandler` is a session option that subagent sessions inherit (`NewSession`, through `sessionContextKey`). It is additive (handlers run in the order added; a child keeps the parent's handlers before its own) and fires from `appendLogs` after `Store.Append` succeeds, serialised by a mutex shared across the session tree. User rejections set `ToolResult.Denied`. There are deliberately no tool hooks: when control over tools is needed, prefer tool middleware on the registry or per-call approval decided by code (discuss first).
- **`Kind` values are persisted.** Never renumber them; append new ones at the end. The value after `KindStateDelta` is a reserved blank (`_`), kept for future compaction.

## Providers

There are only three wire implementations, in `internal/provider`. Every provider maps onto one of them:

- `openai.go`: the OpenAI Responses API. It also serves xAI, DeepSeek, OpenRouter and Ollama.
- `anthropic.go`: Anthropic Messages. `max_tokens` is required and defaults to `defaultAnthropicMaxTokens` (16384). Every request sets prompt-cache breakpoints (`setAnthropicCacheBreakpoints`); the API allows at most 4.
- `gemini.go`: Google GenAI.

Each is a `provider.Step` that takes a `provider.Request` (settings, tools, adapted output schema, and the model-visible log as `provider.Item`s) and returns `Item`s. `provider.go` in the root builds the request from the agent and converts the items back to `Entry` values; refusals come back as `provider.RefusedError` and become errors wrapping `ErrRefused`.

`models.go` holds every provider's model constants (OpenAI ones are prefixed `OpenAI`) and registers each provider in its single `init` with `registerProvider` (`provider.go`), next to the providers' `prepare` hooks. Each `providerSpec` lists the known models (so `New` can infer the provider from the model name), the API key environment variables (used unless `WithAPIKey` is given), the default base URL, the wire `step`, the output-schema adapter and an optional `prepare` hook that `New` runs for provider-specific rules and defaults. Adding a provider means adding its constants and registration to `models.go`; don't add `switch` statements on the provider elsewhere. Every model constant must be in its provider's `models` list. A new agent option that affects requests (like `WithMaxTokens`/`WithTemperature`/`WithReasoning`) must be added to `provider.Request` and `wireRequest` (`provider.go`), used in all three wire files, added to `canonicalData()`, and copied in `Agent.clone` (`agent.go`).

## Conventions

- Functional options: `AgentOption func(*Agent) error` and `SessionOption func(*Session) error`. Validate inside the option and return an error rather than panicking. `Must`/`MustSession` exist for examples and main functions.
- Match the surrounding style: short doc comments on exported identifiers, few inline comments, errors wrapped with `%w` and context.
- Minimum Go version is 1.25.8 (set by `glamour`; `openai-go` needs 1.25). Check a new dependency's `go` directive before adding it: `bubbletea` v2.0.10 needs Go 1.26, so it is pinned at v2.0.9, and `golang.org/x/oauth2` v0.36+ needs Go 1.26, so it is pinned at v0.35.0. Don't use newer standard-library APIs; `go vet` checks this.
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
- CI (`.github/workflows/ci.yml`) runs gofmt, vet, `go test -race`, and a `CGO_ENABLED=0` test on Go 1.25 and stable, on Linux, macOS and Windows (no `-race` on Windows). It also fails if the `crux` package depends on a SQLite driver.

## Layout

| Path | Contents |
|---|---|
| `agent.go` | `Agent`, `New`, every `AgentOption` (including `WithSubAgent`, `WithOutputSchemaFrom`), `canonicalData`, `clone`, turn info and tool hashes |
| `session.go` | `Session`, `NewSession`, session options (`WithStore`, `WithEntryHandler`, …), the run loop, tool dispatch, approvals, `Stream`, state, `Fork` |
| `tools.go` | tool registry, `RegisterTool*`, `Toolset`/`AddToolset*` |
| `toolset.go` | the built-in toolsets: `Filesystem(root)` (`"filesystem"`; tools that change files need approval) and `Network()` (options, approval defaults); public tool names and registration |
| `mcp.go` | `ConfigureMCP`, `WithMCPs`, MCP transports and options, `OAuthConfig`, `TokenStore`; tool registration |
| `provider.go` | `Provider`, `providerSpec` and registration; `wireRequest` and the `Entry`/`provider.Item` conversion; output schema adaptation and validation |
| `models.go` | model constants and each provider's registration in `init` |
| `store.go` | `Store`, `MemoryStore`, `GORMStore` |
| `types.go` | log entry types and sentinel errors |
| `util.go` | generic unexported helpers: URL secret redaction, env lookup, deep copies (`cloneEntries`, `cloneState`, …), `decodeInto` |
| `cli.go` | `CLI(agent, opts...)`: adapts a session to `internal/tui` (entries become `tui.Event`s, subagent sessions are tied to the call that started them) |
| `internal/schema/` | tool input schemas from Go types (`json` + `description` tags), strict argument checks; output schema adapters and validation |
| `internal/provider/` | provider wire code: `Request`/`Item`, `OpenAI`, `Anthropic`, `Gemini` steps, default HTTP client |
| `internal/network/` | network tools: address policy, handles and buffers, sockets, HTTP client and server, DNS, whois |
| `internal/filesystem/` | file tools confined to a root with `os.Root` |
| `internal/mcpclient/` | MCP OAuth (handler, token store), result rendering, transport helpers |
| `internal/tui/` | the terminal chat (Bubble Tea, Lip Gloss, Glamour); it doesn't import `crux` and speaks only `tui.Info`, `tui.Event` and `tui.Backend` |
| `cruxtest/` | mock transport for tests |
| `examples/` | runnable examples (need real API keys) |
| `evals/` | live evals (`-tags evals`) |
