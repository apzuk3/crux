# Tools

A tool is a Go function the model can call. Tools live in a registry and agents
select them by name, so tools can be defined in any package, independently of
where agents are built.

## Registering

```go
type searchArgs struct {
	Query string `json:"query" description:"What to search for"`
	Limit int    `json:"limit,omitempty" description:"Maximum results, default 10"`
}

func init() {
	crux.RegisterTool("search_docs", "Search the product documentation",
		func(ctx context.Context, in searchArgs) ([]string, error) {
			return []string{"Getting started", "Billing"}, nil
		})
}

agent := crux.Must(crux.New("support", crux.OpenAIGPT5_4Mini,
	crux.WithTools([]string{"search_docs"})))
```

- `RegisterTool[In, Out](name, description, fn, opts...)`, where
  `fn func(ctx context.Context, in In) (Out, error)`.
- **In** must be a struct (or a map). Its JSON schema is generated from it:
  field names come from `json` tags, `description` tags are shown to the model,
  and a field is optional when it is a pointer or tagged `omitempty`.
- **Out** can be anything. A `string` or `fmt.Stringer` is sent as text,
  anything else as JSON.
- Arguments are validated strictly before `fn` runs: unknown keys, repeated
  keys and keys that only match a field case-insensitively are rejected, and
  the model is told why.
- Names must match `^[a-zA-Z0-9_-]{1,64}$`. An invalid or already registered
  name **panics**, like `http.HandleFunc`. Register each tool once, in `init` or
  at startup, never per request.

## Selecting tools on an agent

- `crux.WithTools([]string{...})` replaces the agent's tool list.
- `crux.WithToolsets("name", ...)` adds every tool of those toolsets; put it
  after `WithTools` when using both.
- `crux.WithoutTools()` removes them all.
- `crux.New` fails with `ErrToolNotFound` for a name nobody registered.

## Your own registry

The default registry is global. For isolation (tests, plugins, multi-tenant
apps), create one and pass it explicitly:

```go
reg := crux.NewToolsRegistry()
crux.RegisterToolWithRegistry(reg, "lookup", "Look a value up",
	func(ctx context.Context, in searchArgs) (string, *crux.StateDelta, error) {
		return "found", nil, nil
	})
agent := crux.Must(crux.New("bot", crux.ClaudeHaiku4_5,
	crux.WithToolsRegistry([]string{"lookup"}, reg)))
```

Note that `RegisterToolWithRegistry`'s function also returns a `*StateDelta`
(nil when it changes nothing).

## Errors never stop a run

A returned error, a panic or a timeout becomes the tool's result with an error
message, and the model decides what to do next. Return errors the model can act
on ("no customer with id 42"), not stack traces.

## Tool options

Pass them after the function:

| Option | Effect |
|---|---|
| `crux.WithApprovalNeeded(true)` | The call waits for a human; see [approvals.md](approvals.md). |
| `crux.WithToolTimeout(d)` | Cancels the tool's context after `d`; the model gets an error. |
| `crux.WithSequential()` | Calls from one model turn run one at a time, in order (calls normally run concurrently, up to 8). |
| `crux.WithInputSchema(schema)` | A JSON Schema known only at run time replaces the generated one; take the input as `json.RawMessage` or `map[string]any`. |
| `crux.WithToolset("name")` | Labels the tool so `WithToolsets("name")` selects it. |

## Session state

Tools can share state through the session. `RegisterToolStateMutate` returns a
`*crux.StateDelta{Set: ..., Delete: ...}` that is recorded in the log;
`crux.StateFromContext(ctx)` gives a tool a snapshot of the current state, and
`session.StateSnapshot()` gives it to your code.

```go
crux.RegisterToolStateMutate("set_plan", "Choose the customer's plan",
	func(ctx context.Context, in struct {
		Plan string `json:"plan"`
	}) (string, *crux.StateDelta, error) {
		return "ok", &crux.StateDelta{Set: map[string]any{"plan": in.Plan}}, nil
	})
```

## Pitfalls

- Defining tools inside a request handler: registration is global and panics on
  the second call.
- Expecting `crux.New` to take functions: it only takes tool names.
- Returning huge outputs: everything a tool returns goes into the model's
  context. Summarise or paginate.
