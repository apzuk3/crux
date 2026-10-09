# Sessions

A `*crux.Session` is one conversation with an agent. Create one per
conversation and use it from one goroutine at a time; the `*crux.Agent` can be
shared freely.

```go
session, err := crux.NewSession(ctx, agent)
answer, err := session.Run(ctx, "Summarise this ticket", ticket)
```

## Running

- `Run(ctx, inputs...) (string, error)`: sends the inputs as the next user
  message and loops until the model answers: model → tool calls (concurrently)
  → model …, at most `WithMaxTurns` turns (default 10; then `ErrMaxTurns`).
- `RunInto(ctx, &target, inputs...)`: like Run, then decodes the answer into
  `target`. The target comes **first**. See
  [structured-output.md](structured-output.md).
- `Stream(ctx, inputs...)`: an `iter.Seq2[crux.Chunk, error]` of text and
  reasoning deltas. `FinalOutput()` returns the authoritative answer afterwards.

  ```go
  for chunk, err := range session.Stream(ctx, "Explain goroutines") {
  	if err != nil { return err }
  	fmt.Print(chunk.Delta)
  }
  ```
- `Resume(ctx)`: continues after an error or an approval. Use it to retry a
  failed `Run`; calling `Run` again with the same input adds it twice.
- `Run` with no inputs continues the conversation as it is.

## Inputs

Inputs are strings, any value (sent as JSON), and attachments:
`crux.File(path)`, `crux.FileFS(fsys, name)`, `crux.Data(bytes)`,
`crux.Reader(r)`, `crux.URL(url)`, with `.WithName(...)` / `.WithMIME(...)` when
the type can't be detected. Images and PDFs go to the provider natively; text
files are inlined; a file the provider can't take fails before the request.

```go
answer, err := session.Run(ctx, "What's in this receipt?", crux.File("receipt.pdf"))
```

## Persistence and resuming

Sessions live in memory by default. Persist them with a store and continue
them by ID:

```go
import (
	"github.com/glebarez/sqlite" // pure Go; crux builds with CGO_ENABLED=0
	"gorm.io/gorm"
)

db, err := gorm.Open(sqlite.Open("crux.db"), &gorm.Config{})
store, err := crux.NewGORMStore(db) // creates the crux_* tables

session, err := crux.NewSession(ctx, agent,
	crux.WithStore(store),
	crux.WithSessionID(conversationID)) // loads it if it exists, else starts fresh
```

- `session.ID()` is the ID to store on your side.
- `crux.Store` is a two-method interface (`Append`, `Get`) if you need your own.
- `ErrSessionConflict` means another writer appended first: reopen the session
  with `WithSessionID` and retry.
- Use any GORM driver, but a pure-Go one keeps your build cgo-free.

## The log

Everything that happens is an `crux.Entry` in `session.Logs()`: user input,
model output, tool calls and results, approvals, state changes, and lifecycle
entries (`KindRunStarted`, `KindTurnStarted`, `KindToolStarted`,
`KindRunFinished`, …). Follow a session live, subagents included:

```go
session, err := crux.NewSession(ctx, agent,
	crux.WithEntryHandler(func(ctx context.Context, s *crux.Session, e crux.Entry) {
		log.Printf("%s %v", s.Agent().Name(), e.Kind)
	}))
```

Handlers run in order, one at a time, and must not call back into the session.
`session.Usage()` totals tokens.

## Forking

`session.Fork(ctx, opts...)` copies the history into a new session, optionally
on another model or provider: `session.Fork(ctx, crux.WithModel(crux.OpenAIGPT5_4))`.

## Long conversations

Compaction is on by default: near the context window, old tool outputs are
omitted and then older turns are summarised, without changing the log. Tune it
with `crux.WithCompaction(crux.CompactAt(0.7), crux.CompactWith(cheaperModel))`,
set the window with `crux.WithContextWindow(tokens)`, turn it off with
`crux.WithoutCompaction()`, or compact now with `session.Compact(ctx)`.

## Quick terminal chat

`crux.CLI(agent, sessionOpts...)` opens an interactive chat with streaming,
tool progress and approval prompts. Useful for trying an agent out.
