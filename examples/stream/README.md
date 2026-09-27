# Text and reasoning streaming

Set the normal API-key environment variable for your provider, then run:

```sh
go run ./examples/stream -provider openai -model gpt-5.4 \
  -prompt "Explain why binary search is logarithmic"
```

For local Ollama:

```sh
go run ./examples/stream -provider ollama -model YOUR_INSTALLED_MODEL
```

The example labels each text/reasoning section and prints deltas immediately.
Use Ctrl+C to cancel. Reasoning appears only when the provider exposes readable
reasoning or summaries; streaming does not enable thinking or request summaries.

```go
for chunk, err := range session.Stream(ctx, "Explain this code") {
    if err != nil {
        return err
    }
    switch chunk.Kind {
    case crux.ChunkText:
        fmt.Print(chunk.Delta)
    case crux.ChunkReasoning:
        fmt.Fprint(os.Stderr, chunk.Delta)
    }
}
answer, complete := session.FinalOutput()
```

Iterators execute lazily and are single-use. Breaking the loop closes the request.
Chunks are provisional: the stream can fail after displaying partial output.
Only completed provider steps enter history. Tools execute internally and may
lead to another turn; `Turn` starts at 1 for each invocation. Concatenating every
text chunk may include commentary or repair attempts, so use `FinalOutput` after
successful iteration for the final answer. Approval pauses yield
`ErrApprovalNeeded`; approve/reject as usual, then call `Stream(ctx, nil)`.

As with `Run`, do not operate on the same session concurrently or reenter it from
the iteration body.
