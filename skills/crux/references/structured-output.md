# Structured output

Make an agent answer with a Go type instead of text.

```go
type Ticket struct {
	Title    string   `json:"title"`
	Priority string   `json:"priority" jsonschema:"enum=low,enum=medium,enum=high"`
	Labels   []string `json:"labels"`
	Assignee *string  `json:"assignee,omitempty"` // optional: may be null
}

agent := crux.Must(crux.New("extractor", crux.OpenAIGPT5_4Mini,
	crux.WithOutputSchemaFrom[Ticket](),
	crux.WithMaxRepairs(2)))
session := crux.MustSession(crux.NewSession(ctx, agent))

var t Ticket
if err := session.RunInto(ctx, &t, emailBody); err != nil { ... }
```

- `crux.WithOutputSchemaFrom[T]()` reflects `T` into a JSON Schema (with
  `invopop/jsonschema`: `json` tags, plus `jsonschema:"..."` tags for enums,
  bounds and descriptions). crux adapts it to each provider's structured-output
  rules, so the same type works everywhere.
- `crux.WithOutputSchema(*jsonschema.Schema)` takes a schema you built.
- Optional fields (pointers or `omitempty`) also allow `null`.
- The answer is validated against the schema. A mismatch fails with
  `crux.ErrOutputValidation`, unless `crux.WithMaxRepairs(n)` lets the model fix
  it (default 0). Repairs don't count as turns.
- `RunInto(ctx, &target, inputs...)`: target first, then inputs. It works
  without a schema too, decoding whatever JSON the model wrote.
- An agent with an output schema can still use tools; the schema applies to its
  final answer.
- OpenAI-style providers need an object at the root: wrap a slice or scalar in a
  struct field.

For labels, routing, yes/no or scores, prefer [decisions.md](decisions.md):
it's cheaper and, on decision models, returns calibrated probabilities.
