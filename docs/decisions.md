---
title: Decisions
weight: 70
---

`crux.Decide[T]` answers typed questions about some input: yes or no, one of a
set of options, or a level on a scale. Use it for classification, routing,
moderation and scoring instead of an agent with an output schema.

```go
type Triage struct {
	Urgent bool   `json:"urgent" description:"Does the customer need help now?" true:"Money or outages at stake" false:"Can wait"`
	Area   string `json:"area" description:"Which team handles it" choices:"billing=Payments and refunds|technical=Bugs and outages|sales"`
	Mood   int    `json:"mood" description:"How frustrated is the customer?" levels:"calm|frustrated|very angry"`
}

decider := crux.MustDecider(crux.NewDecider(crux.Jev))
res, err := crux.Decide[Triage](ctx, decider, ticketText)
// res.Value.Area == "billing"
// res.Confidence["area"] == 0.81, res.Probabilities["area"]["billing"] == 0.88
```

## Field types

`T` is a struct; each exported field is one question, named by its `json` tag:

| Field | Question | Tags |
|---|---|---|
| `bool` | yes or no; true when P(yes) ≥ 0.5 | `true:"…"`, `false:"…"` describe each answer |
| `string` | pick one option | `choices:"a=when to pick a|b|c"`, or a named string type with `Choices() map[string]string` (2–255 options) |
| integer | level on a scale, rounded to the nearest index | `levels:"low|medium|high"` (2–10, lowest first) |
| float | the weighted position, may fall between levels | `levels:"…"` |

`description` is the question's instructions. Any other type (free text,
slices, nested structs) is rejected before a request is sent.

## Models

- **Decision models** answer natively, with calibrated probabilities in
  `Confidence` and `Probabilities`:
  - TypeSafe's Jev: `crux.Jev`, `crux.Jev1_13`, using `TYPESAFE_API_KEY`;
  - Jev through OpenRouter: `crux.OpenRouterDecisionModelJev1_13` (any
    `typesafe/*` model), using `OPENROUTER_API_KEY`.
- **Any other model** (GPT, Claude, Gemini, …) answers through structured
  output. The same `T` works, but `Confidence` and `Probabilities` are nil.

## Details

- `NewDecider(model, opts...)` takes the connection options (`WithAPIKey`,
  `WithBaseURL`, `WithMaxRetries`, `WithProvider`) and
  `WithInstructions` (context for every question). Tools, output schemas and
  compaction are rejected; decision models also reject sampling options.
- The state passed to `Decide` is strings or JSON values (several become an
  array). Decision models take no attachments.
- A `Decider` is stateless and safe for concurrent use. Nothing is logged.
- `d.WithHTTPClient(client)` returns a copy that sends its requests through
  `client` (a proxy, or `cruxtest`'s `mock.Client()` in tests); `d` is unchanged.
- `Decision[T]` also has `Model`, `Usage` and `Cost` (USD, when reported).
- `crux.New` rejects decision models: they can't run agents.
- Too much state fails with `crux.ErrContextTooLong` (Jev's limit is about 32k tokens).
