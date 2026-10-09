---
title: Testing
weight: 120
---

`crux.foo/cruxtest` mocks every provider's HTTP API, so agent tests need no
keys or network and still exercise the real request and response code. You
test the agents your application ships, unchanged: the only thing a test swaps
is the session's HTTP client.

## An application

Agents and tools are defined once, at package level. The code that runs them
takes an `*http.Client`, like any other dependency that talks to the network.

```go
package support

type order struct {
	ID string `json:"id" description:"The order ID"`
}

func lookupOrder(ctx context.Context, in order) (string, error) {
	return "order " + in.ID + ": delivered 3 days ago, refundable", nil
}

func refundOrder(ctx context.Context, in order) (string, error) {
	return "refunded order " + in.ID, nil
}

var Agent *crux.Agent

// Package variables are set before init runs, so an agent that uses tools
// registered in the same package is built after them. Tools registered by
// another package are registered before this one's variables.
func init() {
	crux.RegisterTool("lookup_order", "Look up an order by its ID", lookupOrder)
	crux.RegisterTool("refund_order", "Refund an order", refundOrder,
		crux.WithApprovalNeeded(true))
	Agent = crux.Must(crux.New("support", crux.ClaudeHaiku4_5,
		crux.WithInstructions("You handle refund requests. Look the order up first."),
		crux.WithTools([]string{"lookup_order", "refund_order"}),
	))
}

type Service struct {
	HTTP  *http.Client // nil uses crux's default client
	Store crux.Store
}

func (s *Service) Ask(ctx context.Context, conversation uuid.UUID, question string) (string, error) {
	session, err := crux.NewSession(ctx, Agent,
		crux.WithHTTPClient(s.HTTP),
		crux.WithStore(s.Store),
		crux.WithSessionID(conversation))
	if err != nil {
		return "", err
	}
	return session.Run(ctx, question)
}
```

## Its tests

```go
package support_test

func TestRefundNeedsApproval(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("lookup_order", map[string]any{"id": "42"})
	mock.Expect().ReturnToolCall("refund_order", map[string]any{"id": "42"})

	svc := &support.Service{HTTP: mock.Client(), Store: crux.NewMemoryStore()}
	_, err := svc.Ask(t.Context(), uuid.New(), "Please refund order 42")

	require.ErrorIs(t, err, crux.ErrApprovalNeeded)
	mock.AssertAllConsumed(t)
	// lookup_order ran for real; its output went back to the model.
	require.Contains(t, mock.Requests()[1].BodyString(), "refundable")
}

func TestRateLimit(t *testing.T) {
	mock := cruxtest.NewMock()
	for range 3 { // the first try and two retries
		mock.Expect().ReturnError(429, `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`).
			WithHeader("Retry-After", "0")
	}

	svc := &support.Service{HTTP: mock.Client(), Store: crux.NewMemoryStore()}
	_, err := svc.Ask(t.Context(), uuid.New(), "Refund order 42")
	require.ErrorIs(t, err, crux.ErrRateLimited)
}
```

## The mock

- `cruxtest.NewMock()`; each `mock.Expect()` queues one model response, in
  order: `ReturnText`, `ReturnJSON(v)`, `ReturnToolCall(name, args)`,
  `ReturnToolCalls(...)`, `ReturnRefusal`, `ReturnError(status, body)`,
  `ReturnRaw(status, body)` (exact bytes, such as an SSE stream),
  `ReturnDecision(map[string]cruxtest.Answer{...})` with `cruxtest.Noul(p)`,
  `cruxtest.Choice(...)`, `cruxtest.Score(...)`. `.WithUsage(...)` sets token
  counts and `.WithHeader(key, value)` adds a response header.
- `mock.Client()` is the `*http.Client` to hand to `crux.WithHTTPClient`. No API
  key is needed.
- One client covers a whole session: its subagents, spawned agents and
  compaction requests go through it too, in the order they are made, and
  `Fork` keeps it.
- Deciders: `crux.Decide[T](ctx, decider.WithHTTPClient(mock.Client()), input)`.
- The provider is detected from the request, so use the model you ship with.
- Inspect what was sent with `mock.Requests()` (`BodyString`, `UnmarshalBody`);
  check with `mock.Calls()`, `AssertAllConsumed`, `AssertTurnCount`.

## Tools

Tools run for real. A test of an application uses the tools the application
registers, as above. Only a test that defines tools of its own needs a
registry of its own, so its names don't clash with other tests in the global
one:

```go
reg := crux.NewToolsRegistry()
crux.RegisterToolWithRegistry(reg, "fake_clock", "The time", fakeClock)
agent := crux.Must(crux.New("timer", crux.ClaudeHaiku4_5,
	crux.WithToolsRegistry([]string{"fake_clock"}, reg)))
```
