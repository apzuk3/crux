# Testing

`crux.foo/cruxtest` mocks every provider's HTTP API, so agent
tests need no keys or network and still exercise the real request and response
code.

```go
func TestRefundAgent(t *testing.T) {
	reg := crux.NewToolsRegistry() // per test, so names don't clash
	crux.RegisterToolWithRegistry(reg, "lookup_order", "Look up an order",
		func(ctx context.Context, in struct {
			ID string `json:"id"`
		}) (string, *crux.StateDelta, error) {
			return "order " + in.ID + ": refundable", nil, nil
		})

	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("lookup_order", map[string]any{"id": "42"})
	mock.Expect().ReturnText("Order 42 is refundable.")

	agent, err := crux.New("support", crux.ClaudeHaiku4_5,
		append(mock.AgentOptions(), crux.WithToolsRegistry([]string{"lookup_order"}, reg))...)
	require.NoError(t, err)
	session, err := crux.NewSession(t.Context(), agent)
	require.NoError(t, err)

	answer, err := session.Run(t.Context(), "Can I refund order 42?")
	require.NoError(t, err)
	require.Equal(t, "Order 42 is refundable.", answer)
	mock.AssertAllConsumed(t)
	require.Contains(t, mock.Requests()[1].BodyString(), "lookup_order")
}
```

- `cruxtest.NewMock()`; each `mock.Expect()` queues one model response, in
  order: `ReturnText`, `ReturnJSON(v)`, `ReturnToolCall(name, args)`,
  `ReturnToolCalls(...)`, `ReturnRefusal`, `ReturnError(status, body)`,
  `ReturnDecision(map[string]cruxtest.Answer{...})` with `cruxtest.Noul(p)`,
  `cruxtest.Choice(...)`, `cruxtest.Score(...)`; `.WithUsage(...)` sets tokens.
- `mock.AgentOptions()` gives the HTTP client and a fake key; append your own
  options. It works for `crux.NewDecider` too.
- The provider is detected from the request, so use the model you ship with.
- Inspect what was sent with `mock.Requests()` (`BodyString`, `UnmarshalBody`);
  check with `mock.Calls()`, `AssertAllConsumed`, `AssertTurnCount`.
- Tools run for real. Use a `crux.NewToolsRegistry()` per test with
  `RegisterToolWithRegistry` and `crux.WithToolsRegistry` to avoid clashing
  names in the global registry.
