# Approvals

Tools registered with `crux.WithApprovalNeeded(true)` (plus the filesystem and
network tools that change things, and MCP tools not marked read-only) don't run
until a human decides.

```go
crux.RegisterTool("refund", "Refund an order",
	func(ctx context.Context, in struct {
		OrderID string `json:"order_id"`
	}) (string, error) {
		return "refunded " + in.OrderID, nil
	}, crux.WithApprovalNeeded(true))

answer, err := session.Run(ctx, "Refund order 42")
for errors.Is(err, crux.ErrApprovalNeeded) {
	for _, call := range session.PendingApprovals() {
		fmt.Printf("%s wants %s(%s)\n", call.Agent, call.Name, call.Args)
		if userSaysYes(call) {
			err = session.Approve(ctx, call.ID)
		} else {
			err = session.Reject(ctx, call.ID, "not allowed")
		}
		if err != nil { ... }
	}
	answer, err = session.Resume(ctx)
}
```

- `Run` returns `crux.ErrApprovalNeeded` after recording the calls. Other calls
  from the same turn still run.
- `PendingApprovals()` returns the waiting calls, including those made by
  subagents; `call.Agent` names the agent.
- `Approve` lets the call run on the next `Resume`. `Reject` tells the model the
  reason, or that the user declined.
- Decide every pending call, then call `Resume`.
- With a persistent store, approvals survive restarts: reopen the session with
  `WithSessionID`, then `PendingApprovals`, `Approve` and `Resume`.
- `crux.CLI` asks for approvals interactively.
