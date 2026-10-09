# Errors

Check errors with `errors.Is`; crux wraps them with context.

| Sentinel | When | What to do |
|---|---|---|
| `crux.ErrApprovalNeeded` | A tool call waits for a human. | `PendingApprovals`, `Approve`/`Reject`, `Resume`. See [approvals.md](approvals.md). |
| `crux.ErrMaxTurns` | The agent used `WithMaxTurns` requests without answering. | Raise the limit or tighten the instructions; `Resume` continues. |
| `crux.ErrRefused` | The model refused. | Show the user; rephrasing may help. |
| `crux.ErrOutputValidation` | The answer doesn't match the output schema. | `WithMaxRepairs(n)`; simplify the schema. |
| `crux.ErrContextTooLong` | The request exceeds the context window even after compaction. | Fork or start a new session; send less. |
| `crux.ErrToolNotFound` | An agent names a tool or toolset nobody registered (from `crux.New`). | Register it before building the agent. |
| `crux.ErrSessionNotFound` | A `Store` has no entries for an ID. | `NewSession` with `WithSessionID` already handles this by starting fresh. |
| `crux.ErrSessionConflict` | Another writer appended to the session first. | Reopen with `WithSessionID` and retry. |

Tool errors are not here: they go to the model, not to you
([tools.md](tools.md)). Provider and network errors are returned as they are;
after any error, `Resume` retries without repeating the input.
