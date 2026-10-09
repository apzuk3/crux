# Multi-agent

## Subagents

Give an agent another agent as a tool:

```go
researcher := crux.Must(crux.New("researcher", crux.ClaudeHaiku4_5,
	crux.WithInstructions("Find facts and cite sources."),
	crux.WithWebSearch()))

lead := crux.Must(crux.New("lead", crux.ClaudeSonnet5_5,
	crux.WithSubAgent(researcher, "Researches a question on the web")))
```

- The parent sees a tool named `agent_<name>` (here `agent_researcher`) that
  takes `{"task": "..."}`.
- Each call runs in its own child session. The child's final output (shaped by
  its output schema, if it has one) is the tool result.
- Approvals the child needs surface on the parent's `PendingApprovals`, with
  `call.Agent` set to the child's name; `Approve` and `Resume` on the parent.
- Entry handlers (`WithEntryHandler`) and the store are shared with children.
- `crux.WithoutSubagents()` removes them.

## Spawning agents

Let the model create agents on the fly:

```go
lead := crux.Must(crux.New("lead", crux.ClaudeSonnet5_5,
	crux.WithTools([]string{"search_docs", "read_ticket"}),
	crux.WithAgentSpawning(
		crux.WithSpawnModels(crux.ClaudeHaiku4_5, crux.ClaudeSonnet5_5),
		crux.WithSpawnMaxTurns(5),
		crux.WithSpawnConcurrency(4))))
```

- Adds a `spawn_agent` tool: the model picks a name, instructions, a task and,
  within your limits, tools and a model.
- Defaults come from the parent: its tools (`WithSpawnTools` /
  `WithSpawnToolsRegistry` narrow them), its model, its turn limit.
- Spawned agents can't spawn, have no output schema, and run like subagents
  (approvals surface on the parent).
