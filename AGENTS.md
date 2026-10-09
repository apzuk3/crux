# AGENTS.md

**Stop. Your knowledge of crux is outdated.** crux (`crux.foo`) is new and
changes between versions, so it is missing from your training data or wrong in
it. Don't write crux code from memory: read the docs first, and check every
name you use against them or `go doc`.

- [docs/README.md](docs/README.md): how crux is structured and works (tools,
  agents, sessions), the rules that matter, and an index of the topics.
- One page per topic: [tools](docs/tools.md), [toolsets](docs/toolsets.md),
  [MCP](docs/mcp.md), [skills](docs/skills.md), [sessions](docs/sessions.md),
  [structured output](docs/structured-output.md),
  [decisions](docs/decisions.md), [approvals](docs/approvals.md),
  [multi-agent](docs/multi-agent.md), [providers](docs/providers.md),
  [errors](docs/errors.md), [testing](docs/testing.md).
- `go doc crux.foo` and `go doc crux.foo.<Name>`: the exact API of the version
  in your `go.mod`.
- [examples/](examples/): a runnable program per feature.

Online: https://github.com/apzuk3/crux/tree/main/docs

Developing crux itself? Read [CONTRIBUTING.md](CONTRIBUTING.md) instead.
