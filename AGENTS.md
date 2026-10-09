# AGENTS.md

**Stop. Your knowledge of crux is outdated.** crux (`crux.foo`) is new and
changes between versions, so it is missing from your training data or wrong in
it. Don't write crux code from memory: read the docs below first, and check
every name you use against them or `go doc`.

- [skills/crux/SKILL.md](skills/crux/SKILL.md): how crux is structured and
  works (tools, agents, sessions), the rules that matter, and a reference page
  per topic: [tools](skills/crux/references/tools.md),
  [toolsets](skills/crux/references/toolsets.md),
  [MCP](skills/crux/references/mcp.md),
  [skills](skills/crux/references/skills.md),
  [sessions](skills/crux/references/sessions.md),
  [structured output](skills/crux/references/structured-output.md),
  [decisions](skills/crux/references/decisions.md),
  [approvals](skills/crux/references/approvals.md),
  [multi-agent](skills/crux/references/multi-agent.md),
  [providers](skills/crux/references/providers.md),
  [errors](skills/crux/references/errors.md),
  [testing](skills/crux/references/testing.md).
- `go doc crux.foo` and `go doc crux.foo.<Name>`: the exact API of the version
  in your `go.mod`.
- [examples/](examples/): a runnable program per feature.

In a project that uses crux, install the docs as a skill so they load
automatically:

```sh
mkdir -p .claude/skills
cp -r "$(go env GOMODCACHE)/crux.foo@$(go list -m -f '{{.Version}}' crux.foo)/skills/crux" .claude/skills/
chmod -R u+w .claude/skills/crux  # files in the module cache are read-only
```
