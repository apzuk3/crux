# Skills

crux agents can use Agent Skills: folders of instructions the model loads only
when it needs them.

```go
if err := crux.AddSkills("./skills"); err != nil { ... }
agent := crux.Must(crux.New("assistant", crux.ClaudeSonnet5_5, crux.WithSkills()))
```

- `crux.AddSkills(dir)` registers every skill in `dir` with the default
  registry. Call it once per process; `dir` must exist and may be empty.
- Each skill is `dir/<name>/SKILL.md`: YAML frontmatter with `name` (lowercase,
  matching the folder) and `description`, then markdown instructions, plus any
  files the instructions refer to.
- `crux.WithSkills()` adds the toolset (`crux.ToolsetSkills`) to the agent. On
  every request the agent is told the skills' names and descriptions, and gets
  three tools: `load_skill`, `load_skill_resource` and `save_skill` (creates or
  updates a skill; needs approval).
- Skills are read from disk on each request, so a saved skill is available
  from the next request on.
- Scripts inside skills are never run.
