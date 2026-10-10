---
title: Toolsets
weight: 20
---

A toolset is a group of tools selected together. Each tool is labelled with
`crux.WithToolset(name)`, and agents pick the whole group with
`crux.WithToolsets(name)`.

## Your own

Implement `crux.Toolset` (`Register(registry crux.ToolsRegistry) error`) and add
it with `crux.AddToolset(ts)`, or `crux.AddToolsetWithRegistry(reg, ts)`.
Unlike `RegisterTool`, adding a toolset returns errors instead of panicking.

```go
type billing struct{}

func (billing) Register(reg crux.ToolsRegistry) error {
	crux.RegisterToolWithRegistry(reg, "billing_invoice", "Get an invoice",
		func(ctx context.Context, in struct {
			ID string `json:"id"`
		}) (string, *crux.StateDelta, error) {
			return "invoice " + in.ID, nil, nil
		}, crux.WithToolset("billing"))
	return nil
}

if err := crux.AddToolset(billing{}); err != nil { ... }
agent := crux.Must(crux.New("finance", crux.ClaudeHaiku4_5, crux.WithToolsets("billing")))
```

## Filesystem

`crux.Filesystem(root, opts...)` gives an agent file tools confined to `root`
(no escaping through `..` or symlinks). Toolset name: `crux.ToolsetFilesystem`
(`"filesystem"`).

- Read-only, no approval: `read_file`, `read_multiple_files`, `list_directory`,
  `directory_tree`, `glob`, `search_files_content`.
- Need approval: `write_file`, `edit_file`, `create_directory`,
  `remove_directory`.
- Calls run one at a time in the model's order.
- `root/AGENTS.md` is added to the agent's instructions on every request;
  `crux.WithFilesystemAgentsFile(false)` turns that off.

```go
if err := crux.AddToolset(crux.Filesystem("./workspace")); err != nil { ... }
agent := crux.Must(crux.New("coder", crux.ClaudeSonnet5_5,
	crux.WithToolsets(crux.ToolsetFilesystem)))
```

Read-only tools can read every file under `root`, secrets included, and send
them to the provider. Scope `root` narrowly.

## Network

`crux.Network(opts...)` returns a `*crux.NetworkToolset`; close it when done.
Toolset name: `crux.ToolsetNetwork` (`"network"`).

- No approval: `dns_lookup`, `whois`, `http_get` (public addresses only),
  `net_read`, `net_list`, `net_close`.
- Need approval: `http_request`, `net_connect`, `net_send`, `net_listen`,
  `http_serve`.
- Private, loopback and link-local addresses (cloud metadata included) are
  reachable by default, behind approval; `http_get` never reaches them.
  Options: `crux.WithNetworkPrivate(false)` blocks them for every tool;
  `crux.WithNetworkHosts("*.example.com")` restricts hosts;
  `crux.WithNetworkApprovalNeeded(needed, tools...)` changes approvals.

```go
net := crux.Network()
defer net.Close()
if err := crux.AddToolset(net); err != nil { ... }
agent := crux.Must(crux.New("netops", crux.OpenAIGPT5_4,
	crux.WithToolsets(crux.ToolsetNetwork)))
```

Treat everything read from the network as untrusted input to the model.

## Also toolsets

MCP servers ([mcp.md](mcp.md)) and skills ([skills.md](skills.md)) register as
toolsets too, with their own sugar: `crux.WithMCPs` and `crux.WithSkills`.
