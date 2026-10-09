# MCP servers

crux connects to Model Context Protocol servers and turns their tools into
ordinary crux tools.

```go
gh, err := crux.ConfigureMCP(ctx, "github",
	crux.MCPCommand("github-mcp-server", "stdio"),
	crux.WithMCPEnv("GITHUB_PERSONAL_ACCESS_TOKEN="+token))
if err != nil { ... }
defer gh.Close()

agent := crux.Must(crux.New("triager", crux.ClaudeSonnet5_5,
	crux.WithMCPs("github")))
```

## How it works

- `ConfigureMCP(ctx, name, transport, opts...)` connects, lists the server's
  tools **once**, and registers each as `<name>_<tool>` (for example
  `github_create_issue`) in the default registry, labelled with toolset `name`.
  Use `ConfigureMCPWithRegistry` for your own registry.
- Agents select them with `crux.WithMCPs(name, ...)`, which adds to the tool
  list like `WithToolsets`, or by full tool name with `WithTools`.
- `server.Tools()` lists the registered names; `server.Close()` disconnects.
- Every name, schema and conflict is checked before anything is registered, so
  a failed `ConfigureMCP` leaves the registry unchanged.
- Results become text: an MCP error result is a tool error for the model.

## Transports

- `crux.MCPCommand(name, args...)` runs a local server over stdin/stdout. Pass
  secrets with `crux.WithMCPEnv("KEY=value")`.
- `crux.MCPRemote(url)` connects over Streamable HTTP. When the server answers
  401, crux runs OAuth in the browser (dynamic client registration, PKCE,
  loopback redirect) and caches the token. Customise it with
  `crux.WithMCPOAuth(crux.OAuthConfig{...})` (client ID/secret, scopes,
  redirect URL, `TokenStore`). Static auth instead:
  `crux.WithMCPHeader("Authorization", "Bearer "+key)`.

## Approvals

A tool needs approval unless the server marks it read-only. Change it with
`crux.WithMCPApprovalNeeded(needed, tools...)` (tool names as the server names
them; none means all). See [approvals.md](approvals.md).

## Not supported yet

Tools added after connecting (`list_changed`), resources, prompts, sampling and
elicitation.
