package crux

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"slices"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// MCPTransport says how to reach an MCP server. Create one with MCPCommand or
// MCPRemote.
type MCPTransport struct {
	command []string
	url     string
	sdk     mcp.Transport // set by tests
}

// MCPCommand runs an MCP server as a subprocess and talks to it over stdin
// and stdout. Give it credentials with WithMCPEnv.
func MCPCommand(name string, args ...string) MCPTransport {
	return MCPTransport{command: append([]string{name}, args...)}
}

// MCPRemote connects to an MCP server over Streamable HTTP. When the server
// asks for authorization, crux runs the OAuth flow in the browser and caches
// the token; see WithMCPOAuth.
func MCPRemote(url string) MCPTransport {
	return MCPTransport{url: url}
}

// mcpToolNamePattern matches characters MCP allows in tool names but crux
// does not.
var mcpToolNamePattern = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// MCPOption configures an MCP server in ConfigureMCP.
type MCPOption func(*mcpConfig) error

type mcpConfig struct {
	env        []string
	header     http.Header
	httpClient *http.Client
	oauth      *OAuthConfig
	approval   []approvalRule
}

// approvalRule overrides whether the named tools, or all of a toolset's
// tools when none are named, need approval.
type approvalRule struct {
	needed bool
	tools  []string
}

// WithMCPEnv adds "KEY=value" variables to the environment of an MCPCommand
// server, on top of the current process's environment.
func WithMCPEnv(env ...string) MCPOption {
	return func(c *mcpConfig) error {
		for _, kv := range env {
			if !strings.Contains(kv, "=") {
				return fmt.Errorf("environment variable %q must have the form KEY=value", kv)
			}
		}
		c.env = append(c.env, env...)
		return nil
	}
}

// WithMCPHeader sends a header with every request to an MCPRemote server,
// such as "Authorization" with "Bearer <api key>". An Authorization header
// turns off the automatic OAuth flow unless WithMCPOAuth is also given.
func WithMCPHeader(key, value string) MCPOption {
	return func(c *mcpConfig) error {
		if c.header == nil {
			c.header = make(http.Header)
		}
		c.header.Add(key, value)
		return nil
	}
}

// WithMCPHTTPClient sets the HTTP client for an MCPRemote server and its
// OAuth requests.
func WithMCPHTTPClient(client *http.Client) MCPOption {
	return func(c *mcpConfig) error {
		if client == nil {
			return errors.New("http client cannot be nil")
		}
		c.httpClient = client
		return nil
	}
}

// WithMCPApprovalNeeded sets whether calls to the server's tools need
// approval, for the named tools (as the server names them) or for all of
// them when none are named. By default a tool needs approval unless the
// server marks it read-only. Later options win.
func WithMCPApprovalNeeded(needed bool, tools ...string) MCPOption {
	return func(c *mcpConfig) error {
		c.approval = append(c.approval, approvalRule{needed: needed, tools: tools})
		return nil
	}
}

// MCPServer is a connected MCP server whose tools are registered as a
// toolset. Close it when the application is done with it.
type MCPServer struct {
	name    string
	session *mcp.ClientSession
	tools   []string
	closed  atomic.Bool
	once    sync.Once
	err     error
}

// Tools returns the names the server's tools are registered under.
func (s *MCPServer) Tools() []string { return slices.Clone(s.tools) }

// Close disconnects from the server. Its tools stay registered, but calls to
// them fail.
func (s *MCPServer) Close() error {
	s.once.Do(func() {
		s.closed.Store(true)
		s.err = s.session.Close()
	})
	return s.err
}

// ConfigureMCP connects to an MCP server, lists its tools and registers them
// in the default registry as the toolset name, so agents can use them with
// WithMCPs(name). Each tool is registered as "<name>_<tool>". A tool needs
// approval unless the server marks it read-only; see WithMCPApprovalNeeded.
//
// The tools are listed once: tools the server adds later are not seen.
func ConfigureMCP(ctx context.Context, name string, transport MCPTransport, opts ...MCPOption) (*MCPServer, error) {
	return ConfigureMCPWithRegistry(ctx, defaultToolsRegistry, name, transport, opts...)
}

// ConfigureMCPWithRegistry is ConfigureMCP for a custom registry. Agents use
// the tools with WithToolsetsRegistry(registry, name).
func ConfigureMCPWithRegistry(ctx context.Context, registry ToolsRegistry, name string, transport MCPTransport, opts ...MCPOption) (*MCPServer, error) {
	registry.mustBeInitialized()
	if err := validateToolName(name); err != nil {
		return nil, fmt.Errorf("mcp server: %w", err)
	}
	var config mcpConfig
	for _, opt := range opts {
		if err := opt(&config); err != nil {
			return nil, fmt.Errorf("mcp server %q: %w", name, err)
		}
	}

	sdkTransport, stderr, err := config.transport(name, transport)
	if err != nil {
		return nil, fmt.Errorf("mcp server %q: %w", name, err)
	}
	client := mcp.NewClient(&mcp.Implementation{Name: "crux"}, nil)
	session, err := client.Connect(ctx, sdkTransport, nil)
	if err != nil {
		if tail := stderr.String(); tail != "" {
			err = fmt.Errorf("%w\nserver stderr:\n%s", err, tail)
		}
		return nil, fmt.Errorf("connect to mcp server %q: %w", name, err)
	}
	server := &MCPServer{name: name, session: session}

	tools, err := server.register(ctx, registry, &config)
	if err != nil {
		session.Close()
		return nil, fmt.Errorf("mcp server %q: %w", name, err)
	}
	server.tools = tools
	return server, nil
}

func (c *mcpConfig) transport(name string, t MCPTransport) (mcp.Transport, *tailBuffer, error) {
	stderr := &tailBuffer{max: 4 << 10}
	switch {
	case t.sdk != nil:
		return t.sdk, stderr, nil
	case len(t.command) > 0:
		if c.header != nil || c.httpClient != nil || c.oauth != nil {
			return nil, nil, errors.New("WithMCPHeader, WithMCPHTTPClient and WithMCPOAuth only apply to MCPRemote")
		}
		cmd := exec.Command(t.command[0], t.command[1:]...)
		cmd.Env = append(os.Environ(), c.env...)
		cmd.Stderr = stderr
		return &mcp.CommandTransport{Command: cmd}, stderr, nil
	case t.url != "":
		if c.env != nil {
			return nil, nil, errors.New("WithMCPEnv only applies to MCPCommand")
		}
		httpClient := c.httpClient
		if httpClient == nil {
			httpClient = http.DefaultClient
		}
		remote := &mcp.StreamableClientTransport{Endpoint: t.url, HTTPClient: httpClient}
		if c.header != nil {
			base := httpClient.Transport
			if base == nil {
				base = http.DefaultTransport
			}
			withHeaders := *httpClient
			withHeaders.Transport = headerTransport{base: base, header: c.header}
			remote.HTTPClient = &withHeaders
		}
		if c.oauth != nil || c.header.Get("Authorization") == "" {
			var oauth OAuthConfig
			if c.oauth != nil {
				oauth = *c.oauth
			}
			remote.OAuthHandler = newOAuthHandler(name, t.url, oauth, httpClient)
		}
		return remote, stderr, nil
	default:
		return nil, nil, errors.New("transport must be created with MCPCommand or MCPRemote")
	}
}

// register registers the server's tools, checking every one before any is
// registered so a failure leaves the registry unchanged.
func (s *MCPServer) register(ctx context.Context, registry ToolsRegistry, config *mcpConfig) ([]string, error) {
	type pending struct {
		mcpName, name, description string
		schema                     map[string]any
		approval                   bool
	}
	var tools []pending
	byMCPName := make(map[string]int)
	taken := make(map[string]string)
	for tool, err := range s.session.Tools(ctx, nil) {
		if err != nil {
			return nil, fmt.Errorf("list tools: %w", err)
		}
		name := s.name + "_" + mcpToolNamePattern.ReplaceAllString(tool.Name, "_")
		if err := validateToolName(name); err != nil {
			return nil, fmt.Errorf("tool %q: %w", tool.Name, err)
		}
		if other, ok := taken[name]; ok {
			return nil, fmt.Errorf("tools %q and %q both map to the name %q", other, tool.Name, name)
		}
		taken[name] = tool.Name
		schema, err := mcpInputSchema(tool.InputSchema)
		if err == nil {
			_, err = compileToolSchema(schema)
		}
		if err != nil {
			return nil, fmt.Errorf("tool %q: %w", tool.Name, err)
		}
		description := tool.Description
		if description == "" {
			description = tool.Title
		}
		readOnly := tool.Annotations != nil && tool.Annotations.ReadOnlyHint
		byMCPName[tool.Name] = len(tools)
		tools = append(tools, pending{mcpName: tool.Name, name: name, description: description, schema: schema, approval: !readOnly})
	}
	if len(tools) == 0 {
		return nil, errors.New("server has no tools")
	}
	for _, rule := range config.approval {
		if len(rule.tools) == 0 {
			for i := range tools {
				tools[i].approval = rule.needed
			}
		}
		for _, mcpName := range rule.tools {
			i, ok := byMCPName[mcpName]
			if !ok {
				return nil, fmt.Errorf("WithMCPApprovalNeeded: server has no tool %q", mcpName)
			}
			tools[i].approval = rule.needed
		}
	}

	registry.mu.Lock()
	for _, tool := range tools {
		if _, exists := registry.tools[tool.name]; exists {
			registry.mu.Unlock()
			return nil, fmt.Errorf("tool %q is already registered", tool.name)
		}
	}
	registry.mu.Unlock()

	names := make([]string, 0, len(tools))
	err := func() (err error) {
		defer func() {
			if r := recover(); r != nil {
				regErr, ok := r.(toolRegistrationError)
				if !ok {
					panic(r)
				}
				err = regErr.err
			}
		}()
		for _, tool := range tools {
			RegisterToolWithRegistry(registry, tool.name, tool.description, s.call(tool.mcpName),
				WithInputSchema(tool.schema), WithToolset(s.name), WithApprovalNeeded(tool.approval))
			names = append(names, tool.name)
		}
		return nil
	}()
	return names, err
}

// mcpInputSchema adapts a tool's input schema to what providers accept: an
// object schema without a "$schema" keyword.
func mcpInputSchema(schema any) (map[string]any, error) {
	raw, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("encode input schema: %w", err)
	}
	var object map[string]any
	if err := json.Unmarshal(raw, &object); err != nil {
		return nil, fmt.Errorf("input schema must be a JSON object, got %s", raw)
	}
	if object == nil {
		object = map[string]any{}
	}
	delete(object, "$schema")
	if _, ok := object["type"]; !ok {
		object["type"] = "object"
	}
	if object["type"] == "object" {
		if _, ok := object["properties"]; !ok {
			object["properties"] = map[string]any{}
		}
	}
	return inputSchemaFrom(object)
}

func (s *MCPServer) call(tool string) func(ctx context.Context, args json.RawMessage) (string, *StateDelta, error) {
	return func(ctx context.Context, args json.RawMessage) (string, *StateDelta, error) {
		if s.closed.Load() {
			return "", nil, fmt.Errorf("mcp server %q is closed", s.name)
		}
		result, err := s.session.CallTool(ctx, &mcp.CallToolParams{Name: tool, Arguments: args})
		if err != nil {
			return "", nil, fmt.Errorf("mcp server %q: %w", s.name, err)
		}
		output, err := renderMCPResult(result)
		return output, nil, err
	}
}

// renderMCPResult turns a tool result into the text the model sees. Text and
// text resources are passed on; other content is described. A result with no
// text gives its structured content as JSON. A tool error becomes an error.
func renderMCPResult(result *mcp.CallToolResult) (string, error) {
	var parts []string
	text := false
	for _, content := range result.Content {
		switch c := content.(type) {
		case *mcp.TextContent:
			parts = append(parts, c.Text)
			text = true
		case *mcp.ImageContent:
			parts = append(parts, fmt.Sprintf("[image %s, %d bytes]", c.MIMEType, len(c.Data)))
		case *mcp.AudioContent:
			parts = append(parts, fmt.Sprintf("[audio %s, %d bytes]", c.MIMEType, len(c.Data)))
		case *mcp.ResourceLink:
			parts = append(parts, fmt.Sprintf("[resource %s]", c.URI))
		case *mcp.EmbeddedResource:
			if c.Resource == nil {
				continue
			}
			if c.Resource.Text != "" {
				parts = append(parts, c.Resource.Text)
				text = true
			} else {
				parts = append(parts, fmt.Sprintf("[resource %s, %d bytes]", c.Resource.URI, len(c.Resource.Blob)))
			}
		}
	}
	if !text && result.StructuredContent != nil {
		raw, err := json.Marshal(result.StructuredContent)
		if err != nil {
			return "", fmt.Errorf("encode structured content: %w", err)
		}
		parts = append(parts, string(raw))
	}
	output := strings.Join(parts, "\n")
	if result.IsError {
		if output == "" {
			output = "tool failed"
		}
		return "", errors.New(output)
	}
	return output, nil
}

// WithMCPs gives the agent the tools of the MCP servers configured with
// ConfigureMCP under these names. Like WithToolsets, it adds to the tool
// list, so put it after WithTools when using both.
func WithMCPs(names ...string) AgentOption {
	return func(a *Agent) error {
		if err := WithToolsets(names...)(a); err != nil {
			if errors.Is(err, ErrToolNotFound) {
				return fmt.Errorf("%w (configure MCP servers with ConfigureMCP before New)", err)
			}
			return err
		}
		return nil
	}
}

type headerTransport struct {
	base   http.RoundTripper
	header http.Header
}

func (t headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	for key, values := range t.header {
		req.Header[key] = slices.Clone(values)
	}
	return t.base.RoundTrip(req)
}

// tailBuffer keeps the last max bytes written to it, for a server's stderr.
type tailBuffer struct {
	mu  sync.Mutex
	max int
	buf []byte
}

func (b *tailBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.buf = append(b.buf, p...)
	if over := len(b.buf) - b.max; over > 0 {
		b.buf = b.buf[over:]
	}
	return len(p), nil
}

func (b *tailBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return strings.TrimSpace(string(b.buf))
}
