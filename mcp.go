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

	"github.com/apzuk3/crux/internal/mcpclient"
	"github.com/apzuk3/crux/internal/schema"
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
	if err := schema.ValidateToolName(name); err != nil {
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

func (c *mcpConfig) transport(name string, t MCPTransport) (mcp.Transport, *mcpclient.TailBuffer, error) {
	stderr := mcpclient.NewTailBuffer(4 << 10)
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
			withHeaders.Transport = mcpclient.HeaderTransport{Base: base, Header: c.header}
			remote.HTTPClient = &withHeaders
		}
		if c.oauth != nil || c.header.Get("Authorization") == "" {
			var oauth OAuthConfig
			if c.oauth != nil {
				oauth = *c.oauth
			}
			remote.OAuthHandler = mcpclient.NewOAuthHandler(name, t.url, oauth.internal(), httpClient)
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
		if err := schema.ValidateToolName(name); err != nil {
			return nil, fmt.Errorf("tool %q: %w", tool.Name, err)
		}
		if other, ok := taken[name]; ok {
			return nil, fmt.Errorf("tools %q and %q both map to the name %q", other, tool.Name, name)
		}
		taken[name] = tool.Name
		inputSchema, err := mcpInputSchema(tool.InputSchema)
		if err == nil {
			_, err = schema.Compile(inputSchema)
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
		tools = append(tools, pending{mcpName: tool.Name, name: name, description: description, schema: inputSchema, approval: !readOnly})
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
func mcpInputSchema(value any) (map[string]any, error) {
	raw, err := json.Marshal(value)
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
	return schema.FromValue(object)
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
		output, err := mcpclient.RenderResult(result)
		return output, nil, err
	}
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

// OAuthConfig customises how crux authorizes with an MCPRemote server. The
// zero value works for servers that support dynamic client registration: crux
// registers itself, opens the browser on the server's login page, receives
// the code on a loopback address and caches the token in a file.
type OAuthConfig struct {
	// ClientID and ClientSecret identify a client registered with the
	// authorization server ahead of time. Leave them empty to register
	// dynamically.
	ClientID     string
	ClientSecret string
	// Scopes to request. Empty uses the scopes the server asks for.
	Scopes []string
	// RedirectURL receives the authorization code. It must be a loopback
	// http URL such as "http://127.0.0.1:8085/callback". Empty picks a free
	// port; a pre-registered client usually needs a fixed one.
	RedirectURL string
	// OpenURL shows the user the login page. The default prints the URL to
	// stderr and opens the browser.
	OpenURL func(ctx context.Context, url string) error
	// TokenStore keeps tokens between runs. The default stores them under
	// the user's config directory (see NewFileTokenStore).
	TokenStore TokenStore
}

// WithMCPOAuth customises the OAuth flow of an MCPRemote server. Without it,
// the flow still runs with the zero OAuthConfig when the server asks for
// authorization.
func WithMCPOAuth(config OAuthConfig) MCPOption {
	return func(c *mcpConfig) error {
		if config.RedirectURL != "" {
			if _, err := mcpclient.ParseLoopbackURL(config.RedirectURL); err != nil {
				return fmt.Errorf("OAuth redirect URL: %w", err)
			}
		}
		if config.ClientSecret != "" && config.ClientID == "" {
			return errors.New("OAuth client secret needs a client ID")
		}
		c.oauth = &config
		return nil
	}
}

// TokenStore keeps OAuth credentials for MCP servers, keyed by the server's
// URL. The data holds secrets (access and refresh tokens), so store it as
// such. Load returns nil data and no error when nothing is stored.
type TokenStore interface {
	Load(ctx context.Context, key string) ([]byte, error)
	Save(ctx context.Context, key string, data []byte) error
}

// NewFileTokenStore stores credentials in dir, one file per server, readable
// only by the current user. An empty dir uses crux/mcp under
// os.UserConfigDir.
func NewFileTokenStore(dir string) TokenStore {
	return mcpclient.NewFileTokenStore(dir)
}

func (c OAuthConfig) internal() mcpclient.OAuthConfig {
	return mcpclient.OAuthConfig{
		ClientID:     c.ClientID,
		ClientSecret: c.ClientSecret,
		Scopes:       c.Scopes,
		RedirectURL:  c.RedirectURL,
		OpenURL:      c.OpenURL,
		TokenStore:   c.TokenStore,
	}
}
