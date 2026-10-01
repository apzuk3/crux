package crux

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/require"
	"golang.org/x/oauth2"
)

type echoArgs struct {
	Text string `json:"text" jsonschema:"the text to echo"`
}

// newTestMCPServer returns an MCP server with tools covering each kind of
// result and annotation.
func newTestMCPServer() *mcp.Server {
	server := mcp.NewServer(&mcp.Implementation{Name: "test"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "echo", Description: "Echo text", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true}},
		func(ctx context.Context, req *mcp.CallToolRequest, in echoArgs) (*mcp.CallToolResult, any, error) {
			return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "echo: " + in.Text}}}, nil, nil
		})
	server.AddTool(&mcp.Tool{Name: "items.delete", Description: "Delete an item", InputSchema: json.RawMessage(`{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"]}`)},
		func(ctx context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "deleted " + string(req.Params.Arguments)}}}, nil
		})
	server.AddTool(&mcp.Tool{Name: "fail", InputSchema: json.RawMessage(`{"type":"object"}`)},
		func(ctx context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			return &mcp.CallToolResult{IsError: true, Content: []mcp.Content{&mcp.TextContent{Text: "no such item"}}}, nil
		})
	server.AddTool(&mcp.Tool{Name: "stats", InputSchema: json.RawMessage(`{"type":"object"}`)},
		func(ctx context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			return &mcp.CallToolResult{
				Content:           []mcp.Content{&mcp.ImageContent{MIMEType: "image/png", Data: []byte("png")}},
				StructuredContent: map[string]any{"count": 3},
			}, nil
		})
	return server
}

func inMemoryMCP(t *testing.T, server *mcp.Server) MCPTransport {
	t.Helper()
	client, serverSide := mcp.NewInMemoryTransports()
	session, err := server.Connect(t.Context(), serverSide, nil)
	require.NoError(t, err)
	t.Cleanup(func() { session.Close() })
	return MCPTransport{sdk: client}
}

func callTool(t *testing.T, reg ToolsRegistry, name, args string) (string, error) {
	t.Helper()
	tools, err := reg.selected([]string{name})
	require.NoError(t, err)
	output, _, err := invokeTool(t.Context(), tools[0], json.RawMessage(args))
	return output, err
}

func TestConfigureMCPRegistersTools(t *testing.T) {
	reg := NewToolsRegistry()
	srv, err := ConfigureMCPWithRegistry(t.Context(), reg, "test", inMemoryMCP(t, newTestMCPServer()))
	require.NoError(t, err)
	t.Cleanup(func() { srv.Close() })

	require.ElementsMatch(t, []string{"test_echo", "test_items_delete", "test_fail", "test_stats"}, srv.Tools())
	tools, err := reg.inToolsets([]string{"test"})
	require.NoError(t, err)
	approval := map[string]bool{}
	for _, tool := range tools {
		approval[tool.name] = tool.approvalNeeded
	}
	require.Equal(t, map[string]bool{"test_echo": false, "test_items_delete": true, "test_fail": true, "test_stats": true}, approval)

	echo, err := reg.selected([]string{"test_echo"})
	require.NoError(t, err)
	require.Equal(t, "Echo text", echo[0].description)
	require.Equal(t, "object", echo[0].schema["type"])
	require.Contains(t, echo[0].schema["properties"], "text")
	require.NotContains(t, echo[0].schema, "$schema")
}

func TestMCPToolCalls(t *testing.T) {
	reg := NewToolsRegistry()
	srv, err := ConfigureMCPWithRegistry(t.Context(), reg, "test", inMemoryMCP(t, newTestMCPServer()))
	require.NoError(t, err)

	out, err := callTool(t, reg, "test_echo", `{"text":"hi"}`)
	require.NoError(t, err)
	require.Equal(t, "echo: hi", out)

	out, err = callTool(t, reg, "test_items_delete", `{"id":7}`)
	require.NoError(t, err)
	require.Equal(t, `deleted {"id":7}`, out)

	_, err = callTool(t, reg, "test_items_delete", `{"id":"seven"}`)
	require.ErrorContains(t, err, "invalid arguments", "arguments are checked against the server's schema")

	_, err = callTool(t, reg, "test_fail", `{}`)
	require.EqualError(t, err, "no such item")

	out, err = callTool(t, reg, "test_stats", ``)
	require.NoError(t, err)
	require.Equal(t, "[image image/png, 3 bytes]\n{\"count\":3}", out)

	require.NoError(t, srv.Close())
	require.NoError(t, srv.Close())
	_, err = callTool(t, reg, "test_echo", `{"text":"hi"}`)
	require.ErrorContains(t, err, `mcp server "test" is closed`)
}

func TestMCPApprovalOverrides(t *testing.T) {
	reg := NewToolsRegistry()
	_, err := ConfigureMCPWithRegistry(t.Context(), reg, "test", inMemoryMCP(t, newTestMCPServer()),
		WithMCPApprovalNeeded(false), WithMCPApprovalNeeded(true, "echo"))
	require.NoError(t, err)
	tools, err := reg.inToolsets([]string{"test"})
	require.NoError(t, err)
	for _, tool := range tools {
		require.Equal(t, tool.name == "test_echo", tool.approvalNeeded, tool.name)
	}

	_, err = ConfigureMCPWithRegistry(t.Context(), NewToolsRegistry(), "test", inMemoryMCP(t, newTestMCPServer()),
		WithMCPApprovalNeeded(true, "nope"))
	require.ErrorContains(t, err, `server has no tool "nope"`)
}

func TestConfigureMCPLeavesRegistryUnchangedOnConflict(t *testing.T) {
	reg := NewToolsRegistry()
	RegisterToolWithRegistry(reg, "test_stats", "taken", func(ctx context.Context, in struct{}) (string, *StateDelta, error) {
		return "", nil, nil
	})
	_, err := ConfigureMCPWithRegistry(t.Context(), reg, "test", inMemoryMCP(t, newTestMCPServer()))
	require.ErrorContains(t, err, `tool "test_stats" is already registered`)
	_, err = reg.inToolsets([]string{"test"})
	require.ErrorIs(t, err, ErrToolNotFound)

	_, err = ConfigureMCPWithRegistry(t.Context(), reg, "bad name", inMemoryMCP(t, newTestMCPServer()))
	require.ErrorContains(t, err, "invalid tool name")
}

func TestWithMCPs(t *testing.T) {
	srv, err := ConfigureMCP(t.Context(), "mcpagent", inMemoryMCP(t, newTestMCPServer()))
	require.NoError(t, err)
	t.Cleanup(func() { srv.Close() })

	agent, err := New("a", OpenAIGPT5_4, WithAPIKey("k"), WithMCPs("mcpagent"))
	require.NoError(t, err)
	require.Len(t, agent.tools, 4)

	_, err = New("a", OpenAIGPT5_4, WithAPIKey("k"), WithMCPs("missing"))
	require.ErrorIs(t, err, ErrToolNotFound)
	require.ErrorContains(t, err, "ConfigureMCP")
}

func TestMCPCommandReportsStderr(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("needs sh")
	}
	_, err := ConfigureMCPWithRegistry(t.Context(), NewToolsRegistry(), "broken",
		MCPCommand("sh", "-c", "echo boom >&2; exit 1"))
	require.Error(t, err)
	require.Contains(t, err.Error(), "boom")
}

func TestMCPHeader(t *testing.T) {
	var auth atomic.Value
	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return newTestMCPServer() }, nil)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth.Store(r.Header.Get("Authorization"))
		handler.ServeHTTP(w, r)
	}))
	t.Cleanup(ts.Close)

	srv, err := ConfigureMCPWithRegistry(t.Context(), NewToolsRegistry(), "test", MCPRemote(ts.URL),
		WithMCPHeader("Authorization", "Bearer key"))
	require.NoError(t, err)
	t.Cleanup(func() { srv.Close() })
	require.Equal(t, "Bearer key", auth.Load())
}

// fakeOAuth is an MCP server behind a minimal OAuth authorization server.
type fakeOAuth struct {
	*httptest.Server
	mu         sync.Mutex
	valid      map[string]bool
	challenges map[string]string // code -> PKCE challenge
	registered atomic.Int32
	refreshed  atomic.Int32
	issued     atomic.Int32
}

func newFakeOAuth(t *testing.T) *fakeOAuth {
	f := &fakeOAuth{valid: map[string]bool{}, challenges: map[string]string{}}
	mcpHandler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return newTestMCPServer() }, nil)
	mux := http.NewServeMux()
	writeJSON := func(w http.ResponseWriter, v any) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(v)
	}
	mux.HandleFunc("/mcp", func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		ok := f.valid[strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")]
		f.mu.Unlock()
		if !ok {
			w.Header().Set("WWW-Authenticate", fmt.Sprintf(`Bearer resource_metadata="%s/.well-known/oauth-protected-resource/mcp"`, f.URL))
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		mcpHandler.ServeHTTP(w, r)
	})
	mux.HandleFunc("/.well-known/oauth-protected-resource/mcp", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{"resource": f.URL + "/mcp", "authorization_servers": []string{f.URL}})
	})
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{
			"issuer":                           f.URL,
			"authorization_endpoint":           f.URL + "/authorize",
			"token_endpoint":                   f.URL + "/token",
			"registration_endpoint":            f.URL + "/register",
			"response_types_supported":         []string{"code"},
			"code_challenge_methods_supported": []string{"S256"},
		})
	})
	mux.HandleFunc("/register", func(w http.ResponseWriter, r *http.Request) {
		f.registered.Add(1)
		var meta map[string]any
		json.NewDecoder(r.Body).Decode(&meta)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]any{"client_id": "client-1", "redirect_uris": meta["redirect_uris"], "token_endpoint_auth_method": "none"})
	})
	mux.HandleFunc("/authorize", func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		if q.Get("client_id") != "client-1" || q.Get("resource") != f.URL+"/mcp" || q.Get("code_challenge_method") != "S256" {
			http.Error(w, "bad request "+r.URL.RawQuery, http.StatusBadRequest)
			return
		}
		code := fmt.Sprintf("code-%d", time.Now().UnixNano())
		f.mu.Lock()
		f.challenges[code] = q.Get("code_challenge")
		f.mu.Unlock()
		redirect, _ := url.Parse(q.Get("redirect_uri"))
		redirect.RawQuery = url.Values{"code": {code}, "state": {q.Get("state")}}.Encode()
		http.Redirect(w, r, redirect.String(), http.StatusFound)
	})
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		r.ParseForm()
		f.mu.Lock()
		defer f.mu.Unlock()
		switch r.Form.Get("grant_type") {
		case "authorization_code":
			sum := sha256.Sum256([]byte(r.Form.Get("code_verifier")))
			if f.challenges[r.Form.Get("code")] != base64.RawURLEncoding.EncodeToString(sum[:]) {
				http.Error(w, `{"error":"invalid_grant"}`, http.StatusBadRequest)
				return
			}
		case "refresh_token":
			if r.Form.Get("refresh_token") != "refresh" {
				http.Error(w, `{"error":"invalid_grant"}`, http.StatusBadRequest)
				return
			}
			f.refreshed.Add(1)
		}
		token := fmt.Sprintf("access-%d", f.issued.Add(1))
		f.valid[token] = true
		writeJSON(w, map[string]any{"access_token": token, "token_type": "Bearer", "refresh_token": "refresh", "expires_in": 3600})
	})
	f.Server = httptest.NewServer(mux)
	t.Cleanup(f.Close)
	return f
}

func TestMCPOAuth(t *testing.T) {
	f := newFakeOAuth(t)
	store := NewFileTokenStore(t.TempDir())
	var opened atomic.Int32
	config := OAuthConfig{
		TokenStore: store,
		OpenURL: func(ctx context.Context, u string) error {
			opened.Add(1)
			go func() {
				resp, err := http.Get(u) // follows the redirect to the loopback listener
				if err == nil {
					resp.Body.Close()
				}
			}()
			return nil
		},
	}
	configure := func(name string) *MCPServer {
		t.Helper()
		reg := NewToolsRegistry()
		srv, err := ConfigureMCPWithRegistry(t.Context(), reg, name, MCPRemote(f.URL+"/mcp"), WithMCPOAuth(config))
		require.NoError(t, err)
		t.Cleanup(func() { srv.Close() })
		out, err := callTool(t, reg, name+"_echo", `{"text":"hi"}`)
		require.NoError(t, err)
		require.Equal(t, "echo: hi", out)
		return srv
	}

	configure("first")
	require.EqualValues(t, 1, opened.Load())
	require.EqualValues(t, 1, f.registered.Load())

	configure("second")
	require.EqualValues(t, 1, opened.Load(), "the cached token is reused")

	// An expired token is refreshed and saved again.
	data, err := store.Load(t.Context(), f.URL+"/mcp")
	require.NoError(t, err)
	var creds oauthCredentials
	require.NoError(t, json.Unmarshal(data, &creds))
	creds.Token.Expiry = time.Now().Add(-time.Hour)
	data, _ = json.Marshal(creds)
	require.NoError(t, store.Save(t.Context(), f.URL+"/mcp", data))

	configure("third")
	require.EqualValues(t, 1, opened.Load())
	require.EqualValues(t, 1, f.refreshed.Load())
	data, err = store.Load(t.Context(), f.URL+"/mcp")
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(data, &creds))
	require.True(t, creds.Token.Expiry.After(time.Now()))

	// A revoked token and refresh token start a new login, reusing the
	// registered client.
	f.mu.Lock()
	f.valid = map[string]bool{}
	f.mu.Unlock()
	creds.Token = &oauth2.Token{AccessToken: "revoked", RefreshToken: "revoked", Expiry: time.Now().Add(-time.Hour)}
	data, _ = json.Marshal(creds)
	require.NoError(t, store.Save(t.Context(), f.URL+"/mcp", data))
	configure("fourth")
	require.EqualValues(t, 2, opened.Load())
	require.EqualValues(t, 1, f.registered.Load(), "the registered client is reused")
}

func TestWithMCPOAuthValidates(t *testing.T) {
	_, err := ConfigureMCPWithRegistry(t.Context(), NewToolsRegistry(), "x", MCPRemote("https://example.com/mcp"),
		WithMCPOAuth(OAuthConfig{RedirectURL: "https://example.com/callback"}))
	require.ErrorContains(t, err, "loopback")
	_, err = ConfigureMCPWithRegistry(t.Context(), NewToolsRegistry(), "x", MCPCommand("true"),
		WithMCPHeader("a", "b"))
	require.ErrorContains(t, err, "only apply to MCPRemote")
}
