package crux

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/modelcontextprotocol/go-sdk/auth"
	"github.com/modelcontextprotocol/go-sdk/oauthex"
	"golang.org/x/oauth2"
)

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
			if _, err := parseLoopbackURL(config.RedirectURL); err != nil {
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
	return &fileTokenStore{dir: dir}
}

type fileTokenStore struct{ dir string }

func (s *fileTokenStore) path(key string) (string, error) {
	dir := s.dir
	if dir == "" {
		config, err := os.UserConfigDir()
		if err != nil {
			return "", fmt.Errorf("token store: %w", err)
		}
		dir = filepath.Join(config, "crux", "mcp")
	}
	sum := sha256.Sum256([]byte(key))
	return filepath.Join(dir, hex.EncodeToString(sum[:16])+".json"), nil
}

func (s *fileTokenStore) Load(_ context.Context, key string) ([]byte, error) {
	path, err := s.path(key)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	return data, err
}

func (s *fileTokenStore) Save(_ context.Context, key string, data []byte) error {
	path, err := s.path(key)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("token store: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".token-*")
	if err != nil {
		return fmt.Errorf("token store: %w", err)
	}
	defer os.Remove(tmp.Name())
	if err := tmp.Chmod(0o600); err != nil && runtime.GOOS != "windows" {
		tmp.Close()
		return fmt.Errorf("token store: %w", err)
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return fmt.Errorf("token store: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("token store: %w", err)
	}
	return os.Rename(tmp.Name(), path)
}

// oauthCredentials is what the token store holds for one server.
type oauthCredentials struct {
	Token        *oauth2.Token    `json:"token"`
	TokenURL     string           `json:"token_url"`
	AuthStyle    oauth2.AuthStyle `json:"auth_style,omitempty"`
	ClientID     string           `json:"client_id"`
	ClientSecret string           `json:"client_secret,omitempty"`
	RedirectURL  string           `json:"redirect_url,omitempty"`
	Registered   bool             `json:"registered,omitempty"` // client ID came from dynamic registration
}

func (c *oauthCredentials) config() *oauth2.Config {
	return &oauth2.Config{
		ClientID:     c.ClientID,
		ClientSecret: c.ClientSecret,
		Endpoint:     oauth2.Endpoint{TokenURL: c.TokenURL, AuthStyle: c.AuthStyle},
		RedirectURL:  c.RedirectURL,
	}
}

// oauthTimeout bounds how long Authorize waits for the user to log in.
const oauthTimeout = 10 * time.Minute

// oauthHandler implements the SDK's auth.OAuthHandler: it supplies the
// cached token, refreshing and saving it as needed, and runs the
// authorization code flow with PKCE when the server answers 401.
type oauthHandler struct {
	server   string // the crux server name, for messages
	resource string // the MCP endpoint URL, also the token store key
	config   OAuthConfig
	client   *http.Client

	authMu sync.Mutex // one authorization at a time
	mu     sync.Mutex
	loaded bool
	creds  *oauthCredentials
	source oauth2.TokenSource
}

var _ auth.OAuthHandler = (*oauthHandler)(nil)

func newOAuthHandler(server, resource string, config OAuthConfig, client *http.Client) *oauthHandler {
	if config.TokenStore == nil {
		config.TokenStore = NewFileTokenStore("")
	}
	if config.OpenURL == nil {
		config.OpenURL = openBrowser(server)
	}
	return &oauthHandler{server: server, resource: resource, config: config, client: client}
}

// TokenSource returns the cached token's source, or nil before the first
// authorization so the request goes out without a token.
func (h *oauthHandler) TokenSource(ctx context.Context) (oauth2.TokenSource, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.loaded {
		h.loaded = true
		data, err := h.config.TokenStore.Load(ctx, h.resource)
		if err != nil {
			return nil, fmt.Errorf("load OAuth token for mcp server %q: %w", h.server, err)
		}
		if data != nil {
			var creds oauthCredentials
			if err := json.Unmarshal(data, &creds); err == nil && creds.Token != nil {
				h.setLocked(&creds)
			}
		}
	}
	return h.source, nil
}

func (h *oauthHandler) setLocked(creds *oauthCredentials) {
	h.creds = creds
	ctx := context.WithValue(context.Background(), oauth2.HTTPClient, h.client)
	h.source = &savingTokenSource{handler: h, base: creds.config().TokenSource(ctx, creds.Token), last: creds.Token}
}

func (h *oauthHandler) save(ctx context.Context, creds *oauthCredentials) error {
	data, err := json.Marshal(creds)
	if err != nil {
		return err
	}
	return h.config.TokenStore.Save(ctx, h.resource, data)
}

// savingTokenSource saves refreshed tokens. When a refresh fails it returns
// the stale token, so the server answers 401 and authorization starts again.
type savingTokenSource struct {
	handler *oauthHandler
	base    oauth2.TokenSource
	mu      sync.Mutex
	last    *oauth2.Token
}

func (s *savingTokenSource) Token() (*oauth2.Token, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	token, err := s.base.Token()
	if err != nil {
		return s.last, nil
	}
	if token.AccessToken != s.last.AccessToken {
		s.last = token
		h := s.handler
		h.mu.Lock()
		creds := *h.creds
		h.mu.Unlock()
		creds.Token = token
		_ = h.save(context.Background(), &creds) // the token still works for this run
	}
	return token, nil
}

// Authorize runs the authorization code flow after a 401, or after a 403
// asking for more scopes.
func (h *oauthHandler) Authorize(ctx context.Context, req *http.Request, resp *http.Response) error {
	defer resp.Body.Close()
	defer io.Copy(io.Discard, resp.Body)

	challenges, err := oauthex.ParseWWWAuthenticate(resp.Header.Values("WWW-Authenticate"))
	if err != nil {
		return fmt.Errorf("mcp server %q: parse WWW-Authenticate: %w", h.server, err)
	}
	if resp.StatusCode == http.StatusForbidden && challengeParam(challenges, "error") != "insufficient_scope" {
		return nil // the retry fails with the server's own error
	}

	h.authMu.Lock()
	defer h.authMu.Unlock()
	h.mu.Lock()
	creds := h.creds
	h.mu.Unlock()
	if creds != nil && req.Header.Get("Authorization") != "Bearer "+creds.Token.AccessToken {
		return nil // another request authorized while this one waited
	}

	prm, err := h.resourceMetadata(ctx, challengeParam(challenges, "resource_metadata"))
	if err != nil {
		return fmt.Errorf("mcp server %q: %w", h.server, err)
	}
	asm, err := auth.GetAuthServerMetadata(ctx, prm.AuthorizationServers[0], h.client)
	if err != nil {
		return fmt.Errorf("mcp server %q: authorization server metadata: %w", h.server, err)
	}
	if asm == nil {
		issuer := strings.TrimSuffix(prm.AuthorizationServers[0], "/")
		asm = &oauthex.AuthServerMeta{
			Issuer:                issuer,
			AuthorizationEndpoint: issuer + "/authorize",
			TokenEndpoint:         issuer + "/token",
			RegistrationEndpoint:  issuer + "/register",
		}
	}

	scopes := h.config.Scopes
	if len(scopes) == 0 {
		scopes = strings.Fields(challengeParam(challenges, "scope"))
	}
	if len(scopes) == 0 {
		scopes = prm.ScopesSupported
	}

	listener, redirectURL, err := h.listen(creds)
	if err != nil {
		return fmt.Errorf("mcp server %q: %w", h.server, err)
	}
	defer listener.Close()

	next, err := h.resolveClient(ctx, asm, creds, redirectURL)
	if err != nil {
		return fmt.Errorf("mcp server %q: %w", h.server, err)
	}
	config := next.config()
	config.Endpoint.AuthURL = asm.AuthorizationEndpoint
	config.Scopes = scopes

	verifier := oauth2.GenerateVerifier()
	state := rand.Text()
	authURL := config.AuthCodeURL(state,
		oauth2.S256ChallengeOption(verifier),
		oauth2.SetAuthURLParam("resource", prm.Resource))

	ctx, cancel := context.WithTimeout(ctx, oauthTimeout)
	defer cancel()
	code, err := waitForCode(ctx, listener, redirectURL, state, func() error { return h.config.OpenURL(ctx, authURL) })
	if err != nil {
		return fmt.Errorf("mcp server %q: authorize: %w", h.server, err)
	}

	token, err := config.Exchange(context.WithValue(ctx, oauth2.HTTPClient, h.client), code,
		oauth2.VerifierOption(verifier),
		oauth2.SetAuthURLParam("resource", prm.Resource))
	if err != nil {
		return fmt.Errorf("mcp server %q: token exchange: %w", h.server, err)
	}
	next.Token = token
	if err := h.save(ctx, next); err != nil {
		return fmt.Errorf("mcp server %q: save OAuth token: %w", h.server, err)
	}
	h.mu.Lock()
	h.setLocked(next)
	h.mu.Unlock()
	return nil
}

// resourceMetadata finds the server's protected resource metadata (RFC 9728)
// where the MCP specification says to look. A server without any is its own
// authorization server, as in the 2025-03-26 specification.
func (h *oauthHandler) resourceMetadata(ctx context.Context, metadataURL string) (*oauthex.ProtectedResourceMetadata, error) {
	type candidate struct{ url, resource string }
	var candidates []candidate
	if metadataURL != "" {
		candidates = append(candidates, candidate{metadataURL, h.resource})
	}
	resource, err := url.Parse(h.resource)
	if err != nil {
		return nil, fmt.Errorf("parse server URL: %w", err)
	}
	atPath := *resource
	atPath.Path = "/.well-known/oauth-protected-resource/" + strings.TrimLeft(resource.Path, "/")
	atPath.RawQuery = ""
	atRoot := atPath
	atRoot.Path = "/.well-known/oauth-protected-resource"
	root := *resource
	root.Path, root.RawQuery = "", ""
	candidates = append(candidates, candidate{atPath.String(), h.resource}, candidate{atRoot.String(), root.String()})

	for _, c := range candidates {
		prm, err := oauthex.GetProtectedResourceMetadata(ctx, c.url, c.resource, h.client)
		if err != nil || prm == nil {
			continue
		}
		if len(prm.AuthorizationServers) == 0 {
			return nil, errors.New("protected resource metadata names no authorization server")
		}
		return prm, nil
	}
	return &oauthex.ProtectedResourceMetadata{Resource: h.resource, AuthorizationServers: []string{root.String()}}, nil
}

// listen opens the loopback listener the authorization code is sent to. It
// reuses the port of a dynamically registered client when it can, so the
// registration stays valid.
func (h *oauthHandler) listen(creds *oauthCredentials) (net.Listener, *url.URL, error) {
	redirect := h.config.RedirectURL
	if redirect == "" && creds != nil && creds.Registered {
		if l, u, err := listenOn(creds.RedirectURL); err == nil {
			return l, u, nil
		}
	}
	if redirect == "" {
		redirect = "http://127.0.0.1:0/callback"
	}
	return listenOn(redirect)
}

func listenOn(redirect string) (net.Listener, *url.URL, error) {
	u, err := parseLoopbackURL(redirect)
	if err != nil {
		return nil, nil, err
	}
	listener, err := net.Listen("tcp", u.Host)
	if err != nil {
		return nil, nil, fmt.Errorf("listen for OAuth redirect: %w", err)
	}
	u.Host = net.JoinHostPort(u.Hostname(), fmt.Sprint(listener.Addr().(*net.TCPAddr).Port))
	return listener, u, nil
}

func parseLoopbackURL(raw string) (*url.URL, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return nil, err
	}
	ip := net.ParseIP(u.Hostname())
	if u.Scheme != "http" || (u.Hostname() != "localhost" && (ip == nil || !ip.IsLoopback())) {
		return nil, fmt.Errorf("%q must be an http URL on a loopback address", raw)
	}
	if u.Port() == "" {
		return nil, fmt.Errorf("%q must have a port", raw)
	}
	if u.Path == "" {
		u.Path = "/"
	}
	return u, nil
}

// resolveClient returns the credentials to authorize with: the configured client,
// the stored dynamic registration if it matches, or a new registration.
func (h *oauthHandler) resolveClient(ctx context.Context, asm *oauthex.AuthServerMeta, stored *oauthCredentials, redirect *url.URL) (*oauthCredentials, error) {
	next := &oauthCredentials{TokenURL: asm.TokenEndpoint, RedirectURL: redirect.String()}
	switch {
	case h.config.ClientID != "":
		next.ClientID, next.ClientSecret = h.config.ClientID, h.config.ClientSecret
		next.AuthStyle = tokenAuthStyle(next.ClientSecret, asm.TokenEndpointAuthMethodsSupported)
	case stored != nil && stored.Registered && stored.TokenURL == asm.TokenEndpoint && stored.RedirectURL == next.RedirectURL:
		next.ClientID, next.ClientSecret, next.AuthStyle, next.Registered = stored.ClientID, stored.ClientSecret, stored.AuthStyle, true
	case asm.RegistrationEndpoint != "":
		registration, err := oauthex.RegisterClient(ctx, asm.RegistrationEndpoint, &oauthex.ClientRegistrationMetadata{
			RedirectURIs:            []string{next.RedirectURL},
			TokenEndpointAuthMethod: "none",
			GrantTypes:              []string{"authorization_code", "refresh_token"},
			ResponseTypes:           []string{"code"},
			ClientName:              "crux",
		}, h.client)
		if err != nil {
			return nil, fmt.Errorf("register OAuth client: %w", err)
		}
		next.ClientID, next.ClientSecret, next.Registered = registration.ClientID, registration.ClientSecret, true
		next.AuthStyle = tokenAuthStyle(next.ClientSecret, []string{registration.TokenEndpointAuthMethod})
	default:
		return nil, errors.New("the authorization server does not support dynamic client registration; set OAuthConfig.ClientID")
	}
	return next, nil
}

func tokenAuthStyle(secret string, methods []string) oauth2.AuthStyle {
	switch {
	case secret == "":
		return oauth2.AuthStyleInParams
	case slices.Contains(methods, "client_secret_post") && !slices.Contains(methods, "client_secret_basic"):
		return oauth2.AuthStyleInParams
	default:
		return oauth2.AuthStyleInHeader
	}
}

// waitForCode serves the redirect on listener and returns the authorization
// code once it arrives with the expected state. open shows the login page
// after the listener is ready.
func waitForCode(ctx context.Context, listener net.Listener, redirect *url.URL, state string, open func() error) (string, error) {
	type result struct {
		code string
		err  error
	}
	results := make(chan result, 1)
	server := &http.Server{
		ReadHeaderTimeout: 10 * time.Second,
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != redirect.Path {
				http.NotFound(w, r)
				return
			}
			query := r.URL.Query()
			var res result
			switch {
			case query.Get("state") != state:
				http.Error(w, "Authorization failed: unexpected state.", http.StatusBadRequest)
				return // not ours; keep waiting
			case query.Get("error") != "":
				res.err = fmt.Errorf("%s: %s", query.Get("error"), query.Get("error_description"))
				fmt.Fprintln(w, "Authorization failed. You can close this tab.")
			case query.Get("code") == "":
				res.err = errors.New("redirect has no authorization code")
				fmt.Fprintln(w, "Authorization failed. You can close this tab.")
			default:
				res.code = query.Get("code")
				fmt.Fprintln(w, "Authorized. You can close this tab and return to the application.")
			}
			select {
			case results <- res:
			default:
			}
		}),
	}
	go server.Serve(listener)
	defer server.Close()

	if err := open(); err != nil {
		return "", fmt.Errorf("open login page: %w", err)
	}
	select {
	case res := <-results:
		return res.code, res.err
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

func challengeParam(challenges []oauthex.Challenge, name string) string {
	for _, c := range challenges {
		if c.Scheme == "bearer" && c.Params[name] != "" {
			return c.Params[name]
		}
	}
	return ""
}

// openBrowser prints the login URL and tries to open it in the browser.
func openBrowser(server string) func(ctx context.Context, url string) error {
	return func(ctx context.Context, url string) error {
		fmt.Fprintf(os.Stderr, "Open this URL to authorize the MCP server %q:\n%s\n", server, url)
		var cmd *exec.Cmd
		switch runtime.GOOS {
		case "darwin":
			cmd = exec.CommandContext(ctx, "open", url)
		case "windows":
			cmd = exec.CommandContext(ctx, "rundll32", "url.dll,FileProtocolHandler", url)
		default:
			cmd = exec.CommandContext(ctx, "xdg-open", url)
		}
		if err := cmd.Start(); err == nil {
			go cmd.Wait()
		}
		return nil // the printed URL still works
	}
}
