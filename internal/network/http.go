package network

import (
	"cmp"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"
)

// httpTransport returns the toolset's shared transport for public-only or
// policy-checked requests. Proxies from the environment are not used: the
// proxy would be dialed instead of the checked host.
func (t *Tools) httpTransport(publicOnly bool) *http.Transport {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.transports == nil {
		t.transports = make(map[bool]*http.Transport)
	}
	if tr, ok := t.transports[publicOnly]; ok {
		return tr
	}
	tr := &http.Transport{
		DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			return t.dial(ctx, network, address, publicOnly)
		},
		ForceAttemptHTTP2:   true,
		TLSHandshakeTimeout: 10 * time.Second,
		IdleConnTimeout:     90 * time.Second,
		MaxIdleConns:        16,
	}
	t.transports[publicOnly] = tr
	return tr
}

// HTTPGetInput holds the arguments of HTTPGet.
type HTTPGetInput struct {
	URL             string            `json:"url" description:"http or https URL"`
	Method          string            `json:"method,omitempty" description:"GET (default) or HEAD"`
	Headers         map[string]string `json:"headers,omitempty" description:"Request headers"`
	FollowRedirects *bool             `json:"follow_redirects,omitempty" description:"Follow redirects, at most 10; default true"`
	TimeoutMS       *int              `json:"timeout_ms,omitempty" description:"Timeout for the whole request in milliseconds; default 30000"`
	MaxBytes        *int              `json:"max_bytes,omitempty" description:"Most body bytes to return; default 65536, at most 1048576"`
}

// HTTPRequestInput holds the arguments of HTTPRequest.
type HTTPRequestInput struct {
	Method          string            `json:"method,omitempty" description:"HTTP method; default GET"`
	URL             string            `json:"url" description:"http or https URL"`
	Headers         map[string]string `json:"headers,omitempty" description:"Request headers"`
	Body            string            `json:"body,omitempty" description:"Request body as text"`
	BodyBase64      string            `json:"body_base64,omitempty" description:"Request body as base64, for binary data"`
	FollowRedirects *bool             `json:"follow_redirects,omitempty" description:"Follow redirects, at most 10; default true"`
	TimeoutMS       *int              `json:"timeout_ms,omitempty" description:"Timeout for the whole request in milliseconds; default 30000"`
	MaxBytes        *int              `json:"max_bytes,omitempty" description:"Most body bytes to return; default 65536, at most 1048576"`
}

// HTTPGet fetches a public URL with GET or HEAD.
func (t *Tools) HTTPGet(ctx context.Context, in HTTPGetInput) (string, error) {
	method := cmp.Or(strings.ToUpper(in.Method), http.MethodGet)
	if method != http.MethodGet && method != http.MethodHead {
		return "", fmt.Errorf("http_get only sends GET or HEAD; use http_request for %s", method)
	}
	return t.doHTTP(ctx, HTTPRequestInput{
		Method: method, URL: in.URL, Headers: in.Headers,
		FollowRedirects: in.FollowRedirects, TimeoutMS: in.TimeoutMS, MaxBytes: in.MaxBytes,
	}, true)
}

// HTTPRequest sends any HTTP request to an allowed host.
func (t *Tools) HTTPRequest(ctx context.Context, in HTTPRequestInput) (string, error) {
	return t.doHTTP(ctx, in, false)
}

func (t *Tools) doHTTP(ctx context.Context, in HTTPRequestInput, publicOnly bool) (string, error) {
	u, err := url.Parse(in.URL)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return "", fmt.Errorf("url %q must be an http or https URL", in.URL)
	}
	body, err := httpRequestBody(in)
	if err != nil {
		return "", err
	}
	limit, err := optionalLimit(in.MaxBytes, netDefaultHTTPBytes, netMaxHTTPBytes)
	if err != nil {
		return "", err
	}
	timeout, err := optionalTimeout(in.TimeoutMS, netDefaultTimeout)
	if err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	req, err := newHTTPRequest(ctx, in, u, body)
	if err != nil {
		return "", err
	}
	client := newHTTPClient(t.httpTransport(publicOnly), in.FollowRedirects == nil || *in.FollowRedirects)
	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("%s %s: %w", req.Method, in.URL, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, int64(limit)+1))
	if err != nil {
		return "", fmt.Errorf("read response body: %w", err)
	}
	return formatHTTPResponse(resp, u.String(), start, data, limit), nil
}

// optionalLimit returns the max_bytes argument, def when absent, capped at most.
func optionalLimit(p *int, def, most int) (int, error) {
	if p == nil {
		return def, nil
	}
	if *p < 1 {
		return 0, errors.New("max_bytes must be at least 1")
	}
	return min(*p, most), nil
}

// optionalTimeout returns the timeout_ms argument, or def when absent.
func optionalTimeout(p *int, def time.Duration) (time.Duration, error) {
	if p == nil {
		return def, nil
	}
	if *p < 1 {
		return 0, errors.New("timeout_ms must be at least 1")
	}
	return time.Duration(*p) * time.Millisecond, nil
}

func httpRequestBody(in HTTPRequestInput) ([]byte, error) {
	if in.Body != "" && in.BodyBase64 != "" {
		return nil, errors.New("give body or body_base64, not both")
	}
	if in.BodyBase64 != "" {
		body, err := base64.StdEncoding.DecodeString(in.BodyBase64)
		if err != nil {
			return nil, fmt.Errorf("decode body_base64: %w", err)
		}
		return body, nil
	}
	return []byte(in.Body), nil
}

func newHTTPRequest(ctx context.Context, in HTTPRequestInput, u *url.URL, body []byte) (*http.Request, error) {
	var reqBody io.Reader
	if len(body) > 0 {
		reqBody = strings.NewReader(string(body))
	}
	req, err := http.NewRequestWithContext(ctx, cmp.Or(strings.ToUpper(in.Method), http.MethodGet), u.String(), reqBody)
	if err != nil {
		return nil, err
	}
	for key, value := range in.Headers {
		req.Header.Set(key, value)
	}
	if req.Header.Get("User-Agent") == "" {
		req.Header.Set("User-Agent", "crux")
	}
	return req, nil
}

// newHTTPClient returns a client that follows at most 10 redirects, or none
// when follow is false.
func newHTTPClient(transport http.RoundTripper, follow bool) *http.Client {
	return &http.Client{
		Transport: transport,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if !follow {
				return http.ErrUseLastResponse
			}
			if len(via) >= 10 {
				return errors.New("stopped after 10 redirects")
			}
			return nil
		},
	}
}

// formatHTTPResponse shows the status, headers and at most limit bytes of
// the body; data holds up to limit+1 bytes so truncation can be detected.
func formatHTTPResponse(resp *http.Response, requested string, start time.Time, data []byte, limit int) string {
	truncated := len(data) > limit
	if truncated {
		data = data[:limit]
	}
	var out strings.Builder
	fmt.Fprintf(&out, "%s %s (%s)\n", resp.Proto, resp.Status, time.Since(start).Round(time.Millisecond))
	if final := resp.Request.URL.String(); final != requested {
		fmt.Fprintf(&out, "Final URL: %s\n", final)
	}
	writeHeaders(&out, resp.Header)
	out.WriteString("\n")
	switch {
	case len(data) == 0:
		out.WriteString("(empty body)")
	case printable(data):
		out.Write(data)
	default:
		fmt.Fprintf(&out, "(binary body, %s)\nbase64:%s", cmp.Or(resp.Header.Get("Content-Type"), "unknown type"), base64.StdEncoding.EncodeToString(data))
	}
	if truncated {
		shown := fmt.Sprintf("the first %d bytes", limit)
		if resp.ContentLength > 0 {
			shown = fmt.Sprintf("%d of %d bytes", limit, resp.ContentLength)
		}
		fmt.Fprintf(&out, "\n(truncated: showed %s; max_bytes can raise this up to %d)", shown, netMaxHTTPBytes)
	}
	return out.String()
}

func writeHeaders(out *strings.Builder, header http.Header) {
	keys := make([]string, 0, len(header))
	for key := range header {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	for _, key := range keys {
		for _, value := range header[key] {
			fmt.Fprintf(out, "%s: %s\n", key, value)
		}
	}
}

type httpRoute struct {
	Method  string            `json:"method,omitempty" description:"HTTP method to match; any method when empty"`
	Path    string            `json:"path" description:"Path pattern, such as / (everything), /api/ (a subtree), /items/{id} or /exact/{$}"`
	Status  int               `json:"status,omitempty" description:"Response status; default 200"`
	Headers map[string]string `json:"headers,omitempty" description:"Response headers"`
	Body    string            `json:"body,omitempty" description:"Response body"`
}

// HTTPServeInput holds the arguments of HTTPServe.
type HTTPServeInput struct {
	Address       string      `json:"address" description:"host:port to listen on. A port alone, such as :8080, listens on 127.0.0.1; port 0 picks a free one"`
	Routes        []httpRoute `json:"routes,omitempty" description:"Fixed responses by path"`
	DefaultStatus int         `json:"default_status,omitempty" description:"Status for requests no route matches; default 404"`
}

// HTTPServe starts an HTTP server with fixed responses.
func (t *Tools) HTTPServe(ctx context.Context, in HTTPServeInput) (string, error) {
	address, err := listenAddress("tcp", in.Address)
	if err != nil {
		return "", err
	}
	defaultStatus := cmp.Or(in.DefaultStatus, http.StatusNotFound)
	if defaultStatus < 100 || defaultStatus > 999 {
		return "", fmt.Errorf("default_status %d is not an HTTP status", defaultStatus)
	}
	mux := http.NewServeMux()
	for _, route := range in.Routes {
		if err := addRoute(mux, route); err != nil {
			return "", err
		}
	}
	listener, err := (&net.ListenConfig{}).Listen(ctx, "tcp", address)
	if err != nil {
		return "", fmt.Errorf("listen on %s: %w", address, err)
	}

	buf := newNetBuffer()
	h := &netHandle{kind: "http", local: "http://" + listener.Addr().String(), buf: buf}
	server := &http.Server{
		ReadHeaderTimeout: 10 * time.Second,
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(io.LimitReader(r.Body, netMaxLoggedBody+1))
			buf.pushRequest(netMessage{label: "request", data: []byte(formatRequest(r, body)), text: true})
			handler, pattern := mux.Handler(r)
			if pattern == "" {
				w.WriteHeader(defaultStatus)
				return
			}
			handler.ServeHTTP(w, r)
		}),
	}
	h.close = func() error {
		err := server.Close()
		buf.end(errHandleClosed)
		return err
	}
	if err := t.add("http", h); err != nil {
		listener.Close()
		return "", err
	}
	go func() {
		if err := server.Serve(listener); !errors.Is(err, http.ErrServerClosed) {
			buf.end(err)
		}
	}()
	return fmt.Sprintf("%s: serving %d routes on %s; net_read shows the requests", h.id, len(in.Routes), h.local), nil
}

// addRoute adds a fixed response to mux. ServeMux panics on an invalid or
// conflicting pattern, which is returned as an error.
func addRoute(mux *http.ServeMux, route httpRoute) (err error) {
	if !strings.HasPrefix(route.Path, "/") {
		return fmt.Errorf("route path %q must start with /", route.Path)
	}
	status := cmp.Or(route.Status, http.StatusOK)
	if status < 100 || status > 999 {
		return fmt.Errorf("route %s: status %d is not an HTTP status", route.Path, status)
	}
	pattern := route.Path
	if route.Method != "" {
		pattern = strings.ToUpper(route.Method) + " " + route.Path
	}
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("route %q: %v", pattern, r)
		}
	}()
	mux.HandleFunc(pattern, func(w http.ResponseWriter, r *http.Request) {
		for key, value := range route.Headers {
			w.Header().Set(key, value)
		}
		w.WriteHeader(status)
		io.WriteString(w, route.Body)
	})
	return nil
}

func formatRequest(r *http.Request, body []byte) string {
	var out strings.Builder
	fmt.Fprintf(&out, "%s %s %s from %s\n", r.Method, r.URL.RequestURI(), r.Proto, r.RemoteAddr)
	fmt.Fprintf(&out, "Host: %s\n", r.Host)
	writeHeaders(&out, r.Header)
	if len(body) > 0 {
		truncated := len(body) > netMaxLoggedBody
		if truncated {
			body = body[:netMaxLoggedBody]
		}
		out.WriteString("\n")
		if printable(body) {
			out.Write(body)
		} else {
			out.WriteString("base64:" + base64.StdEncoding.EncodeToString(body))
		}
		if truncated {
			fmt.Fprintf(&out, "\n(body truncated to %d bytes)", netMaxLoggedBody)
		}
	}
	return out.String()
}
