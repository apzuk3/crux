package network

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/stretchr/testify/require"
)

func TestPublicIP(t *testing.T) {
	for _, ip := range []string{"127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1", "169.254.169.254", "100.64.0.1", "0.0.0.0", "::1", "fd00::1", "fe80::1", "::ffff:127.0.0.1", "224.0.0.1"} {
		require.False(t, publicIP(netip.MustParseAddr(ip).Unmap()), ip)
	}
	for _, ip := range []string{"8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"} {
		require.True(t, publicIP(netip.MustParseAddr(ip)), ip)
	}
}

func TestBufferLimits(t *testing.T) {
	b := newNetBuffer()
	b.push(netMessage{data: make([]byte, netMaxBuffer)})
	b.push(netMessage{data: []byte("x")})
	require.Equal(t, 1, b.dropped)

	b = newNetBuffer()
	require.True(t, b.write(make([]byte, netMaxBuffer)))
	written := make(chan bool)
	go func() { written <- b.write([]byte("more")) }()
	select {
	case <-written:
		t.Fatal("write did not wait for room in a full buffer")
	case <-time.After(50 * time.Millisecond):
	}
	close(b.done)
	require.False(t, <-written)
}

func TestReverseName(t *testing.T) {
	require.Equal(t, "4.3.2.1.in-addr.arpa.", reverseName(netip.MustParseAddr("1.2.3.4")))
	require.Equal(t, "1.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.8.b.d.0.1.0.0.2.ip6.arpa.", reverseName(netip.MustParseAddr("2001:db8::1")))
}

func TestWhoisReferral(t *testing.T) {
	require.Equal(t, "whois.verisign-grs.com", whoisReferral("x\nwhois:        whois.verisign-grs.com\n"))
	require.Equal(t, "rwhois.example.net:4321", whoisReferral("ReferralServer:  whois://rwhois.example.net:4321\n"))
	require.Equal(t, "", whoisReferral("ReferralServer:  rwhois://rwhois.example.net:4321\n"))
	require.Equal(t, "whois.arin.net", whoisReferral("refer:\nRegistrar WHOIS Server: whois.arin.net\n"))
	require.Equal(t, "", whoisReferral("no colon here\nother: value\n"))
}

func intp(n int) *int { return &n }

// newTestTools returns tools that may reach loopback addresses.
func newTestTools(t *testing.T) *Tools {
	t.Helper()
	tools := New(true, nil)
	require.NoError(t, tools.Init())
	t.Cleanup(func() { tools.Close() })
	return tools
}

// bufferHandle registers a handle whose buffer the test fills by hand.
func bufferHandle(t *testing.T, tools *Tools) *netHandle {
	t.Helper()
	buf := newNetBuffer()
	h := &netHandle{kind: "test", buf: buf}
	h.close = func() error {
		buf.end(errHandleClosed)
		return nil
	}
	require.NoError(t, tools.add("test", h))
	return h
}

func TestRead(t *testing.T) {
	tests := []struct {
		name string
		fill func(b *netBuffer)
		in   ReadInput
		want string
	}{
		{
			name: "messages",
			fill: func(b *netBuffer) {
				b.push(netMessage{label: "text", data: []byte("hello"), text: true})
				b.push(netMessage{label: "from 10.0.0.1:53", data: []byte{0, 1}})
			},
			want: "[text] hello\n[from 10.0.0.1:53] base64:AAE=",
		},
		{
			name: "messages up to max_bytes",
			fill: func(b *netBuffer) {
				b.push(netMessage{label: "a", data: []byte("1234")})
				b.push(netMessage{label: "b", data: []byte("5678")})
			},
			in:   ReadInput{MaxBytes: intp(6)},
			want: "[a] 1234\n\n(4 more bytes buffered; read again)",
		},
		{
			name: "stream",
			fill: func(b *netBuffer) { b.write([]byte("hello world")) },
			want: "hello world",
		},
		{
			name: "stream up to max_bytes",
			fill: func(b *netBuffer) { b.write([]byte("hello world")) },
			in:   ReadInput{MaxBytes: intp(5)},
			want: "hello\n(6 more bytes buffered; read again)",
		},
		{
			name: "until",
			fill: func(b *netBuffer) { b.write([]byte("line one\nline two\n")) },
			in:   ReadInput{Until: "\n"},
			want: "line one\n\n(9 more bytes buffered; read again)",
		},
		{
			name: "until beyond max_bytes",
			fill: func(b *netBuffer) { b.write([]byte("abcdef\n")) },
			in:   ReadInput{Until: "\n", MaxBytes: intp(3)},
			want: "abc\n(4 more bytes buffered; read again)",
		},
		{
			name: "until not arrived returns what is there",
			fill: func(b *netBuffer) { b.write([]byte("abc")) },
			in:   ReadInput{Until: "\n", TimeoutMS: intp(10)},
			want: "abc",
		},
		{
			name: "nothing",
			in:   ReadInput{TimeoutMS: intp(10)},
			want: "(nothing received within 10ms)",
		},
		{
			name: "closed by the peer",
			fill: func(b *netBuffer) { b.end(io.EOF) },
			in:   ReadInput{TimeoutMS: intp(10)},
			want: "(nothing received within 10ms)\n(closed by the peer)",
		},
		{
			name: "ended with an error",
			fill: func(b *netBuffer) { b.end(errors.New("boom")) },
			in:   ReadInput{TimeoutMS: intp(10)},
			want: "(nothing received within 10ms)\n(ended: boom)",
		},
		{
			name: "dropped messages",
			fill: func(b *netBuffer) {
				b.dropped = 2
				b.push(netMessage{label: "m", data: []byte("x")})
			},
			want: "[m] x\n\n(2 messages were dropped because the buffer was full)",
		},
		{
			name: "hex",
			fill: func(b *netBuffer) { b.write([]byte("hi")) },
			in:   ReadInput{Encoding: "hex"},
			want: "hex:6869",
		},
		{
			name: "binary shown as base64",
			fill: func(b *netBuffer) { b.write([]byte{0, 1}) },
			want: "base64:AAE=",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tools := newTestTools(t)
			h := bufferHandle(t, tools)
			if tt.fill != nil {
				tt.fill(h.buf)
			}
			tt.in.ID = h.id
			out, err := tools.Read(t.Context(), tt.in)
			require.NoError(t, err)
			require.Equal(t, tt.want, out)
		})
	}
}

func TestReadErrors(t *testing.T) {
	tools := newTestTools(t)
	h := bufferHandle(t, tools)
	for _, tt := range []struct {
		in   ReadInput
		want string
	}{
		{ReadInput{ID: "nope"}, `no open handle "nope"`},
		{ReadInput{ID: h.id, TimeoutMS: intp(-1)}, "timeout_ms cannot be negative"},
		{ReadInput{ID: h.id, MaxBytes: intp(0)}, "max_bytes must be at least 1"},
		{ReadInput{ID: h.id, Encoding: "utf7"}, `unknown encoding "utf7"`},
	} {
		_, err := tools.Read(t.Context(), tt.in)
		require.ErrorContains(t, err, tt.want)
	}
}

func TestReadWaits(t *testing.T) {
	tools := newTestTools(t)
	h := bufferHandle(t, tools)
	go func() {
		time.Sleep(20 * time.Millisecond)
		h.buf.write([]byte("late"))
	}()
	out, err := tools.Read(t.Context(), ReadInput{ID: h.id, TimeoutMS: intp(5000)})
	require.NoError(t, err)
	require.Equal(t, "late", out)

	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
	defer cancel()
	_, err = tools.Read(ctx, ReadInput{ID: h.id, TimeoutMS: intp(5000)})
	require.ErrorIs(t, err, context.DeadlineExceeded)
}

func httpTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/text", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.Header().Set("X-Agent", r.UserAgent())
		io.WriteString(w, "hello")
	})
	mux.HandleFunc("/echo", func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		fmt.Fprintf(w, "%s %s %s", r.Method, r.Header.Get("X-Token"), body)
	})
	mux.HandleFunc("/binary", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Write([]byte{0, 1, 2})
	})
	mux.HandleFunc("/empty", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})
	mux.HandleFunc("/redirect", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/text", http.StatusFound)
	})
	mux.HandleFunc("/loop", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/loop", http.StatusFound)
	})
	mux.HandleFunc("/long", func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, "0123456789")
	})
	mux.HandleFunc("/slow", func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(5 * time.Second):
		case <-r.Context().Done():
		}
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func TestHTTPRequest(t *testing.T) {
	srv := httpTestServer(t)
	tools := newTestTools(t)
	tests := []struct {
		name     string
		in       HTTPRequestInput
		prefix   string
		contains []string
		suffix   string
		absent   string
	}{
		{
			name:     "get",
			in:       HTTPRequestInput{URL: srv.URL + "/text"},
			prefix:   "HTTP/1.1 200 OK (",
			contains: []string{"\nContent-Type: text/plain\n", "\nX-Agent: crux\n"},
			suffix:   "\n\nhello",
		},
		{
			name:   "post with headers and body",
			in:     HTTPRequestInput{Method: "post", URL: srv.URL + "/echo", Headers: map[string]string{"X-Token": "tok"}, Body: "data"},
			prefix: "HTTP/1.1 200 OK (",
			suffix: "\n\nPOST tok data",
		},
		{
			name:   "body_base64",
			in:     HTTPRequestInput{Method: "PUT", URL: srv.URL + "/echo", BodyBase64: base64.StdEncoding.EncodeToString([]byte("raw"))},
			suffix: "\n\nPUT  raw",
		},
		{
			name:   "binary body",
			in:     HTTPRequestInput{URL: srv.URL + "/binary"},
			suffix: "\n\n(binary body, application/octet-stream)\nbase64:AAEC",
		},
		{
			name:   "empty body",
			in:     HTTPRequestInput{URL: srv.URL + "/empty"},
			prefix: "HTTP/1.1 204 No Content (",
			suffix: "\n\n(empty body)",
		},
		{
			name:     "redirect followed",
			in:       HTTPRequestInput{URL: srv.URL + "/redirect"},
			prefix:   "HTTP/1.1 200 OK (",
			contains: []string{"\nFinal URL: " + srv.URL + "/text\n"},
			suffix:   "\n\nhello",
		},
		{
			name:   "redirect not followed",
			in:     HTTPRequestInput{URL: srv.URL + "/redirect", FollowRedirects: new(bool)},
			prefix: "HTTP/1.1 302 Found (",
			absent: "Final URL",
		},
		{
			name:   "truncated",
			in:     HTTPRequestInput{URL: srv.URL + "/long", MaxBytes: intp(4)},
			suffix: "\n\n0123\n(truncated: showed 4 of 10 bytes; max_bytes can raise this up to 1048576)",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, err := tools.HTTPRequest(t.Context(), tt.in)
			require.NoError(t, err)
			if tt.prefix != "" {
				require.True(t, strings.HasPrefix(out, tt.prefix), out)
			}
			for _, s := range tt.contains {
				require.Contains(t, out, s)
			}
			if tt.suffix != "" {
				require.True(t, strings.HasSuffix(out, tt.suffix), out)
			}
			if tt.absent != "" {
				require.NotContains(t, out, tt.absent)
			}
		})
	}
}

func TestHTTPRequestErrors(t *testing.T) {
	srv := httpTestServer(t)
	tools := newTestTools(t)
	for _, tt := range []struct {
		in   HTTPRequestInput
		want string
	}{
		{HTTPRequestInput{URL: "ftp://example.com/x"}, "must be an http or https URL"},
		{HTTPRequestInput{URL: "http://"}, "must be an http or https URL"},
		{HTTPRequestInput{URL: srv.URL, Body: "a", BodyBase64: "YQ=="}, "give body or body_base64, not both"},
		{HTTPRequestInput{URL: srv.URL, BodyBase64: "!!"}, "decode body_base64"},
		{HTTPRequestInput{URL: srv.URL, MaxBytes: intp(0)}, "max_bytes must be at least 1"},
		{HTTPRequestInput{URL: srv.URL, TimeoutMS: intp(0)}, "timeout_ms must be at least 1"},
		{HTTPRequestInput{URL: srv.URL + "/loop"}, "stopped after 10 redirects"},
		{HTTPRequestInput{URL: srv.URL + "/slow", TimeoutMS: intp(20)}, "GET " + srv.URL + "/slow"},
	} {
		_, err := tools.HTTPRequest(t.Context(), tt.in)
		require.ErrorContains(t, err, tt.want)
	}

	_, err := tools.HTTPGet(t.Context(), HTTPGetInput{URL: srv.URL + "/text"})
	require.ErrorContains(t, err, "not a public address")
	_, err = tools.HTTPGet(t.Context(), HTTPGetInput{URL: srv.URL + "/text", Method: "POST"})
	require.ErrorContains(t, err, "http_get only sends GET or HEAD")
}

// webSocketTestServer echoes messages, prefixed with the X-Prefix handshake
// header, and returns its ws:// URL.
func webSocketTestServer(t *testing.T) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/forbidden" {
			http.Error(w, "no", http.StatusForbidden)
			return
		}
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer c.CloseNow()
		if r.URL.Path == "/bye" {
			c.Close(websocket.StatusGoingAway, "bye")
			return
		}
		prefix := []byte(r.Header.Get("X-Prefix"))
		for {
			kind, data, err := c.Read(r.Context())
			if err != nil {
				return
			}
			if err := c.Write(r.Context(), kind, append(prefix, data...)); err != nil {
				return
			}
		}
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http")
}

func TestConnectWebSocket(t *testing.T) {
	wsURL := webSocketTestServer(t)
	tools := newTestTools(t)
	ctx := t.Context()

	out, err := tools.Connect(ctx, ConnectInput{Protocol: "ws", Address: wsURL + "/echo", Headers: map[string]string{"X-Prefix": "echo:"}})
	require.NoError(t, err)
	require.Equal(t, "ws-1: connected to "+wsURL+"/echo (101 Switching Protocols)", out)

	out, err = tools.Send(ctx, SendInput{ID: "ws-1", Data: "hello"})
	require.NoError(t, err)
	require.Equal(t, "sent 5 bytes on ws-1", out)
	out, err = tools.Read(ctx, ReadInput{ID: "ws-1", TimeoutMS: intp(5000)})
	require.NoError(t, err)
	require.Equal(t, "[text] echo:hello", out)

	_, err = tools.Send(ctx, SendInput{ID: "ws-1", Data: "0001", Encoding: "hex", Binary: true})
	require.NoError(t, err)
	out, err = tools.Read(ctx, ReadInput{ID: "ws-1", TimeoutMS: intp(5000)})
	require.NoError(t, err)
	require.Equal(t, "[binary] base64:"+base64.StdEncoding.EncodeToString([]byte("echo:\x00\x01")), out)

	_, err = tools.Send(ctx, SendInput{ID: "ws-1", Data: "x", To: "127.0.0.1:1"})
	require.ErrorContains(t, err, "to is only for UDP listeners")

	out, err = tools.List(ctx, ListInput{})
	require.NoError(t, err)
	require.Contains(t, out, "ws-1  ws  ")
	require.Contains(t, out, " -> "+wsURL+"/echo  open")

	out, err = tools.CloseHandle(ctx, CloseInput{ID: "ws-1"})
	require.NoError(t, err)
	require.Equal(t, "closed ws-1", out)

	_, err = tools.Connect(ctx, ConnectInput{Protocol: "ws", Address: wsURL + "/bye"})
	require.NoError(t, err)
	out, err = tools.Read(ctx, ReadInput{ID: "ws-2", TimeoutMS: intp(5000)})
	require.NoError(t, err)
	require.Equal(t, "(nothing received within 5s)\n(ended: WebSocket closed by the peer with status 1001 (StatusGoingAway))", out)
}

func TestConnectWebSocketErrors(t *testing.T) {
	wsURL := webSocketTestServer(t)
	tools := newTestTools(t)
	for _, tt := range []struct {
		in   ConnectInput
		want string
	}{
		{ConnectInput{Protocol: "ws", Address: "http://example.com"}, "must be a ws:// or wss:// URL"},
		{ConnectInput{Protocol: "ws", Address: "ws://"}, "must be a ws:// or wss:// URL"},
		{ConnectInput{Protocol: "ws", Address: "wss://example.com"}, `does not match protocol "ws"`},
		{ConnectInput{Protocol: "ws", Address: wsURL + "/forbidden"}, "403 Forbidden"},
		{ConnectInput{Protocol: "ws", Address: "ws://127.0.0.1:1/x"}, "WebSocket handshake with ws://127.0.0.1:1/x: "},
	} {
		_, err := tools.Connect(t.Context(), tt.in)
		require.ErrorContains(t, err, tt.want)
	}
}
