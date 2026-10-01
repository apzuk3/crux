package crux

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/stretchr/testify/require"
	"golang.org/x/net/dns/dnsmessage"
)

func newTestNetwork(t *testing.T, opts ...NetworkOption) (*NetworkToolset, ToolsRegistry) {
	t.Helper()
	network := Network(opts...)
	reg := NewToolsRegistry()
	require.NoError(t, AddToolsetWithRegistry(reg, network))
	t.Cleanup(func() { network.Close() })
	return network, reg
}

func netCall(t *testing.T, reg ToolsRegistry, name string, args any) (string, error) {
	t.Helper()
	raw, err := json.Marshal(args)
	require.NoError(t, err)
	return callTool(t, reg, name, string(raw))
}

func mustNetCall(t *testing.T, reg ToolsRegistry, name string, args any) string {
	t.Helper()
	out, err := netCall(t, reg, name, args)
	require.NoError(t, err, "%s %v", name, args)
	return out
}

var handleID = regexp.MustCompile(`^([a-z]+-\d+):`)

// openHandle calls a tool that opens a handle and returns its id and the
// address it reports last on the first line.
func openHandle(t *testing.T, reg ToolsRegistry, name string, args any) (id, address string) {
	t.Helper()
	out := mustNetCall(t, reg, name, args)
	m := handleID.FindStringSubmatch(out)
	require.NotNil(t, m, out)
	first := strings.SplitN(out, "\n", 2)[0]
	fields := strings.Fields(strings.TrimSuffix(strings.Split(first, ";")[0], ";"))
	return m[1], fields[len(fields)-1]
}

func TestNetworkApproval(t *testing.T) {
	_, reg := newTestNetwork(t)
	tools, err := reg.inToolsets([]string{ToolsetNetwork})
	require.NoError(t, err)
	require.Len(t, tools, len(netApprovalDefaults))
	for _, tool := range tools {
		require.Equal(t, netApprovalDefaults[tool.name], tool.approvalNeeded, tool.name)
	}

	_, reg = newTestNetwork(t, WithNetworkApprovalNeeded(false), WithNetworkApprovalNeeded(true, NetDNSLookup))
	tools, err = reg.inToolsets([]string{ToolsetNetwork})
	require.NoError(t, err)
	for _, tool := range tools {
		require.Equal(t, tool.name == NetDNSLookup, tool.approvalNeeded, tool.name)
	}

	err = AddToolsetWithRegistry(NewToolsRegistry(), Network(WithNetworkApprovalNeeded(true, "nope")))
	require.ErrorContains(t, err, `unknown tool "nope"`)
	err = AddToolsetWithRegistry(NewToolsRegistry(), Network(WithNetworkHosts("10.0.0.0/99")))
	require.ErrorContains(t, err, "WithNetworkHosts")
}

func TestPublicIP(t *testing.T) {
	for _, ip := range []string{"127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1", "169.254.169.254", "100.64.0.1", "0.0.0.0", "::1", "fd00::1", "fe80::1", "::ffff:127.0.0.1", "224.0.0.1"} {
		require.False(t, publicIP(netip.MustParseAddr(ip).Unmap()), ip)
	}
	for _, ip := range []string{"8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"} {
		require.True(t, publicIP(netip.MustParseAddr(ip)), ip)
	}
}

func TestNetworkHTTP(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/text", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Test", "yes")
		fmt.Fprintf(w, "hello %s %s", r.Method, r.Header.Get("X-Who"))
	})
	mux.HandleFunc("/echo", func(w http.ResponseWriter, r *http.Request) { io.Copy(w, r.Body) })
	mux.HandleFunc("/bin", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte{0, 1, 2, 255}) })
	mux.HandleFunc("/big", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(strings.Repeat("a", 100))) })
	mux.HandleFunc("/redirect", func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, "/text", http.StatusFound) })
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	_, reg := newTestNetwork(t)

	out := mustNetCall(t, reg, NetHTTPRequest, map[string]any{"url": ts.URL + "/text", "headers": map[string]string{"X-Who": "me"}})
	require.Contains(t, out, "200 OK")
	require.Contains(t, out, "X-Test: yes")
	require.True(t, strings.HasSuffix(out, "hello GET me"), out)

	out = mustNetCall(t, reg, NetHTTPRequest, map[string]any{"method": "post", "url": ts.URL + "/echo", "body": "payload"})
	require.True(t, strings.HasSuffix(out, "payload"), out)

	out = mustNetCall(t, reg, NetHTTPRequest, map[string]any{"url": ts.URL + "/bin"})
	require.Contains(t, out, "binary body")
	require.Contains(t, out, "base64:AAEC/w==")

	out = mustNetCall(t, reg, NetHTTPRequest, map[string]any{"url": ts.URL + "/big", "max_bytes": 10})
	require.Contains(t, out, "\naaaaaaaaaa\n(truncated: showed 10 of 100 bytes")

	out = mustNetCall(t, reg, NetHTTPRequest, map[string]any{"url": ts.URL + "/redirect"})
	require.Contains(t, out, "Final URL: "+ts.URL+"/text")
	out = mustNetCall(t, reg, NetHTTPRequest, map[string]any{"url": ts.URL + "/redirect", "follow_redirects": false})
	require.Contains(t, out, "302 Found")

	_, err := netCall(t, reg, NetHTTPRequest, map[string]any{"url": "ftp://example.com"})
	require.ErrorContains(t, err, "http or https")
}

func TestNetworkHTTPGetIsPublicOnly(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, "secret") }))
	t.Cleanup(ts.Close)
	_, reg := newTestNetwork(t) // private addresses allowed for the other tools

	_, err := netCall(t, reg, NetHTTPGet, map[string]any{"url": ts.URL})
	require.ErrorContains(t, err, "not a public address")
	_, err = netCall(t, reg, NetHTTPGet, map[string]any{"url": strings.Replace(ts.URL, "127.0.0.1", "localhost", 1)})
	require.ErrorContains(t, err, "not a public address")
	_, err = netCall(t, reg, NetHTTPGet, map[string]any{"url": ts.URL, "method": "POST"})
	require.ErrorContains(t, err, "use http_request")
}

func TestNetworkPolicy(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, "ok") }))
	t.Cleanup(ts.Close)
	localhost := strings.Replace(ts.URL, "127.0.0.1", "localhost", 1)

	_, reg := newTestNetwork(t, WithNetworkPrivate(false))
	_, err := netCall(t, reg, NetHTTPRequest, map[string]any{"url": ts.URL})
	require.ErrorContains(t, err, "private and local addresses are blocked")
	_, err = netCall(t, reg, NetConnect, map[string]any{"protocol": "unix", "address": "/tmp/x.sock"})
	require.ErrorContains(t, err, "unix sockets are blocked")

	_, reg = newTestNetwork(t, WithNetworkHosts("10.0.0.0/8"))
	_, err = netCall(t, reg, NetHTTPRequest, map[string]any{"url": ts.URL})
	require.ErrorContains(t, err, "not in the allowed hosts")

	for _, hosts := range []string{"127.0.0.0/8", "127.0.0.1", "localhost", "*.localhost"} {
		_, reg = newTestNetwork(t, WithNetworkHosts(hosts))
		target := ts.URL
		if strings.Contains(hosts, "localhost") {
			target = localhost
		}
		if hosts == "*.localhost" {
			_, err = netCall(t, reg, NetHTTPRequest, map[string]any{"url": target})
			require.ErrorContains(t, err, "not in the allowed hosts", "a wildcard does not match the bare name")
			continue
		}
		out := mustNetCall(t, reg, NetHTTPRequest, map[string]any{"url": target})
		require.True(t, strings.HasSuffix(out, "ok"), hosts)
	}
}

func TestNetworkTCP(t *testing.T) {
	_, reg := newTestNetwork(t)
	listener, address := openHandle(t, reg, NetListen, map[string]any{"protocol": "tcp", "address": ":0"})
	require.True(t, strings.HasPrefix(address, "127.0.0.1:"), address)

	client, _ := openHandle(t, reg, NetConnect, map[string]any{"protocol": "tcp", "address": address})
	accepted := mustNetCall(t, reg, NetRead, map[string]any{"id": listener, "timeout_ms": 2000})
	m := regexp.MustCompile(`\[accepted\] (tcp-\d+) from`).FindStringSubmatch(accepted)
	require.NotNil(t, m, accepted)
	server := m[1]

	mustNetCall(t, reg, NetSend, map[string]any{"id": client, "data": "hello\nrest"})
	out := mustNetCall(t, reg, NetRead, map[string]any{"id": server, "until": "\n", "timeout_ms": 2000})
	require.True(t, strings.HasPrefix(out, "hello\n"), out)
	require.Contains(t, out, "more bytes buffered")
	out = mustNetCall(t, reg, NetRead, map[string]any{"id": server, "timeout_ms": 2000})
	require.Equal(t, "rest", out)

	mustNetCall(t, reg, NetSend, map[string]any{"id": server, "data": "00ff", "encoding": "hex"})
	out = mustNetCall(t, reg, NetRead, map[string]any{"id": client, "timeout_ms": 2000, "encoding": "hex"})
	require.Equal(t, "hex:00ff", out)

	out = mustNetCall(t, reg, NetRead, map[string]any{"id": client, "timeout_ms": 50})
	require.Equal(t, "(nothing received within 50ms)", out)

	list := mustNetCall(t, reg, NetList, map[string]any{})
	require.Contains(t, list, listener+"  listen-tcp")
	require.Contains(t, list, client+"  tcp")

	mustNetCall(t, reg, NetClose, map[string]any{"id": client})
	out = mustNetCall(t, reg, NetRead, map[string]any{"id": server, "timeout_ms": 2000})
	require.Contains(t, out, "closed by the peer")
	_, err := netCall(t, reg, NetSend, map[string]any{"id": client, "data": "x"})
	require.ErrorContains(t, err, "no open handle")
	_, err = netCall(t, reg, NetSend, map[string]any{"id": listener, "data": "x"})
	require.ErrorContains(t, err, "cannot send")
}

func TestNetworkUDP(t *testing.T) {
	_, reg := newTestNetwork(t)
	listener, address := openHandle(t, reg, NetListen, map[string]any{"protocol": "udp", "address": "127.0.0.1:0"})
	client, _ := openHandle(t, reg, NetConnect, map[string]any{"protocol": "udp", "address": address})

	mustNetCall(t, reg, NetSend, map[string]any{"id": client, "data": "ping"})
	out := mustNetCall(t, reg, NetRead, map[string]any{"id": listener, "timeout_ms": 2000})
	m := regexp.MustCompile(`^\[from (\S+)\] ping$`).FindStringSubmatch(out)
	require.NotNil(t, m, out)

	_, err := netCall(t, reg, NetSend, map[string]any{"id": listener, "data": "pong"})
	require.ErrorContains(t, err, "needs to")
	mustNetCall(t, reg, NetSend, map[string]any{"id": listener, "data": "pong", "to": m[1]})
	out = mustNetCall(t, reg, NetRead, map[string]any{"id": client, "timeout_ms": 2000})
	require.Equal(t, fmt.Sprintf("[from %s] pong", address), out)
}

func TestNetworkUnixSocket(t *testing.T) {
	dir, err := os.MkdirTemp("", "crux") // short: socket paths are limited to about 100 bytes
	require.NoError(t, err)
	t.Cleanup(func() { os.RemoveAll(dir) })
	path := filepath.Join(dir, "s.sock")

	_, reg := newTestNetwork(t)
	listener, _ := openHandle(t, reg, NetListen, map[string]any{"protocol": "unix", "address": path})
	client, _ := openHandle(t, reg, NetConnect, map[string]any{"protocol": "unix", "address": path})
	accepted := mustNetCall(t, reg, NetRead, map[string]any{"id": listener, "timeout_ms": 2000})
	server := regexp.MustCompile(`unix-\d+`).FindString(accepted)
	require.NotEmpty(t, server, accepted)
	mustNetCall(t, reg, NetSend, map[string]any{"id": client, "data": "over unix"})
	require.Equal(t, "over unix", mustNetCall(t, reg, NetRead, map[string]any{"id": server, "timeout_ms": 2000}))
}

func TestNetworkWebSocket(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer conn.CloseNow()
		conn.Write(r.Context(), websocket.MessageText, []byte("hello "+r.Header.Get("X-Who")))
		for {
			kind, data, err := conn.Read(r.Context())
			if err != nil {
				return
			}
			conn.Write(r.Context(), kind, data)
		}
	}))
	t.Cleanup(ts.Close)
	_, reg := newTestNetwork(t)
	url := "ws" + strings.TrimPrefix(ts.URL, "http")

	_, err := netCall(t, reg, NetConnect, map[string]any{"protocol": "wss", "address": url})
	require.ErrorContains(t, err, "does not match")

	id, _ := openHandle(t, reg, NetConnect, map[string]any{"protocol": "ws", "address": url, "headers": map[string]string{"X-Who": "crux"}})
	require.Equal(t, "[text] hello crux", mustNetCall(t, reg, NetRead, map[string]any{"id": id, "timeout_ms": 2000}))
	mustNetCall(t, reg, NetSend, map[string]any{"id": id, "data": "AAEC", "encoding": "base64", "binary": true})
	require.Equal(t, "[binary] base64:AAEC", mustNetCall(t, reg, NetRead, map[string]any{"id": id, "timeout_ms": 2000}))
}

func TestNetworkHTTPServe(t *testing.T) {
	_, reg := newTestNetwork(t)
	_, err := netCall(t, reg, NetHTTPServe, map[string]any{"address": ":0", "routes": []map[string]any{{"path": "/a"}, {"path": "/a"}}})
	require.ErrorContains(t, err, `route "/a"`)

	id, address := openHandle(t, reg, NetHTTPServe, map[string]any{
		"address": ":0",
		"routes": []map[string]any{
			{"method": "GET", "path": "/hello", "body": `{"hi":true}`, "headers": map[string]string{"Content-Type": "application/json"}},
			{"path": "/teapot", "status": 418},
		},
	})
	require.True(t, strings.HasPrefix(address, "http://127.0.0.1:"), address)

	out := mustNetCall(t, reg, NetHTTPRequest, map[string]any{"url": address + "/hello"})
	require.Contains(t, out, "Content-Type: application/json")
	require.True(t, strings.HasSuffix(out, `{"hi":true}`), out)
	require.Contains(t, mustNetCall(t, reg, NetHTTPRequest, map[string]any{"url": address + "/teapot"}), "418")
	require.Contains(t, mustNetCall(t, reg, NetHTTPRequest, map[string]any{"method": "POST", "url": address + "/nope", "body": "data"}), "404 Not Found")

	log := mustNetCall(t, reg, NetRead, map[string]any{"id": id, "timeout_ms": 2000})
	require.Contains(t, log, "[request] GET /hello HTTP/1.1 from 127.0.0.1:")
	require.Contains(t, log, "[request] POST /nope HTTP/1.1")
	require.Contains(t, log, "\n\ndata")
	mustNetCall(t, reg, NetClose, map[string]any{"id": id})
	_, err = netCall(t, reg, NetHTTPRequest, map[string]any{"url": address + "/hello"})
	require.Error(t, err)
}

func TestNetworkLimits(t *testing.T) {
	network, reg := newTestNetwork(t)
	for range netMaxHandles {
		mustNetCall(t, reg, NetConnect, map[string]any{"protocol": "udp", "address": "127.0.0.1:9"})
	}
	_, err := netCall(t, reg, NetConnect, map[string]any{"protocol": "udp", "address": "127.0.0.1:9"})
	require.ErrorContains(t, err, "too many open handles")
	require.NoError(t, network.Close())
	require.Equal(t, "no open handles", mustNetCall(t, reg, NetList, map[string]any{}))

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

// startDNS serves example.test records over UDP and TCP on one port. Answers
// for big.test. are truncated over UDP.
func startDNS(t *testing.T) string {
	t.Helper()
	var udp net.PacketConn
	var tcp net.Listener
	for range 10 {
		var err error
		udp, err = net.ListenPacket("udp", "127.0.0.1:0")
		require.NoError(t, err)
		tcp, err = net.Listen("tcp", udp.LocalAddr().String())
		if err == nil {
			break
		}
		udp.Close()
	}
	require.NotNil(t, tcp)
	t.Cleanup(func() { udp.Close(); tcp.Close() })

	answer := func(query []byte, overUDP bool) []byte {
		var msg dnsmessage.Message
		if msg.Unpack(query) != nil || len(msg.Questions) == 0 {
			return nil
		}
		q := msg.Questions[0]
		resp := dnsmessage.Message{Header: dnsmessage.Header{ID: msg.ID, Response: true, RecursionAvailable: true}, Questions: msg.Questions}
		rh := dnsmessage.ResourceHeader{Name: q.Name, Type: q.Type, Class: dnsmessage.ClassINET, TTL: 300}
		switch {
		case q.Name.String() == "example.test." && q.Type == dnsmessage.TypeA:
			resp.Answers = append(resp.Answers, dnsmessage.Resource{Header: rh, Body: &dnsmessage.AResource{A: [4]byte{192, 0, 2, 1}}})
		case q.Name.String() == "example.test." && q.Type == dnsmessage.TypeMX:
			resp.Answers = append(resp.Answers, dnsmessage.Resource{Header: rh, Body: &dnsmessage.MXResource{Pref: 10, MX: dnsmessage.MustNewName("mail.example.test.")}})
		case q.Name.String() == "1.2.0.192.in-addr.arpa." && q.Type == dnsmessage.TypePTR:
			resp.Answers = append(resp.Answers, dnsmessage.Resource{Header: rh, Body: &dnsmessage.PTRResource{PTR: dnsmessage.MustNewName("example.test.")}})
		case q.Name.String() == "big.test." && q.Type == dnsmessage.TypeTXT:
			if overUDP {
				resp.Truncated = true
			} else {
				resp.Answers = append(resp.Answers, dnsmessage.Resource{Header: rh, Body: &dnsmessage.TXTResource{TXT: []string{"over tcp"}}})
			}
		default:
			resp.RCode = dnsmessage.RCodeNameError
		}
		packed, _ := resp.Pack()
		return packed
	}
	go func() {
		buf := make([]byte, 65535)
		for {
			n, from, err := udp.ReadFrom(buf)
			if err != nil {
				return
			}
			udp.WriteTo(answer(buf[:n], true), from)
		}
	}()
	go func() {
		for {
			conn, err := tcp.Accept()
			if err != nil {
				return
			}
			go func() {
				defer conn.Close()
				var size [2]byte
				if _, err := io.ReadFull(conn, size[:]); err != nil {
					return
				}
				query := make([]byte, binary.BigEndian.Uint16(size[:]))
				io.ReadFull(conn, query)
				resp := answer(query, false)
				conn.Write(append(binary.BigEndian.AppendUint16(nil, uint16(len(resp))), resp...))
			}()
		}
	}()
	return udp.LocalAddr().String()
}

func TestNetworkDNS(t *testing.T) {
	server := startDNS(t)
	_, reg := newTestNetwork(t)

	out := mustNetCall(t, reg, NetDNSLookup, map[string]any{"name": "example.test", "server": server})
	require.Contains(t, out, ";; status: NOERROR")
	require.Contains(t, out, "example.test.\t300\tIN\tA\t192.0.2.1")

	out = mustNetCall(t, reg, NetDNSLookup, map[string]any{"name": "example.test", "type": "mx", "server": server})
	require.Contains(t, out, "IN\tMX\t10 mail.example.test.")

	out = mustNetCall(t, reg, NetDNSLookup, map[string]any{"name": "192.0.2.1", "server": server})
	require.Contains(t, out, "1.2.0.192.in-addr.arpa.\t300\tIN\tPTR\texample.test.")

	out = mustNetCall(t, reg, NetDNSLookup, map[string]any{"name": "big.test", "type": "TXT", "server": server})
	require.Contains(t, out, "(tcp)")
	require.Contains(t, out, `"over tcp"`)

	out = mustNetCall(t, reg, NetDNSLookup, map[string]any{"name": "missing.test", "server": server, "tcp": true})
	require.Contains(t, out, "status: NXDOMAIN")
	require.Contains(t, out, "no answer records")

	_, err := netCall(t, reg, NetDNSLookup, map[string]any{"name": "example.test", "type": "BOGUS"})
	require.ErrorContains(t, err, "unknown record type")

	_, reg = newTestNetwork(t, WithNetworkPrivate(false))
	_, err = netCall(t, reg, NetDNSLookup, map[string]any{"name": "example.test", "server": server})
	require.ErrorContains(t, err, "blocked", "a server the model names is checked")
}

func TestReverseName(t *testing.T) {
	require.Equal(t, "4.3.2.1.in-addr.arpa.", reverseName(netip.MustParseAddr("1.2.3.4")))
	require.Equal(t, "1.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.8.b.d.0.1.0.0.2.ip6.arpa.", reverseName(netip.MustParseAddr("2001:db8::1")))
}

// startWhois serves one canned answer and records the query.
func startWhois(t *testing.T, answer string) (string, <-chan string) {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	t.Cleanup(func() { l.Close() })
	queries := make(chan string, 4)
	go func() {
		for {
			conn, err := l.Accept()
			if err != nil {
				return
			}
			line, _ := bufio.NewReader(conn).ReadString('\n')
			queries <- strings.TrimSpace(line)
			io.WriteString(conn, answer)
			conn.Close()
		}
	}()
	return l.Addr().String(), queries
}

func TestNetworkWhois(t *testing.T) {
	registrar, registrarQueries := startWhois(t, "Domain Name: EXAMPLE.TEST\r\nRegistrar: Test\r\n")
	registry, _ := startWhois(t, "domain: TEST\nrefer: "+registrar+"\n")
	_, reg := newTestNetwork(t)

	out := mustNetCall(t, reg, NetWhois, map[string]any{"query": "example.test", "server": registry})
	require.Equal(t, fmt.Sprintf("%% %s -> %s\n\nDomain Name: EXAMPLE.TEST\nRegistrar: Test", registry, registrar), out)
	require.Equal(t, "example.test", <-registrarQueries)

	_, err := netCall(t, reg, NetWhois, map[string]any{"query": "a\r\nb", "server": registry})
	require.ErrorContains(t, err, "one line")

	require.Equal(t, "whois.verisign-grs.com", whoisReferral("x\nwhois:        whois.verisign-grs.com\n"))
	require.Equal(t, "rwhois.example.net:4321", whoisReferral("ReferralServer:  whois://rwhois.example.net:4321\n"))
	require.Equal(t, "", whoisReferral("ReferralServer:  rwhois://rwhois.example.net:4321\n"))
}

func TestNetworkReadWaitsForContext(t *testing.T) {
	_, reg := newTestNetwork(t)
	id, _ := openHandle(t, reg, NetListen, map[string]any{"protocol": "udp", "address": ":0"})
	tools, err := reg.selected([]string{NetRead})
	require.NoError(t, err)
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
	defer cancel()
	_, _, err = invokeTool(ctx, tools[0], json.RawMessage(fmt.Sprintf(`{"id":%q,"timeout_ms":5000}`, id)))
	require.ErrorIs(t, err, context.DeadlineExceeded)
}
