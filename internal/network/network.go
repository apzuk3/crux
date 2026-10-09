// Package network implements the network tools: DNS, whois, HTTP, sockets,
// listeners and servers, with every dial checked against an address policy.
// The crux package registers them as the "network" toolset.
package network

import (
	"bytes"
	"cmp"
	"context"
	"crypto/tls"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/coder/websocket"
)

const (
	MaxHandles           = 32      // open handles per toolset
	netMaxBuffer         = 1 << 20 // bytes buffered per handle
	netDefaultReadWait   = time.Second
	netMaxReadWait       = 30 * time.Second
	netDefaultReadBytes  = 16 << 10
	netMaxReadBytes      = 1 << 20
	netIdleTimeout       = 30 * time.Minute
	netMaxLoggedRequests = 100
	netMaxLoggedBody     = 4 << 10
	netDefaultTimeout    = 30 * time.Second
	netDefaultHTTPBytes  = 64 << 10
	netMaxHTTPBytes      = 1 << 20
)

// Tools holds the network tools' connections, listeners and servers. Its
// methods are the tool handlers.
type Tools struct {
	private bool
	hosts   []string

	mu         sync.Mutex
	policy     *netPolicy
	handles    map[string]*netHandle
	nextID     int
	transports map[bool]*http.Transport // by public-only
	initOnce   sync.Once
	initErr    error
}

// New returns the tools. private allows loopback, private and link-local
// addresses and unix sockets; hosts, when not empty, limits the tools to
// those names, wildcards, addresses and CIDR ranges.
func New(private bool, hosts []string) *Tools {
	return &Tools{private: private, hosts: hosts, handles: make(map[string]*netHandle)}
}

// Init checks the host patterns. It must succeed before a tool is used.
func (t *Tools) Init() error {
	t.initOnce.Do(func() {
		t.policy, t.initErr = newNetPolicy(t.private, t.hosts)
	})
	return t.initErr
}

// Close closes every connection, listener and server the toolset has open.
// The toolset can still be used afterwards.
func (t *Tools) Close() error {
	t.mu.Lock()
	handles := make([]*netHandle, 0, len(t.handles))
	for _, h := range t.handles {
		handles = append(handles, h)
	}
	t.handles = make(map[string]*netHandle)
	t.mu.Unlock()
	var errs []error
	for _, h := range handles {
		errs = append(errs, h.close())
	}
	return errors.Join(errs...)
}

// netPolicy decides which addresses the tools may dial.
type netPolicy struct {
	private bool
	names   []string // exact names and "*.suffix" wildcards, lower case
	ranges  []netip.Prefix
}

func newNetPolicy(private bool, patterns []string) (*netPolicy, error) {
	p := &netPolicy{private: private}
	for _, pattern := range patterns {
		pattern = strings.ToLower(strings.TrimSpace(pattern))
		switch {
		case pattern == "":
			return nil, errors.New("WithNetworkHosts: empty host pattern")
		case strings.Contains(pattern, "/"):
			prefix, err := netip.ParsePrefix(pattern)
			if err != nil {
				return nil, fmt.Errorf("WithNetworkHosts: %w", err)
			}
			p.ranges = append(p.ranges, prefix.Masked())
		default:
			if addr, err := netip.ParseAddr(pattern); err == nil {
				p.ranges = append(p.ranges, netip.PrefixFrom(addr.Unmap(), addr.Unmap().BitLen()))
				continue
			}
			p.names = append(p.names, strings.TrimSuffix(pattern, "."))
		}
	}
	return p, nil
}

func (p *netPolicy) limited() bool { return len(p.names) > 0 || len(p.ranges) > 0 }

func (p *netPolicy) nameAllowed(host string) bool {
	host = strings.TrimSuffix(strings.ToLower(host), ".")
	for _, name := range p.names {
		if suffix, ok := strings.CutPrefix(name, "*."); ok {
			if strings.HasSuffix(host, "."+suffix) {
				return true
			}
		} else if host == name {
			return true
		}
	}
	return false
}

// checkIP reports whether ip may be dialed for host. publicOnly refuses
// non-public addresses whatever the policy allows.
func (p *netPolicy) checkIP(host string, ip netip.Addr, publicOnly bool) error {
	ip = ip.Unmap()
	if (publicOnly || !p.private) && !publicIP(ip) {
		if publicOnly {
			return fmt.Errorf("%s resolves to %s, which is not a public address; http_get only reaches public hosts", host, ip)
		}
		return fmt.Errorf("%s resolves to %s; private and local addresses are blocked", host, ip)
	}
	if p.limited() && !p.nameAllowed(host) && !slices.ContainsFunc(p.ranges, func(r netip.Prefix) bool { return r.Contains(ip) }) {
		return fmt.Errorf("host %s (%s) is not in the allowed hosts", host, ip)
	}
	return nil
}

var cgnat = netip.MustParsePrefix("100.64.0.0/10")
var thisNetwork = netip.MustParsePrefix("0.0.0.0/8")

// publicIP reports whether ip is a global unicast address that is not
// private, loopback, link-local, CGNAT or unique-local.
func publicIP(ip netip.Addr) bool {
	return ip.IsValid() && ip.IsGlobalUnicast() && !ip.IsPrivate() && !ip.IsLoopback() &&
		!ip.IsLinkLocalUnicast() && !cgnat.Contains(ip) && !thisNetwork.Contains(ip)
}

// dialer returns a dialer that checks every address it connects to for host.
func (t *Tools) dialer(host string, publicOnly bool) *net.Dialer {
	return &net.Dialer{
		Timeout: netDefaultTimeout,
		Control: func(network, address string, _ syscall.RawConn) error {
			ipStr, _, err := net.SplitHostPort(address)
			if err != nil {
				return err
			}
			ip, err := netip.ParseAddr(ipStr)
			if err != nil {
				return err
			}
			return t.policy.checkIP(host, ip, publicOnly)
		},
	}
}

func (t *Tools) dial(ctx context.Context, network, address string, publicOnly bool) (net.Conn, error) {
	if strings.HasPrefix(network, "unix") {
		if publicOnly || !t.policy.private {
			return nil, errors.New("unix sockets are blocked with private addresses")
		}
		if t.policy.limited() {
			return nil, errors.New("unix sockets are not allowed when the hosts are limited")
		}
		return (&net.Dialer{Timeout: netDefaultTimeout}).DialContext(ctx, network, address)
	}
	host, _, err := net.SplitHostPort(address)
	if err != nil {
		return nil, err
	}
	return t.dialer(host, publicOnly).DialContext(ctx, network, address)
}

// netHandle is an open connection, listener or server. Its background reader
// fills buf; net_read drains it.
type netHandle struct {
	id      string
	kind    string // protocol, "listen-tcp" and so on, or "http"
	local   string
	remote  string
	created time.Time
	buf     *netBuffer

	send  func(ctx context.Context, data []byte, in SendInput) error
	close func() error

	mu       sync.Mutex
	lastUsed time.Time
}

func (h *netHandle) touch() {
	h.mu.Lock()
	h.lastUsed = time.Now()
	h.mu.Unlock()
}

func (h *netHandle) idle() time.Duration {
	h.mu.Lock()
	defer h.mu.Unlock()
	return time.Since(h.lastUsed)
}

// add registers a handle under a new ID with the given prefix. It fails
// when too many handles are open.
func (t *Tools) add(prefix string, h *netHandle) error {
	t.sweep()
	t.mu.Lock()
	defer t.mu.Unlock()
	if len(t.handles) >= MaxHandles {
		return fmt.Errorf("too many open handles (%d); close some with net_close", MaxHandles)
	}
	t.nextID++
	h.id = fmt.Sprintf("%s-%d", prefix, t.nextID)
	h.created = time.Now()
	h.lastUsed = h.created
	t.handles[h.id] = h
	return nil
}

func (t *Tools) handle(id string) (*netHandle, error) {
	t.sweep()
	t.mu.Lock()
	defer t.mu.Unlock()
	h, ok := t.handles[id]
	if !ok {
		return nil, fmt.Errorf("no open handle %q; net_list shows the open ones", id)
	}
	h.touch()
	return h, nil
}

func (t *Tools) remove(id string) *netHandle {
	t.mu.Lock()
	defer t.mu.Unlock()
	h := t.handles[id]
	delete(t.handles, id)
	return h
}

// sweep closes handles that have not been used for netIdleTimeout.
func (t *Tools) sweep() {
	t.mu.Lock()
	var idle []*netHandle
	for id, h := range t.handles {
		if h.idle() > netIdleTimeout {
			idle = append(idle, h)
			delete(t.handles, id)
		}
	}
	t.mu.Unlock()
	for _, h := range idle {
		h.close()
	}
}

// netMessage is one datagram, WebSocket message, accepted connection or
// logged request.
type netMessage struct {
	label string // shown in brackets before the data
	data  []byte
	text  bool // data is text whatever its bytes look like
}

// netBuffer holds what arrived on a handle until net_read takes it. Streams
// append to data; everything else appends messages.
type netBuffer struct {
	mu      sync.Mutex
	data    []byte
	msgs    []netMessage
	size    int
	dropped int
	ended   error // why the source stopped; io.EOF for a clean close
	wake    chan struct{}
	done    chan struct{} // closed when the handle is closed
}

func newNetBuffer() *netBuffer {
	return &netBuffer{wake: make(chan struct{}), done: make(chan struct{})}
}

func (b *netBuffer) signalLocked() {
	close(b.wake)
	b.wake = make(chan struct{})
}

// write appends stream data, waiting while the buffer is full so the sender
// is slowed down rather than data lost. It returns false once the handle is
// closed.
func (b *netBuffer) write(p []byte) bool {
	b.mu.Lock()
	for b.size >= netMaxBuffer {
		wake := b.wake
		b.mu.Unlock()
		select {
		case <-wake:
		case <-b.done:
			return false
		}
		b.mu.Lock()
	}
	b.data = append(b.data, p...)
	b.size += len(p)
	b.signalLocked()
	b.mu.Unlock()
	return true
}

// pushRequest appends a logged request, keeping at most
// netMaxLoggedRequests unread.
func (b *netBuffer) pushRequest(m netMessage) {
	b.mu.Lock()
	full := len(b.msgs) >= netMaxLoggedRequests
	if full {
		b.dropped++
	}
	b.mu.Unlock()
	if !full {
		b.push(m)
	}
}

// push appends a message, dropping it when the buffer is full.
func (b *netBuffer) push(m netMessage) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.size+len(m.data) > netMaxBuffer {
		b.dropped++
		return
	}
	b.msgs = append(b.msgs, m)
	b.size += len(m.data)
	b.signalLocked()
}

func (b *netBuffer) end(err error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.ended == nil {
		b.ended = err
		b.signalLocked()
	}
}

func (b *netBuffer) buffered() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.size
}

// ReadInput holds the arguments of Read.
type ReadInput struct {
	ID        string `json:"id" description:"Handle id from net_connect, net_listen, http_serve or net_list"`
	TimeoutMS *int   `json:"timeout_ms,omitempty" description:"How long to wait for data, in milliseconds; default 1000, at most 30000"`
	MaxBytes  *int   `json:"max_bytes,omitempty" description:"Most bytes to return; default 16384. The rest stays buffered for the next read"`
	Until     string `json:"until,omitempty" description:"For connections: wait until this text arrives (for example \\n or \\r\\n\\r\\n) and return data up to and including it"`
	Encoding  string `json:"encoding,omitempty" description:"How to show the data: auto (text when printable, otherwise base64), text, hex or base64. Default auto"`
}

// Read reads what has arrived on a handle.
func (t *Tools) Read(ctx context.Context, in ReadInput) (string, error) {
	h, err := t.handle(in.ID)
	if err != nil {
		return "", err
	}
	wait := netDefaultReadWait
	if in.TimeoutMS != nil {
		if *in.TimeoutMS < 0 {
			return "", errors.New("timeout_ms cannot be negative")
		}
		wait = min(time.Duration(*in.TimeoutMS)*time.Millisecond, netMaxReadWait)
	}
	limit := netDefaultReadBytes
	if in.MaxBytes != nil {
		if *in.MaxBytes < 1 {
			return "", errors.New("max_bytes must be at least 1")
		}
		limit = min(*in.MaxBytes, netMaxReadBytes)
	}
	encoding, err := netEncoding(in.Encoding, true)
	if err != nil {
		return "", err
	}

	b := h.buf
	timer := time.NewTimer(wait)
	defer timer.Stop()
	ready := func() bool {
		return len(b.msgs) > 0 || b.ended != nil ||
			(len(b.data) > 0 && (in.Until == "" || bytes.Contains(b.data, []byte(in.Until)) || len(b.data) >= limit))
	}
	b.mu.Lock()
	for timedOut := false; !timedOut && !ready(); {
		wake := b.wake
		b.mu.Unlock()
		select {
		case <-wake:
		case <-timer.C:
			timedOut = true
		case <-ctx.Done():
			return "", ctx.Err()
		}
		b.mu.Lock()
	}
	var out strings.Builder
	switch {
	case len(b.msgs) > 0:
		n := 0
		for n < len(b.msgs) {
			m := b.msgs[n]
			if n > 0 && out.Len()+len(m.data) > limit {
				break
			}
			fmt.Fprintf(&out, "[%s] %s\n", m.label, renderNetData(m.data, encoding, m.text, limit))
			b.size -= len(m.data)
			n++
		}
		b.msgs = b.msgs[n:]
	case len(b.data) > 0:
		n := min(len(b.data), limit)
		if in.Until != "" {
			if i := bytes.Index(b.data, []byte(in.Until)); i >= 0 && i+len(in.Until) <= limit {
				n = i + len(in.Until)
			}
		}
		out.WriteString(renderNetData(b.data[:n], encoding, false, limit))
		b.data = b.data[n:]
		b.size -= n
		if len(b.data) == 0 {
			b.data = nil
		}
		b.signalLocked() // room for the reader again
	}
	got := out.Len() > 0
	if !got {
		fmt.Fprintf(&out, "(nothing received within %s)", wait)
	}
	if b.dropped > 0 {
		fmt.Fprintf(&out, "\n(%d messages were dropped because the buffer was full)", b.dropped)
		b.dropped = 0
	}
	if b.size > 0 {
		fmt.Fprintf(&out, "\n(%d more bytes buffered; read again)", b.size)
	} else if b.ended != nil {
		fmt.Fprintf(&out, "\n(%s)", endedText(b.ended))
	}
	b.mu.Unlock()
	return strings.TrimRight(out.String(), "\n"), nil
}

func endedText(err error) string {
	if errors.Is(err, errHandleClosed) || errors.Is(err, net.ErrClosed) {
		return "closed"
	}
	if errors.Is(err, io.EOF) {
		return "closed by the peer"
	}
	return "ended: " + err.Error()
}

var errHandleClosed = errors.New("closed")

func netEncoding(encoding string, auto bool) (string, error) {
	switch encoding {
	case "":
		if auto {
			return "auto", nil
		}
		return "text", nil
	case "text", "hex", "base64":
		return encoding, nil
	case "auto":
		if auto {
			return encoding, nil
		}
	}
	return "", fmt.Errorf("unknown encoding %q", encoding)
}

// renderNetData shows data in the encoding; auto shows printable UTF-8 as
// text and anything else as base64.
func renderNetData(data []byte, encoding string, text bool, limit int) string {
	if len(data) > limit {
		data = data[:limit]
	}
	switch {
	case encoding == "hex":
		return "hex:" + hex.EncodeToString(data)
	case encoding == "base64":
		return "base64:" + base64.StdEncoding.EncodeToString(data)
	case encoding == "text" || text || printable(data):
		return string(data)
	default:
		return "base64:" + base64.StdEncoding.EncodeToString(data)
	}
}

func printable(data []byte) bool {
	if !utf8.Valid(data) {
		return false
	}
	for _, r := range string(data) {
		if unicode.IsControl(r) && r != '\n' && r != '\r' && r != '\t' {
			return false
		}
	}
	return true
}

// SendInput holds the arguments of Send.
type SendInput struct {
	ID       string `json:"id" description:"Handle id of an open connection"`
	Data     string `json:"data" description:"Data to send"`
	Encoding string `json:"encoding,omitempty" description:"How data is written: text (default), hex or base64, for binary data"`
	To       string `json:"to,omitempty" description:"For a UDP listener: the address to send the datagram to, as host:port"`
	Binary   bool   `json:"binary,omitempty" description:"For a WebSocket: send a binary message instead of a text one"`
}

// Send sends data on a connection handle.
func (t *Tools) Send(ctx context.Context, in SendInput) (string, error) {
	h, err := t.handle(in.ID)
	if err != nil {
		return "", err
	}
	if h.send == nil {
		return "", fmt.Errorf("cannot send on %s, a %s handle", h.id, h.kind)
	}
	encoding, err := netEncoding(in.Encoding, false)
	if err != nil {
		return "", err
	}
	var data []byte
	switch encoding {
	case "hex":
		data, err = hex.DecodeString(strings.Join(strings.Fields(in.Data), ""))
	case "base64":
		data, err = base64.StdEncoding.DecodeString(in.Data)
	default:
		data = []byte(in.Data)
	}
	if err != nil {
		return "", fmt.Errorf("decode %s data: %w", encoding, err)
	}
	if err := h.send(ctx, data, in); err != nil {
		return "", fmt.Errorf("send on %s: %w", h.id, err)
	}
	return fmt.Sprintf("sent %d bytes on %s", len(data), h.id), nil
}

// CloseInput holds the arguments of CloseHandle.
type CloseInput struct {
	ID string `json:"id" description:"Handle id to close"`
}

// CloseHandle closes a handle.
func (t *Tools) CloseHandle(ctx context.Context, in CloseInput) (string, error) {
	h := t.remove(in.ID)
	if h == nil {
		return "", fmt.Errorf("no open handle %q; net_list shows the open ones", in.ID)
	}
	if err := h.close(); err != nil && !errors.Is(err, net.ErrClosed) {
		return "", fmt.Errorf("close %s: %w", h.id, err)
	}
	return "closed " + h.id, nil
}

// ListInput holds the arguments of List.
type ListInput struct{}

// List lists the open handles.
func (t *Tools) List(ctx context.Context, _ ListInput) (string, error) {
	t.sweep()
	t.mu.Lock()
	handles := make([]*netHandle, 0, len(t.handles))
	for _, h := range t.handles {
		handles = append(handles, h)
	}
	t.mu.Unlock()
	if len(handles) == 0 {
		return "no open handles", nil
	}
	slices.SortFunc(handles, func(a, b *netHandle) int { return a.created.Compare(b.created) })
	var out strings.Builder
	for _, h := range handles {
		fmt.Fprintf(&out, "%s  %s  %s", h.id, h.kind, h.local)
		if h.remote != "" {
			fmt.Fprintf(&out, " -> %s", h.remote)
		}
		state := "open"
		h.buf.mu.Lock()
		if h.buf.ended != nil {
			state = endedText(h.buf.ended)
		}
		h.buf.mu.Unlock()
		fmt.Fprintf(&out, "  %s  buffered %d bytes  idle %s\n", state, h.buf.buffered(), h.idle().Round(time.Second))
	}
	return strings.TrimRight(out.String(), "\n"), nil
}

// ConnectInput holds the arguments of Connect.
type ConnectInput struct {
	Protocol           string            `json:"protocol" description:"tcp, udp, tls, unix, unixgram, ws or wss"`
	Address            string            `json:"address" description:"host:port, a socket path for unix, or a URL for ws and wss (ws://host/path)"`
	TimeoutMS          *int              `json:"timeout_ms,omitempty" description:"Connection timeout in milliseconds; default 30000"`
	ServerName         string            `json:"server_name,omitempty" description:"For tls: the name to verify the certificate against, if not the host"`
	InsecureSkipVerify bool              `json:"insecure_skip_verify,omitempty" description:"For tls and wss: accept any certificate"`
	Headers            map[string]string `json:"headers,omitempty" description:"For ws and wss: headers for the handshake request"`
}

// Connect opens a connection and returns its handle.
func (t *Tools) Connect(ctx context.Context, in ConnectInput) (string, error) {
	timeout := netDefaultTimeout
	if in.TimeoutMS != nil {
		if *in.TimeoutMS < 1 {
			return "", errors.New("timeout_ms must be at least 1")
		}
		timeout = time.Duration(*in.TimeoutMS) * time.Millisecond
	}
	dialCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	switch in.Protocol {
	case "tcp", "udp", "unix", "unixgram":
		conn, err := t.dial(dialCtx, in.Protocol, in.Address, false)
		if err != nil {
			return "", fmt.Errorf("connect to %s: %w", in.Address, err)
		}
		h := t.connHandle(in.Protocol, conn, in.Protocol == "udp" || in.Protocol == "unixgram")
		if err := t.add(in.Protocol, h); err != nil {
			conn.Close()
			return "", err
		}
		return fmt.Sprintf("%s: connected %s %s -> %s", h.id, in.Protocol, h.local, h.remote), nil
	case "tls":
		host, _, err := net.SplitHostPort(in.Address)
		if err != nil {
			return "", fmt.Errorf("address %q: %w", in.Address, err)
		}
		config := &tls.Config{ServerName: cmp.Or(in.ServerName, host), InsecureSkipVerify: in.InsecureSkipVerify}
		raw, err := t.dial(dialCtx, "tcp", in.Address, false)
		if err != nil {
			return "", fmt.Errorf("connect to %s: %w", in.Address, err)
		}
		conn := tls.Client(raw, config)
		if err := conn.HandshakeContext(dialCtx); err != nil {
			raw.Close()
			return "", fmt.Errorf("TLS handshake with %s: %w", in.Address, err)
		}
		h := t.connHandle("tls", conn, false)
		if err := t.add("tls", h); err != nil {
			conn.Close()
			return "", err
		}
		return fmt.Sprintf("%s: connected tls %s -> %s\n%s", h.id, h.local, h.remote, describeTLS(conn.ConnectionState())), nil
	case "ws", "wss":
		return t.connectWebSocket(dialCtx, in)
	default:
		return "", fmt.Errorf("unknown protocol %q; use tcp, udp, tls, unix, unixgram, ws or wss", in.Protocol)
	}
}

func describeTLS(state tls.ConnectionState) string {
	out := fmt.Sprintf("%s, %s", tls.VersionName(state.Version), tls.CipherSuiteName(state.CipherSuite))
	if state.NegotiatedProtocol != "" {
		out += ", ALPN " + state.NegotiatedProtocol
	}
	if len(state.PeerCertificates) > 0 {
		cert := state.PeerCertificates[0]
		out += fmt.Sprintf("\ncertificate: %s, issued by %s, valid %s to %s", cert.Subject, cert.Issuer,
			cert.NotBefore.Format(time.DateOnly), cert.NotAfter.Format(time.DateOnly))
		if len(cert.DNSNames) > 0 {
			out += ", names " + strings.Join(cert.DNSNames, " ")
		}
	}
	return out
}

// connHandle wraps a connection and starts its reader. Packet connections
// buffer each datagram as a message.
func (t *Tools) connHandle(kind string, conn net.Conn, packets bool) *netHandle {
	buf := newNetBuffer()
	h := &netHandle{kind: kind, local: conn.LocalAddr().String(), remote: conn.RemoteAddr().String(), buf: buf}
	var once sync.Once
	h.close = func() error {
		var err error
		once.Do(func() {
			close(buf.done)
			err = conn.Close()
			buf.end(errHandleClosed)
		})
		return err
	}
	h.send = func(ctx context.Context, data []byte, in SendInput) error {
		if in.To != "" {
			return errors.New("to is only for UDP listeners")
		}
		if deadline, ok := ctx.Deadline(); ok {
			conn.SetWriteDeadline(deadline)
		} else {
			conn.SetWriteDeadline(time.Now().Add(netDefaultTimeout))
		}
		_, err := conn.Write(data)
		return err
	}
	go func() {
		p := make([]byte, 64<<10)
		for {
			n, err := conn.Read(p)
			if n > 0 {
				if packets {
					buf.push(netMessage{label: "from " + h.remote, data: bytes.Clone(p[:n])})
				} else if !buf.write(p[:n]) {
					return
				}
			}
			if err != nil {
				buf.end(err)
				return
			}
		}
	}()
	return h
}

func (t *Tools) connectWebSocket(ctx context.Context, in ConnectInput) (string, error) {
	u, err := url.Parse(in.Address)
	if err != nil || (u.Scheme != "ws" && u.Scheme != "wss") || u.Host == "" {
		return "", fmt.Errorf("address %q must be a ws:// or wss:// URL", in.Address)
	}
	if u.Scheme != in.Protocol {
		return "", fmt.Errorf("address %q does not match protocol %q", in.Address, in.Protocol)
	}
	header := make(http.Header)
	for key, value := range in.Headers {
		header.Set(key, value)
	}
	transport := t.httpTransport(false)
	if in.InsecureSkipVerify {
		transport = transport.Clone()
		transport.TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
	}
	conn, resp, err := websocket.Dial(ctx, in.Address, &websocket.DialOptions{
		HTTPClient: &http.Client{Transport: transport},
		HTTPHeader: header,
	})
	if err != nil {
		if resp != nil {
			return "", fmt.Errorf("WebSocket handshake with %s: %s: %w", in.Address, resp.Status, err)
		}
		return "", fmt.Errorf("WebSocket handshake with %s: %w", in.Address, err)
	}
	conn.SetReadLimit(netMaxBuffer)

	buf := newNetBuffer()
	readCtx, cancelRead := context.WithCancel(context.WithoutCancel(ctx))
	h := &netHandle{kind: in.Protocol, local: "", remote: in.Address, buf: buf}
	var once sync.Once
	h.close = func() error {
		var err error
		once.Do(func() {
			close(buf.done)
			err = conn.Close(websocket.StatusNormalClosure, "")
			cancelRead()
			buf.end(errHandleClosed)
		})
		return err
	}
	h.send = func(ctx context.Context, data []byte, in SendInput) error {
		if in.To != "" {
			return errors.New("to is only for UDP listeners")
		}
		kind := websocket.MessageText
		if in.Binary {
			kind = websocket.MessageBinary
		}
		ctx, cancel := context.WithTimeout(ctx, netDefaultTimeout)
		defer cancel()
		return conn.Write(ctx, kind, data)
	}
	if err := t.add(in.Protocol, h); err != nil {
		h.close()
		return "", err
	}
	go func() {
		for {
			kind, data, err := conn.Read(readCtx)
			if err != nil {
				if status := websocket.CloseStatus(err); status != -1 {
					err = fmt.Errorf("WebSocket closed by the peer with status %d (%s)", int(status), status)
				}
				buf.end(err)
				return
			}
			if kind == websocket.MessageText {
				buf.push(netMessage{label: "text", data: data, text: true})
			} else {
				buf.push(netMessage{label: "binary", data: data})
			}
		}
	}()
	return fmt.Sprintf("%s: connected to %s (%s)", h.id, in.Address, resp.Status), nil
}

// ListenInput holds the arguments of Listen.
type ListenInput struct {
	Protocol string `json:"protocol" description:"tcp, udp or unix"`
	Address  string `json:"address" description:"host:port, or a socket path for unix. A port alone, such as :8080, listens on 127.0.0.1"`
}

// listenAddress puts a port without a host on 127.0.0.1, so nothing is
// exposed beyond this machine unless the model names an interface.
func listenAddress(protocol, address string) (string, error) {
	if protocol == "unix" {
		if address == "" {
			return "", errors.New("address must be a socket path")
		}
		return address, nil
	}
	if _, err := strconv.Atoi(address); err == nil {
		address = ":" + address
	}
	host, port, err := net.SplitHostPort(address)
	if err != nil {
		return "", fmt.Errorf("address %q: %w", address, err)
	}
	if host == "" {
		host = "127.0.0.1"
	}
	return net.JoinHostPort(host, port), nil
}

// Listen listens for connections or datagrams.
func (t *Tools) Listen(ctx context.Context, in ListenInput) (string, error) {
	address, err := listenAddress(in.Protocol, in.Address)
	if err != nil {
		return "", err
	}
	var lc net.ListenConfig
	switch in.Protocol {
	case "tcp", "unix":
		listener, err := lc.Listen(ctx, in.Protocol, address)
		if err != nil {
			return "", fmt.Errorf("listen on %s: %w", address, err)
		}
		h := t.listenerHandle(in.Protocol, listener)
		if err := t.add("listen", h); err != nil {
			listener.Close()
			return "", err
		}
		return fmt.Sprintf("%s: listening on %s %s; net_read shows accepted connections", h.id, in.Protocol, h.local), nil
	case "udp":
		conn, err := lc.ListenPacket(ctx, "udp", address)
		if err != nil {
			return "", fmt.Errorf("listen on %s: %w", address, err)
		}
		h := packetListenerHandle(conn)
		if err := t.add("listen", h); err != nil {
			conn.Close()
			return "", err
		}
		return fmt.Sprintf("%s: listening on udp %s; net_read shows datagrams, net_send with to replies", h.id, h.local), nil
	default:
		return "", fmt.Errorf("unknown protocol %q; use tcp, udp or unix", in.Protocol)
	}
}

// listenerHandle accepts connections in the background; each becomes a
// handle of its own and is announced as a message.
func (t *Tools) listenerHandle(protocol string, listener net.Listener) *netHandle {
	buf := newNetBuffer()
	h := &netHandle{kind: "listen-" + protocol, local: listener.Addr().String(), buf: buf}
	var once sync.Once
	h.close = func() error {
		var err error
		once.Do(func() {
			close(buf.done)
			err = listener.Close()
			buf.end(errHandleClosed)
		})
		return err
	}
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				buf.end(err)
				return
			}
			ch := t.connHandle(protocol, conn, false)
			if err := t.add(protocol, ch); err != nil {
				conn.Close()
				buf.push(netMessage{label: "refused", data: []byte(fmt.Sprintf("connection from %s: %v", ch.remote, err)), text: true})
				continue
			}
			buf.push(netMessage{label: "accepted", data: []byte(fmt.Sprintf("%s from %s", ch.id, ch.remote)), text: true})
		}
	}()
	return h
}

func packetListenerHandle(conn net.PacketConn) *netHandle {
	buf := newNetBuffer()
	h := &netHandle{kind: "listen-udp", local: conn.LocalAddr().String(), buf: buf}
	var once sync.Once
	h.close = func() error {
		var err error
		once.Do(func() {
			close(buf.done)
			err = conn.Close()
			buf.end(errHandleClosed)
		})
		return err
	}
	h.send = func(ctx context.Context, data []byte, in SendInput) error {
		if in.To == "" {
			return errors.New("a UDP listener needs to, the address to send to")
		}
		addr, err := net.ResolveUDPAddr("udp", in.To)
		if err != nil {
			return err
		}
		conn.SetWriteDeadline(time.Now().Add(netDefaultTimeout))
		_, err = conn.WriteTo(data, addr)
		return err
	}
	go func() {
		p := make([]byte, 64<<10)
		for {
			n, from, err := conn.ReadFrom(p)
			if n > 0 {
				buf.push(netMessage{label: "from " + from.String(), data: bytes.Clone(p[:n])})
			}
			if err != nil {
				buf.end(err)
				return
			}
		}
	}()
	return h
}
