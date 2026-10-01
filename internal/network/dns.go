package network

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"net/netip"
	"strconv"
	"strings"
	"time"

	"golang.org/x/net/dns/dnsmessage"
)

const dnsTimeout = 5 * time.Second

// dnsTypeCAA is not defined by dnsmessage; its records arrive as
// UnknownResource.
const dnsTypeCAA dnsmessage.Type = 257

var dnsTypes = map[string]dnsmessage.Type{
	"A": dnsmessage.TypeA, "AAAA": dnsmessage.TypeAAAA, "CNAME": dnsmessage.TypeCNAME,
	"MX": dnsmessage.TypeMX, "NS": dnsmessage.TypeNS, "TXT": dnsmessage.TypeTXT,
	"SOA": dnsmessage.TypeSOA, "SRV": dnsmessage.TypeSRV, "PTR": dnsmessage.TypePTR,
	"CAA": dnsTypeCAA, "HTTPS": dnsmessage.TypeHTTPS, "SVCB": dnsmessage.TypeSVCB,
	"ANY": dnsmessage.TypeALL,
}

var dnsTypeNames = func() map[dnsmessage.Type]string {
	names := make(map[dnsmessage.Type]string, len(dnsTypes))
	for name, t := range dnsTypes {
		names[t] = name
	}
	return names
}()

// DNSLookupInput holds the arguments of DNSLookup.
type DNSLookupInput struct {
	Name   string `json:"name" description:"Domain name to look up, or an IP address for a PTR (reverse) lookup"`
	Type   string `json:"type,omitempty" description:"Record type: A (default), AAAA, CNAME, MX, NS, TXT, SOA, SRV, PTR, CAA, HTTPS, SVCB or ANY"`
	Server string `json:"server,omitempty" description:"DNS server to ask, such as 1.1.1.1 or ns1.example.com:53; default the system's resolver"`
	TCP    bool   `json:"tcp,omitempty" description:"Query over TCP instead of UDP"`
}

// DNSLookup looks up DNS records.
func (t *Tools) DNSLookup(ctx context.Context, in DNSLookupInput) (string, error) {
	typeName := strings.ToUpper(strings.TrimSpace(in.Type))
	if typeName == "" {
		typeName = "A"
	}
	qtype, ok := dnsTypes[typeName]
	if !ok {
		return "", fmt.Errorf("unknown record type %q", in.Type)
	}
	name := strings.TrimSpace(in.Name)
	if ip, err := netip.ParseAddr(name); err == nil {
		name = reverseName(ip)
		if in.Type == "" {
			qtype, typeName = dnsmessage.TypePTR, "PTR"
		}
	}
	if name == "" {
		return "", errors.New("name must not be empty")
	}
	if !strings.HasSuffix(name, ".") {
		name += "."
	}
	qname, err := dnsmessage.NewName(name)
	if err != nil {
		return "", fmt.Errorf("name %q: %w", in.Name, err)
	}

	checked := in.Server != ""
	servers := []string{in.Server}
	if !checked {
		servers = systemDNSServers()
		if len(servers) == 0 {
			return resolverLookup(ctx, strings.TrimSuffix(name, "."), typeName)
		}
	}
	var lastErr error
	for _, server := range servers {
		if _, _, err := net.SplitHostPort(server); err != nil {
			server = net.JoinHostPort(strings.Trim(server, "[]"), "53")
		}
		out, err := t.dnsExchange(ctx, server, qname, qtype, in.TCP, checked)
		if err == nil {
			return out, nil
		}
		lastErr = err
		if checked || ctx.Err() != nil {
			break
		}
	}
	return "", lastErr
}

// dnsExchange sends one query to server and formats the answer, retrying
// over TCP when the UDP answer is truncated. The system's resolver is not
// subject to the address policy; a server the model names is.
func (t *Tools) dnsExchange(ctx context.Context, server string, name dnsmessage.Name, qtype dnsmessage.Type, useTCP, checked bool) (string, error) {
	var id [2]byte
	rand.Read(id[:])
	query := dnsmessage.Message{
		Header:    dnsmessage.Header{ID: binary.BigEndian.Uint16(id[:]), RecursionDesired: true},
		Questions: []dnsmessage.Question{{Name: name, Type: qtype, Class: dnsmessage.ClassINET}},
	}
	var opt dnsmessage.ResourceHeader
	if err := opt.SetEDNS0(1232, dnsmessage.RCodeSuccess, false); err != nil {
		return "", err
	}
	query.Additionals = []dnsmessage.Resource{{Header: opt, Body: &dnsmessage.OPTResource{}}}
	packed, err := query.Pack()
	if err != nil {
		return "", fmt.Errorf("build query: %w", err)
	}

	ctx, cancel := context.WithTimeout(ctx, dnsTimeout)
	defer cancel()
	start := time.Now()
	protocol := "udp"
	if useTCP {
		protocol = "tcp"
	}
	resp, err := t.dnsSend(ctx, protocol, server, packed, query.ID, checked)
	if err == nil && resp.Truncated && !useTCP {
		protocol = "tcp"
		resp, err = t.dnsSend(ctx, protocol, server, packed, query.ID, checked)
	}
	if err != nil {
		return "", fmt.Errorf("query %s: %w", server, err)
	}
	return formatDNS(resp, server, protocol, time.Since(start)), nil
}

func (t *Tools) dnsSend(ctx context.Context, protocol, server string, query []byte, id uint16, checked bool) (*dnsmessage.Message, error) {
	var conn net.Conn
	var err error
	if checked {
		conn, err = t.dial(ctx, protocol, server, false)
	} else {
		conn, err = (&net.Dialer{}).DialContext(ctx, protocol, server)
	}
	if err != nil {
		return nil, err
	}
	defer conn.Close()
	if deadline, ok := ctx.Deadline(); ok {
		conn.SetDeadline(deadline)
	}

	var raw []byte
	if protocol == "tcp" {
		framed := binary.BigEndian.AppendUint16(nil, uint16(len(query)))
		if _, err := conn.Write(append(framed, query...)); err != nil {
			return nil, err
		}
		var size [2]byte
		if _, err := io.ReadFull(conn, size[:]); err != nil {
			return nil, err
		}
		raw = make([]byte, binary.BigEndian.Uint16(size[:]))
		if _, err := io.ReadFull(conn, raw); err != nil {
			return nil, err
		}
	} else {
		if _, err := conn.Write(query); err != nil {
			return nil, err
		}
		buf := make([]byte, 65535)
		for {
			n, err := conn.Read(buf)
			if err != nil {
				return nil, err
			}
			if n >= 2 && binary.BigEndian.Uint16(buf[:2]) == id {
				raw = buf[:n]
				break
			}
		}
	}
	var msg dnsmessage.Message
	if err := msg.Unpack(raw); err != nil {
		return nil, fmt.Errorf("parse answer: %w", err)
	}
	if msg.ID != id {
		return nil, errors.New("answer does not match the query")
	}
	return &msg, nil
}

func reverseName(ip netip.Addr) string {
	ip = ip.Unmap()
	if ip.Is4() {
		b := ip.As4()
		return fmt.Sprintf("%d.%d.%d.%d.in-addr.arpa.", b[3], b[2], b[1], b[0])
	}
	b := ip.As16()
	digits := hex.EncodeToString(b[:])
	var out strings.Builder
	for i := len(digits) - 1; i >= 0; i-- {
		out.WriteByte(digits[i])
		out.WriteByte('.')
	}
	return out.String() + "ip6.arpa."
}

func formatDNS(msg *dnsmessage.Message, server, protocol string, took time.Duration) string {
	var out strings.Builder
	flags := []string{"qr"}
	for _, f := range []struct {
		on   bool
		name string
	}{{msg.Authoritative, "aa"}, {msg.Truncated, "tc"}, {msg.RecursionDesired, "rd"}, {msg.RecursionAvailable, "ra"}, {msg.AuthenticData, "ad"}} {
		if f.on {
			flags = append(flags, f.name)
		}
	}
	rcode, ok := dnsRCodes[msg.RCode]
	if !ok {
		rcode = fmt.Sprintf("RCODE%d", msg.RCode)
	}
	fmt.Fprintf(&out, ";; status: %s, flags: %s; server %s (%s), %s\n", rcode, strings.Join(flags, " "), server, protocol, took.Round(time.Millisecond))
	for _, q := range msg.Questions {
		fmt.Fprintf(&out, ";; question: %s %s\n", q.Name, dnsTypeName(q.Type))
	}
	section := func(title string, records []dnsmessage.Resource) {
		var lines []string
		for _, r := range records {
			if r.Header.Type == dnsmessage.TypeOPT {
				continue
			}
			lines = append(lines, fmt.Sprintf("%s\t%d\tIN\t%s\t%s", r.Header.Name, r.Header.TTL, dnsTypeName(r.Header.Type), dnsRecordData(r)))
		}
		if len(lines) > 0 {
			fmt.Fprintf(&out, "\n;; %s:\n%s\n", title, strings.Join(lines, "\n"))
		}
	}
	section("answer", msg.Answers)
	section("authority", msg.Authorities)
	section("additional", msg.Additionals)
	if len(msg.Answers) == 0 {
		out.WriteString("\n;; no answer records\n")
	}
	return strings.TrimRight(out.String(), "\n")
}

var dnsRCodes = map[dnsmessage.RCode]string{
	dnsmessage.RCodeSuccess:        "NOERROR",
	dnsmessage.RCodeFormatError:    "FORMERR",
	dnsmessage.RCodeServerFailure:  "SERVFAIL",
	dnsmessage.RCodeNameError:      "NXDOMAIN",
	dnsmessage.RCodeNotImplemented: "NOTIMP",
	dnsmessage.RCodeRefused:        "REFUSED",
}

func dnsTypeName(t dnsmessage.Type) string {
	if name, ok := dnsTypeNames[t]; ok && t != dnsmessage.TypeALL {
		return name
	}
	return strings.TrimPrefix(t.String(), "Type")
}

func dnsRecordData(r dnsmessage.Resource) string {
	switch body := r.Body.(type) {
	case *dnsmessage.AResource:
		return netip.AddrFrom4(body.A).String()
	case *dnsmessage.AAAAResource:
		return netip.AddrFrom16(body.AAAA).String()
	case *dnsmessage.CNAMEResource:
		return body.CNAME.String()
	case *dnsmessage.NSResource:
		return body.NS.String()
	case *dnsmessage.PTRResource:
		return body.PTR.String()
	case *dnsmessage.MXResource:
		return fmt.Sprintf("%d %s", body.Pref, body.MX)
	case *dnsmessage.SOAResource:
		return fmt.Sprintf("%s %s %d %d %d %d %d", body.NS, body.MBox, body.Serial, body.Refresh, body.Retry, body.Expire, body.MinTTL)
	case *dnsmessage.SRVResource:
		return fmt.Sprintf("%d %d %d %s", body.Priority, body.Weight, body.Port, body.Target)
	case *dnsmessage.TXTResource:
		quoted := make([]string, len(body.TXT))
		for i, s := range body.TXT {
			quoted[i] = strconv.Quote(s)
		}
		return strings.Join(quoted, " ")
	case *dnsmessage.HTTPSResource:
		return formatSVCB(body.SVCBResource)
	case *dnsmessage.SVCBResource:
		return formatSVCB(*body)
	case *dnsmessage.UnknownResource:
		if r.Header.Type == dnsTypeCAA {
			if caa, ok := formatCAA(body.Data); ok {
				return caa
			}
		}
		return fmt.Sprintf("\\# %d %s", len(body.Data), hex.EncodeToString(body.Data))
	default:
		return fmt.Sprint(body)
	}
}

func formatSVCB(r dnsmessage.SVCBResource) string {
	parts := []string{strconv.Itoa(int(r.Priority)), r.Target.String()}
	for _, param := range r.Params {
		parts = append(parts, fmt.Sprintf("%s=%s", strings.ToLower(param.Key.String()), svcbValue(param)))
	}
	return strings.Join(parts, " ")
}

func svcbValue(param dnsmessage.SVCParam) string {
	switch param.Key {
	case dnsmessage.SVCParamALPN:
		var alpn []string
		for v := param.Value; len(v) > 0 && int(v[0]) < len(v); v = v[1+int(v[0]):] {
			alpn = append(alpn, string(v[1:1+int(v[0])]))
		}
		return strings.Join(alpn, ",")
	case dnsmessage.SVCParamPort:
		if len(param.Value) == 2 {
			return strconv.Itoa(int(binary.BigEndian.Uint16(param.Value)))
		}
	case dnsmessage.SVCParamIPv4Hint, dnsmessage.SVCParamIPv6Hint:
		size := 4
		if param.Key == dnsmessage.SVCParamIPv6Hint {
			size = 16
		}
		var ips []string
		for v := param.Value; len(v) >= size; v = v[size:] {
			if addr, ok := netip.AddrFromSlice(v[:size]); ok {
				ips = append(ips, addr.String())
			}
		}
		return strings.Join(ips, ",")
	}
	return hex.EncodeToString(param.Value)
}

func formatCAA(data []byte) (string, bool) {
	if len(data) < 2 || int(data[1])+2 > len(data) {
		return "", false
	}
	tagLen := int(data[1])
	return fmt.Sprintf("%d %s %q", data[0], data[2:2+tagLen], data[2+tagLen:]), true
}

// resolverLookup answers with the OS resolver when no DNS server address is
// known. It has no TTLs and supports fewer record types.
func resolverLookup(ctx context.Context, name, typeName string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, dnsTimeout)
	defer cancel()
	r := net.DefaultResolver
	var lines []string
	var err error
	switch typeName {
	case "A", "AAAA":
		var ips []netip.Addr
		ips, err = r.LookupNetIP(ctx, map[string]string{"A": "ip4", "AAAA": "ip6"}[typeName], name)
		for _, ip := range ips {
			lines = append(lines, fmt.Sprintf("%s.\tIN\t%s\t%s", name, typeName, ip))
		}
	case "CNAME":
		var cname string
		cname, err = r.LookupCNAME(ctx, name)
		lines = append(lines, fmt.Sprintf("%s.\tIN\tCNAME\t%s", name, cname))
	case "MX":
		var mxs []*net.MX
		mxs, err = r.LookupMX(ctx, name)
		for _, mx := range mxs {
			lines = append(lines, fmt.Sprintf("%s.\tIN\tMX\t%d %s", name, mx.Pref, mx.Host))
		}
	case "NS":
		var nss []*net.NS
		nss, err = r.LookupNS(ctx, name)
		for _, ns := range nss {
			lines = append(lines, fmt.Sprintf("%s.\tIN\tNS\t%s", name, ns.Host))
		}
	case "TXT":
		var txts []string
		txts, err = r.LookupTXT(ctx, name)
		for _, txt := range txts {
			lines = append(lines, fmt.Sprintf("%s.\tIN\tTXT\t%q", name, txt))
		}
	case "SRV":
		var srvs []*net.SRV
		_, srvs, err = r.LookupSRV(ctx, "", "", name)
		for _, srv := range srvs {
			lines = append(lines, fmt.Sprintf("%s.\tIN\tSRV\t%d %d %d %s", name, srv.Priority, srv.Weight, srv.Port, srv.Target))
		}
	case "PTR":
		return "", errors.New("PTR lookups need a DNS server; pass server")
	default:
		return "", fmt.Errorf("%s lookups need a DNS server; pass server, such as 1.1.1.1", typeName)
	}
	if err != nil {
		return "", fmt.Errorf("look up %s %s: %w", name, typeName, err)
	}
	return ";; answered by the system resolver (no TTLs)\n" + strings.Join(lines, "\n"), nil
}
