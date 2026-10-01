package network

import (
	"net/netip"
	"testing"
	"time"

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
}
