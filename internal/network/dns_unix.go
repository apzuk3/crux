//go:build !windows

package network

import (
	"bufio"
	"net/netip"
	"os"
	"strings"
)

// systemDNSServers returns the nameservers in /etc/resolv.conf.
func systemDNSServers() []string {
	f, err := os.Open("/etc/resolv.conf")
	if err != nil {
		return nil
	}
	defer f.Close()
	var servers []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 2 || fields[0] != "nameserver" {
			continue
		}
		// A link-local IPv6 server carries its zone, as in fe80::1%en0.
		if addr, err := netip.ParseAddr(fields[1]); err == nil {
			servers = append(servers, addr.String())
		}
	}
	return servers
}
