//go:build windows

package crux

import (
	"net/netip"
	"unsafe"

	"golang.org/x/sys/windows"
)

// systemDNSServers returns the DNS servers of the network adapters that are
// up, as Windows reports them.
func systemDNSServers() []string {
	size := uint32(15000)
	for range 3 {
		buf := make([]byte, size)
		first := (*windows.IpAdapterAddresses)(unsafe.Pointer(&buf[0]))
		err := windows.GetAdaptersAddresses(windows.AF_UNSPEC, windows.GAA_FLAG_SKIP_ANYCAST|windows.GAA_FLAG_SKIP_MULTICAST, 0, first, &size)
		if err == windows.ERROR_BUFFER_OVERFLOW {
			continue
		}
		if err != nil {
			return nil
		}
		var servers []string
		seen := make(map[string]bool)
		for adapter := first; adapter != nil; adapter = adapter.Next {
			if adapter.OperStatus != windows.IfOperStatusUp {
				continue
			}
			for dns := adapter.FirstDnsServerAddress; dns != nil; dns = dns.Next {
				addr, ok := netip.AddrFromSlice(dns.Address.IP())
				if !ok || deprecatedSiteLocalDNS(addr) {
					continue
				}
				if s := addr.Unmap().String(); !seen[s] {
					seen[s] = true
					servers = append(servers, s)
				}
			}
		}
		return servers
	}
	return nil
}

// deprecatedSiteLocalDNS reports the fec0:0:0:ffff::1-3 addresses Windows
// lists by default on adapters without IPv6 DNS servers.
func deprecatedSiteLocalDNS(addr netip.Addr) bool {
	b := addr.As16()
	return addr.Is6() && b[0] == 0xfe && b[1] == 0xc0 && b[6] == 0xff && b[7] == 0xff
}
