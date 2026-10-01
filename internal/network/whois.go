package network

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"strings"
	"time"
)

const (
	whoisIANA     = "whois.iana.org"
	whoisMaxHops  = 3
	whoisMaxBytes = 256 << 10
	whoisMaxShown = 64 << 10
)

// WhoisInput holds the arguments of Whois.
type WhoisInput struct {
	Query  string `json:"query" description:"Domain name, IP address or AS number (such as AS15169)"`
	Server string `json:"server,omitempty" description:"Whois server to start at; default whois.iana.org"`
}

// Whois looks up whois registration data, following referrals.
func (t *Tools) Whois(ctx context.Context, in WhoisInput) (string, error) {
	query := strings.TrimSpace(in.Query)
	if query == "" || strings.ContainsAny(query, "\r\n") {
		return "", errors.New("query must be one line of text")
	}
	server := in.Server
	if server == "" {
		server = whoisIANA
	}
	ctx, cancel := context.WithTimeout(ctx, netDefaultTimeout)
	defer cancel()

	var path []string
	var answer string
	for hop := 0; hop <= whoisMaxHops; hop++ {
		address := whoisAddress(server)
		response, err := t.whoisQuery(ctx, address, query)
		if err != nil {
			if answer != "" {
				// Keep the answer we have; say why the referral failed.
				return fmt.Sprintf("%% %s\n%% referral to %s failed: %v\n\n%s", strings.Join(path, " -> "), server, err, answer), nil
			}
			return "", fmt.Errorf("whois %s: %w", server, err)
		}
		path = append(path, server)
		answer = response
		next := whoisReferral(response)
		if next == "" || strings.EqualFold(whoisAddress(next), address) {
			break
		}
		server = next
	}
	if len(answer) > whoisMaxShown {
		answer = answer[:whoisMaxShown] + fmt.Sprintf("\n(truncated to %d bytes)", whoisMaxShown)
	}
	return fmt.Sprintf("%% %s\n\n%s", strings.Join(path, " -> "), strings.TrimSpace(answer)), nil
}

func whoisAddress(server string) string {
	if _, _, err := net.SplitHostPort(server); err == nil {
		return server
	}
	return net.JoinHostPort(server, "43")
}

func (t *Tools) whoisQuery(ctx context.Context, address, query string) (string, error) {
	conn, err := t.dial(ctx, "tcp", address, false)
	if err != nil {
		return "", err
	}
	defer conn.Close()
	deadline, ok := ctx.Deadline()
	if !ok {
		deadline = time.Now().Add(netDefaultTimeout)
	}
	conn.SetDeadline(deadline)
	if _, err := io.WriteString(conn, query+"\r\n"); err != nil {
		return "", err
	}
	data, err := io.ReadAll(io.LimitReader(conn, whoisMaxBytes))
	if err != nil && len(data) == 0 {
		return "", err
	}
	return strings.ToValidUTF8(strings.ReplaceAll(string(data), "\r\n", "\n"), "?"), nil
}

// whoisReferral finds the next whois server named in a response: IANA's
// "refer:" and "whois:", registries' "Registrar WHOIS Server:" and ARIN's
// "ReferralServer: whois://host".
func whoisReferral(response string) string {
	for _, line := range strings.Split(response, "\n") {
		key, value, ok := strings.Cut(strings.TrimSpace(line), ":")
		if !ok {
			continue
		}
		value = strings.TrimSpace(value)
		switch strings.ToLower(strings.TrimSpace(key)) {
		case "refer", "whois", "registrar whois server", "referralserver":
			if value == "" {
				continue
			}
			if strings.Contains(value, "://") {
				u, err := url.Parse(value)
				if err != nil || u.Scheme != "whois" {
					continue
				}
				value = u.Host
			}
			return value
		}
	}
	return ""
}
