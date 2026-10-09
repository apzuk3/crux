// Command domains opens a terminal chat that proposes domain names for an
// idea and checks that they are still available to register, using the
// Network toolset: http_get asks the registry's RDAP server and whois covers
// top-level domains without one. Both run without approval, so the agent
// works on its own. Describe your idea in the chat, then ask for more names,
// other styles or other TLDs.
//
//	go run ./examples/domains
//
// "Available" means not registered; premium or reserved names may still cost
// more or be refused at checkout.
package main

import (
	"log"

	"crux.foo"
)

// RDAP servers of the registries, from IANA's bootstrap file
// (https://data.iana.org/rdap/dns.json). .io and .co have none.
const instructions = `You propose domain names for the user's idea.

Brainstorm short, memorable, easy to spell names: one or two words, no hyphens or digits, across .com, .ai, .dev, .app, .io and .co unless the user asks for others.
Check every name before proposing it, and keep brainstorming new names until at least 5 are available. Check many names at once by calling tools in parallel.

How to check a name:
- .com, .ai, .dev, .app: call http_get with method HEAD on the registry's RDAP URL. 404 Not Found means available, 200 OK means taken; anything else is unknown, so skip the name.
    .com  https://rdap.verisign.com/com/v1/domain/NAME.com
    .ai   https://rdap.identitydigital.services/rdap/domain/NAME.ai
    .dev  https://pubapi.registry.google/rdap/domain/NAME.dev
    .app  https://pubapi.registry.google/rdap/domain/NAME.app
- .io, .co and other top-level domains: call whois with the domain. "Domain not found", "No match" or "NOT FOUND" means available; a "Domain Name:" record means taken.

Only propose names a check showed as available. Answer with a short list: each available domain in bold, with one sentence on why it fits the idea.`

func main() {
	network := crux.Network()
	defer network.Close()
	if err := crux.AddToolset(network); err != nil {
		log.Fatal(err)
	}

	agent := crux.Must(crux.New("namer", crux.Gemini3_8Flash,
		crux.WithInstructions(instructions),
		crux.WithTools([]string{crux.NetHTTPGet, crux.NetWhois}),
		crux.WithMaxTurns(15),
	))
	if err := crux.CLI(agent); err != nil {
		log.Fatal(err)
	}
}
