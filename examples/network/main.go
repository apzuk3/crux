// Command network opens a terminal chat with a network assistant: it can look
// up DNS records and whois data, fetch web pages, open TCP, UDP, TLS and
// WebSocket connections, and listen on ports or serve HTTP. Tools that send
// data or open a port ask for approval in the chat.
//
//	go run ./examples/network
package main

import (
	"log"

	"crux.foo"
)

func main() {
	network := crux.Network()
	defer network.Close()
	if err := crux.AddToolset(network); err != nil {
		log.Fatal(err)
	}

	agent := crux.Must(crux.New("netops", crux.Gemini3_8Flash,
		crux.WithInstructions("You are a network engineer. Use the network tools to answer: check DNS, whois, "+
			"HTTP endpoints, TLS certificates and ports, and explain what you find briefly. "+
			"Close connections and servers you no longer need."),
		crux.WithToolsets(crux.ToolsetNetwork),
		crux.WithMaxTurns(40),
	))
	if err := crux.CLI(agent); err != nil {
		log.Fatal(err)
	}
}
