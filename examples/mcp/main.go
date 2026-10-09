// Command mcp opens a terminal chat about your Linear issues, using Linear's
// MCP server. On the first run the browser opens to log in to Linear with
// OAuth; the token is cached, so later runs start straight away. Tools that
// change Linear ask for approval in the chat.
//
//	go run ./examples/mcp
package main

import (
	"context"
	"log"

	"crux.foo"
)

func main() {
	ctx := context.Background()

	linear, err := crux.ConfigureMCP(ctx, "linear", crux.MCPRemote("https://mcp.linear.app/mcp"))
	if err != nil {
		log.Fatal(err)
	}
	defer linear.Close()

	agent := crux.Must(crux.New("linear", crux.Gemini3_8Flash,
		crux.WithInstructions("You help the user with their Linear issues and projects. "+
			"Look issues up before answering, cite their identifiers (such as ENG-123), and keep answers short. "+
			"Confirm what you are about to change before creating or updating anything."),
		crux.WithMCPs("linear"),
		crux.WithMaxTurns(30),
	))
	if err := crux.CLI(agent); err != nil {
		log.Fatal(err)
	}
}
