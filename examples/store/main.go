package main

import (
	"context"
	"fmt"
	"log"

	"github.com/apzuk3/crux"
)

func main() {
	ctx := context.Background()

	// 1. Initialize an in-memory SQLite GORM store
	store, err := crux.NewInMemoryStore()
	if err != nil {
		log.Fatalf("failed to create store: %v", err)
	}

	// 2. Define the stateless Agent blueprint
	agent := crux.Must(crux.New(
		"store-agent",
		crux.ChatModelGPT4,
		crux.WithInstructions("You are a store assistant. You can inspect logs or wipe server disks."),
		crux.WithMaxTurns(15),
	))

	// 3. Start a new session wired to the store
	session, err := crux.NewSession(ctx, agent, crux.WithStore(store))
	if err != nil {
		log.Fatalf("failed to create session: %v", err)
	}

	fmt.Printf("Created new session with ID: %s (Agent ID: %s)\n", session.ID(), agent.ID())

	// 4. Resume the session by its ID using the same store
	resumed, err := crux.ResumeSession(ctx, session.ID(), store, agent)
	if err != nil {
		log.Fatalf("failed to resume session: %v", err)
	}

	fmt.Printf("Successfully resumed session with ID: %s (Logs count: %d)\n", resumed.ID(), len(resumed.Logs()))
}
