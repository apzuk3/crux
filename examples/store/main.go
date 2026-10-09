package main

import (
	"context"
	"flag"
	"fmt"
	"log"

	"crux.foo"
	"github.com/glebarez/sqlite"
	"github.com/google/uuid"
	"gorm.io/gorm"
)

// Run it twice to see the conversation survive a restart:
//
//	go run ./examples/store
//	go run ./examples/store -session <id printed by the first run>
func main() {
	sessionFlag := flag.String("session", "", "ID of a session to continue")
	flag.Parse()
	ctx := context.Background()

	// Any GORM driver works; this pure-Go one keeps sessions in a local SQLite file.
	db, err := gorm.Open(sqlite.Open("crux-example.db"), &gorm.Config{})
	if err != nil {
		log.Fatalf("open database: %v", err)
	}
	store, err := crux.NewGORMStore(db)
	if err != nil {
		log.Fatalf("create store: %v", err)
	}

	agent := crux.Must(crux.New(
		"store-agent",
		crux.OpenAIGPT5_4,
		crux.WithInstructions("You are a concise assistant. Remember what the user tells you."),
	))

	opts := []crux.SessionOption{crux.WithStore(store)}
	prompt := "My favourite colour is teal. Please remember it."
	if *sessionFlag != "" {
		id, err := uuid.Parse(*sessionFlag)
		if err != nil {
			log.Fatalf("invalid session ID: %v", err)
		}
		// An ID the store already knows continues that conversation.
		opts = append(opts, crux.WithSessionID(id))
		prompt = "What is my favourite colour?"
	}

	session, err := crux.NewSession(ctx, agent, opts...)
	if err != nil {
		log.Fatalf("create session: %v", err)
	}

	answer, err := session.Run(ctx, prompt)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(answer)
	fmt.Printf("\nsession %s (%d entries)\n", session.ID(), len(session.Logs()))
}
