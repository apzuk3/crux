// Command skills opens a terminal chat with an agent that uses the skills in
// examples/skills/skills and can write new ones. Ask it for a commit message
// to see a skill loaded, or teach it a way of working and ask it to save that
// as a skill; saving asks for approval in the chat.
//
//	go run ./examples/skills
package main

import (
	"log"

	"github.com/apzuk3/crux"
)

func main() {
	if err := crux.AddSkills("examples/skills/skills"); err != nil {
		log.Fatal(err)
	}

	agent := crux.Must(crux.New("assistant", crux.OpenAIGPT5_4,
		crux.WithInstructions("You are a helpful assistant for a software team."),
		crux.WithSkills(),
	))

	if err := crux.CLI(agent); err != nil {
		log.Fatal(err)
	}
}
