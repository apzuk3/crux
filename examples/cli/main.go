// Command cli opens a terminal chat with a coding agent that can read and
// edit the files in a directory, with a reviewer subagent it can ask for a
// second opinion. Tools that change files ask for approval in the chat.
//
//	go run ./examples/cli [dir]
package main

import (
	"log"
	"os"

	"crux.foo"
)

func main() {
	dir := "."
	if len(os.Args) > 1 {
		dir = os.Args[1]
	}
	if err := crux.AddToolset(crux.Filesystem(dir)); err != nil {
		log.Fatal(err)
	}

	reviewer := crux.Must(crux.New("reviewer", crux.OpenAIGPT5_4,
		crux.WithInstructions("You review code in the project for bugs and unclear naming. Answer with a short list."),
		crux.WithTools([]string{"read_file", "glob", "search_files_content"}),
	))
	coder := crux.Must(crux.New("coder", crux.OpenAIGPT5_4,
		crux.WithInstructions("You help with the files in the current project. Paths are relative to the project root."),
		crux.WithToolsets("filesystem"),
		crux.WithSubAgent(reviewer, "Reviews files for bugs and unclear naming"),
		crux.WithMaxTurns(30),
	))

	if err := crux.CLI(coder); err != nil {
		log.Fatal(err)
	}
}
