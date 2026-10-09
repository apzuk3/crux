// Command filesystem lets an agent explore and edit files in a directory.
// Tools that change files ask for approval on the terminal first.
//
//	go run ./examples/filesystem [dir]
package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"

	"crux.foo"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx := context.Background()
	dir := "."
	if len(os.Args) > 1 {
		dir = os.Args[1]
	}

	session, err := newFilesAgentSession(ctx, dir)
	if err != nil {
		return err
	}

	in := bufio.NewScanner(os.Stdin)
	fmt.Print("> ")
	for in.Scan() {
		answer, err := answerWithApprovals(ctx, session, in, in.Text())
		if err != nil {
			return err
		}
		fmt.Printf("%s\n> ", answer)
	}
	return in.Err()
}

func newFilesAgentSession(ctx context.Context, dir string) (*crux.Session, error) {
	if err := crux.AddToolset(crux.Filesystem(dir)); err != nil {
		return nil, err
	}

	agent, err := crux.New("files", crux.OpenAIGPT5_4,
		crux.WithInstructions("You help with files in the current project. Paths are relative to the project root."),
		crux.WithToolsets("filesystem"),
		crux.WithMaxTurns(30),
	)
	if err != nil {
		return nil, err
	}

	return crux.NewSession(ctx, agent)
}

// answerWithApprovals runs one prompt, asking on the terminal about each tool
// call that needs approval, and returns the answer.
func answerWithApprovals(ctx context.Context, session *crux.Session, in *bufio.Scanner, prompt string) (string, error) {
	answer, err := session.Run(ctx, prompt)
	for errors.Is(err, crux.ErrApprovalNeeded) {
		for _, call := range session.PendingApprovals() {
			if err := decide(ctx, session, in, call); err != nil {
				return "", err
			}
		}
		answer, err = session.Resume(ctx)
	}
	return answer, err
}

func decide(ctx context.Context, session *crux.Session, in *bufio.Scanner, call *crux.ToolCall) error {
	fmt.Printf("Allow %s %s? [y/N] ", call.Name, call.Args)
	if in.Scan() && strings.EqualFold(strings.TrimSpace(in.Text()), "y") {
		return session.Approve(ctx, call.ID)
	}
	return session.Reject(ctx, call.ID, "the user declined")
}
