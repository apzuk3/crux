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

	"github.com/apzuk3/crux"
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

	if err := crux.AddToolset(crux.Filesystem(dir)); err != nil {
		return err
	}

	agent, err := crux.New("files", crux.ChatModelGPT5_4,
		crux.WithInstructions("You help with files in the current project. Paths are relative to the project root."),
		crux.WithToolsets("filesystem"),
		crux.WithMaxTurns(30),
	)
	if err != nil {
		return err
	}

	session, err := crux.NewSession(ctx, agent)
	if err != nil {
		return err
	}

	in := bufio.NewScanner(os.Stdin)
	fmt.Print("> ")
	for in.Scan() {
		answer, err := session.Run(ctx, in.Text())
		for errors.Is(err, crux.ErrApprovalNeeded) {
			for _, call := range session.PendingApprovals() {
				fmt.Printf("Allow %s %s? [y/N] ", call.Name, call.Args)
				if in.Scan() && strings.EqualFold(strings.TrimSpace(in.Text()), "y") {
					err = session.Approve(ctx, call.ID)
				} else {
					err = session.Reject(ctx, call.ID, "the user declined")
				}
				if err != nil {
					return err
				}
			}
			answer, err = session.Resume(ctx)
		}
		if err != nil {
			return err
		}
		fmt.Printf("%s\n> ", answer)
	}
	return in.Err()
}
