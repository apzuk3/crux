// Spawning: an agent that finds out it can create agents.
//
// The assistant has no instructions and one tool, spawn_agent. Nothing tells
// it to delegate: the tool's description says it can create an agent and
// lists the tools it may give one, and the request below doesn't mention
// agents at all. What it spawns, with which tools and models, is its own
// decision; the trace shows it as it happens.
//
// The tools are fictional product lookups and an email tool that needs your
// approval, even when an agent the assistant created calls it.
//
//	OPENAI_API_KEY=... go run ./examples/spawning
package main

import (
	"bufio"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/apzuk3/crux"
)

// Fictional data the tools serve.
var (
	specs = map[string]string{
		"trailight": "TrailLight lantern: 180 g, 12 h per charge, 400 lumens, splash-resistant (IPX4), $39.",
		"campglow":  "CampGlow lantern: 240 g, 20 h per charge, 300 lumens, waterproof (IPX7), $55.",
	}
	reviews = map[string]string{
		"trailight": "4.4/5 from 812 reviews. Praised: light, bright. Complaints: dies in heavy rain, clip breaks.",
		"campglow":  "4.6/5 from 455 reviews. Praised: survives storms, long battery. Complaints: heavy, dim for reading.",
	}
)

type productArgs struct {
	Product string `json:"product" description:"Product name, such as TrailLight"`
}

type reportArgs struct {
	To   string `json:"to" description:"Email address"`
	Body string `json:"body" description:"The report"`
}

func lookup(data map[string]string) func(context.Context, productArgs) (string, error) {
	return func(ctx context.Context, in productArgs) (string, error) {
		if text, ok := data[strings.ToLower(strings.TrimSpace(in.Product))]; ok {
			return text, nil
		}
		return "", fmt.Errorf("unknown product %q", in.Product)
	}
}

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	if os.Getenv("OPENAI_API_KEY") == "" {
		return errors.New("set OPENAI_API_KEY before running this example")
	}

	crux.RegisterTool("get_specs", "Get a product's specifications", lookup(specs))
	crux.RegisterTool("get_reviews", "Get a summary of a product's customer reviews", lookup(reviews))
	crux.RegisterTool("send_report", "Email a report",
		func(ctx context.Context, in reportArgs) (string, error) {
			fmt.Printf("\n📧 to %s:\n%s\n", in.To, in.Body)
			return "sent", nil
		}, crux.WithApprovalNeeded(true))

	// No instructions and no tools of its own: only the ability to create
	// agents, within these limits.
	assistant := crux.Must(crux.New("assistant", crux.OpenAIGPT5_4,
		crux.WithAgentSpawning(
			crux.WithSpawnModels(crux.OpenAIGPT5_4Mini, crux.OpenAIGPT5_4),
			crux.WithSpawnTools("get_specs", "get_reviews", "send_report"),
			crux.WithSpawnMaxTurns(6),
			crux.WithSpawnConcurrency(2),
		),
	))

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	request := "I'm going hiking in the rain next week. Should I take the TrailLight or the CampGlow lantern? Email the answer to team@example.com."
	fmt.Printf("The assistant's tools: %v\n\n> %s\n\n", assistant.ToolNames(), request)

	session := crux.MustSession(crux.NewSession(ctx, assistant, crux.WithEntryHandler(trace)))
	output, err := session.Run(ctx, request)
	for errors.Is(err, crux.ErrApprovalNeeded) {
		if err := decide(ctx, session); err != nil {
			return err
		}
		output, err = session.Resume(ctx)
	}
	if err != nil {
		return fmt.Errorf("run assistant: %w", err)
	}
	fmt.Printf("\n[assistant]\n%s\n", output)
	return nil
}

// trace prints each agent the assistant creates and each tool call made, by
// the assistant or by an agent it created.
func trace(ctx context.Context, s *crux.Session, e crux.Entry) {
	if e.Kind != crux.KindToolCall || e.ToolCall == nil {
		return
	}
	agent := s.Agent().Name()
	if e.ToolCall.Name != "spawn_agent" {
		fmt.Printf("  [%s] %s %s\n", agent, e.ToolCall.Name, e.ToolCall.Args)
		return
	}
	var spawn struct {
		Name  string   `json:"name"`
		Model string   `json:"model"`
		Tools []string `json:"tools"`
		Task  string   `json:"task"`
	}
	_ = json.Unmarshal(e.ToolCall.Args, &spawn)
	fmt.Printf("[%s] spawns %q (model %s, tools %v)\n    task: %s\n", agent, spawn.Name, cmp.Or(spawn.Model, "default"), spawn.Tools, spawn.Task)
}

// decide asks on the terminal about each call waiting for approval.
func decide(ctx context.Context, session *crux.Session) error {
	in := bufio.NewReader(os.Stdin)
	for _, call := range session.PendingApprovals() {
		fmt.Printf("\n%s wants to call %s %s\nApprove? [y/N] ", call.Agent, call.Name, call.Args)
		answer, _ := in.ReadString('\n')
		var err error
		if strings.EqualFold(strings.TrimSpace(answer), "y") {
			err = session.Approve(ctx, call.ID)
		} else {
			err = session.Reject(ctx, call.ID, "the user declined to send it")
		}
		if err != nil {
			return err
		}
	}
	return nil
}
