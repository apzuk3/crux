// Spawning: a coordinator that creates its own agents.
//
// The coordinator has no fixed subagents. With WithAgentSpawning it writes
// an agent for each piece of work (its name, instructions and tools) and runs
// it. Spawned agents may only use the tools and models allowed here, and the
// one tool that sends a report needs your approval, even when a spawned agent
// calls it.
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

	// The coordinator itself has no tools besides spawn_agent: everything it
	// learns comes from the agents it creates.
	coordinator := crux.Must(crux.New("coordinator", crux.OpenAIGPT5_4,
		crux.WithoutTools(),
		crux.WithInstructions(`You coordinate work by spawning agents; you have no other tools.
Spawn one analyst per product, giving each only the tools it needs, and a cheaper model for simple lookups.
Then spawn a writer to compare the analyses and send the report. Give every agent all the context it needs in its task.`),
		crux.WithAgentSpawning(
			crux.WithSpawnModels(crux.OpenAIGPT5_4Mini, crux.OpenAIGPT5_4),
			crux.WithSpawnTools("get_specs", "get_reviews", "send_report"),
			crux.WithSpawnMaxTurns(6),
			crux.WithSpawnConcurrency(2),
		),
	))

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	session := crux.MustSession(crux.NewSession(ctx, coordinator, crux.WithEntryHandler(trace)))
	output, err := session.Run(ctx, "Compare the TrailLight and CampGlow lanterns for a rainy hiking trip and email the report to team@example.com.")
	for errors.Is(err, crux.ErrApprovalNeeded) {
		if err := decide(ctx, session); err != nil {
			return err
		}
		output, err = session.Resume(ctx)
	}
	if err != nil {
		return fmt.Errorf("run coordinator: %w", err)
	}
	fmt.Printf("\n[coordinator]\n%s\n", output)
	return nil
}

// trace prints each agent the coordinator spawns and each tool call made,
// by the coordinator or by a spawned agent.
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
