package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"time"

	"crux.foo"
)

// The session log records the whole lifecycle of a run: when it started and
// how it ended, every provider request, and every tool start. This example
// watches those entries live with WithEntryHandler, stops for an approval,
// and then rebuilds a summary of each run from the stored log alone.
//
//	go run ./examples/lifecycle

type orderArgs struct {
	OrderID string `json:"order_id" description:"Order ID, such as A-1001"`
}

type Order struct {
	ID     string `json:"id"`
	Status string `json:"status"`
	Cents  int    `json:"total_cents"`
}

// Fictional fixtures.
var orders = map[string]Order{
	"A-1001": {"A-1001", "delivered", 4999},
	"A-1002": {"A-1002", "delivered, damaged", 12900},
}

func lookupOrder(ctx context.Context, args orderArgs) (Order, error) {
	time.Sleep(150 * time.Millisecond) // pretend to call a slow backend
	order, ok := orders[args.OrderID]
	if !ok {
		return Order{}, fmt.Errorf("unknown order %q", args.OrderID)
	}
	return order, nil
}

func refundOrder(ctx context.Context, args orderArgs) (string, error) {
	return "refund issued for " + args.OrderID, nil
}

func init() {
	crux.RegisterTool("lookup_order", "Look up an order's status and total", lookupOrder)
	// Refunds move money, so a person approves each one before it runs.
	crux.RegisterTool("refund_order", "Refund an order in full", refundOrder, crux.WithApprovalNeeded(true))
}

// trace prints each entry as the session stores it. Subagent sessions report
// to the same handler; s is the session that stored the entry.
func trace(ctx context.Context, s *crux.Session, e crux.Entry) {
	who := fmt.Sprintf("%-8s", s.Agent().Name())
	switch e.Kind {
	case crux.KindRunStarted:
		fmt.Printf("%s ▶ run started\n", who)
	case crux.KindRunFinished:
		line := fmt.Sprintf("%s ■ run finished: %s", who, e.Run.Outcome)
		if e.Run.Outcome == crux.RunFailed {
			line += " (" + e.Run.Error + ")"
		}
		fmt.Println(line)
	case crux.KindTurnStarted:
		fmt.Printf("%s   → request to %s/%s\n", who, e.Turn.Provider, e.Turn.Model)
	case crux.KindToolCall:
		fmt.Printf("%s   ← model calls %s %s\n", who, e.ToolCall.Name, e.ToolCall.Args)
	case crux.KindToolStarted:
		fmt.Printf("%s     tool %s started\n", who, e.ToolCall.Name)
	case crux.KindToolResult:
		status := "ok"
		switch {
		case e.ToolResult.Denied:
			status = "denied: " + e.ToolResult.Error
		case e.ToolResult.Error != "":
			status = "error: " + e.ToolResult.Error
		}
		fmt.Printf("%s     tool result in %s (%s)\n", who, e.Duration.Round(time.Millisecond), status)
	case crux.KindApproval:
		fmt.Printf("%s   ✓ approval recorded for %s: %v\n", who, e.Approval.CallID, e.Approval.Approved)
	case crux.KindAssistant:
		fmt.Printf("%s   ← answer: %q\n", who, e.Text())
	}
}

// runSummary is what one Run, Resume or Stream did, read back from the log.
type runSummary struct {
	outcome    crux.RunOutcome
	requests   int
	tools      []string
	usage      crux.Usage
	firstToken time.Duration // only for streamed responses
	took       time.Duration
}

// summarize rebuilds each run of the session from its stored entries. It needs
// nothing but the log, so it works just as well on a session loaded from a
// GORMStore after a restart.
func summarize(entries []crux.Entry) []runSummary {
	var runs []runSummary
	var current *runSummary
	var started time.Time
	for _, e := range entries {
		switch e.Kind {
		case crux.KindRunStarted:
			runs = append(runs, runSummary{})
			current, started = &runs[len(runs)-1], e.At
		case crux.KindRunFinished:
			current.outcome = e.Run.Outcome
			current.took = e.At.Sub(started)
		case crux.KindTurnStarted:
			current.requests++
		case crux.KindToolStarted:
			current.tools = append(current.tools, e.ToolCall.Name)
		}
		if current == nil {
			continue
		}
		if e.Usage != nil {
			current.usage.InputTokens += e.Usage.InputTokens
			current.usage.OutputTokens += e.Usage.OutputTokens
		}
		if e.Response != nil && e.Response.FirstTokenAfter > 0 {
			current.firstToken = e.Response.FirstTokenAfter
		}
	}
	return runs
}

func main() {
	ctx := context.Background()

	// A subagent's entries reach the parent's handler too.
	orderDesk := crux.Must(crux.New(
		"orders",
		crux.Gemini3_8Flash,
		crux.WithInstructions("Look up the orders you are asked about and report their status and total."),
		crux.WithTools([]string{"lookup_order"}),
	))
	support := crux.Must(crux.New(
		"support",
		crux.Gemini3_8Flash,
		crux.WithInstructions(`You are a support agent. Ask the orders agent about orders.
Refund an order only when it arrived damaged. Answer in one or two sentences.`),
		crux.WithTools([]string{"refund_order"}),
		crux.WithSubAgent(orderDesk, "Looks up orders by ID"),
	))

	session := crux.MustSession(crux.NewSession(ctx, support, crux.WithEntryHandler(trace)))

	fmt.Println("── Run ──")
	_, err := session.Run(ctx, "Orders A-1001 and A-1002 arrived. Anything you can do about them?")
	for errors.Is(err, crux.ErrApprovalNeeded) {
		for _, call := range session.PendingApprovals() {
			// A real application would ask a person here.
			fmt.Printf("\n%s wants to run %s %s: approving\n", call.Agent, call.Name, call.Args)
			if err := session.Approve(ctx, call.ID); err != nil {
				log.Fatal(err)
			}
		}

		// Resume by streaming, so the log also records the time to first token.
		fmt.Println("\n── Resume ──")
		err = nil
		for _, streamErr := range session.Stream(ctx, nil) {
			if streamErr != nil {
				err = streamErr
			}
		}
	}
	if err != nil {
		log.Fatal(err)
	}

	answer, _ := session.FinalOutput()
	fmt.Printf("\nAnswer: %s\n", answer)

	fmt.Println("\n── Summary from the log ──")
	for i, run := range summarize(session.Logs()) {
		line := fmt.Sprintf("run %d: %-15s requests: %d, tokens: %d in + %d out, took %s",
			i+1, run.outcome, run.requests, run.usage.InputTokens, run.usage.OutputTokens, run.took.Round(time.Millisecond))
		if len(run.tools) > 0 {
			line += ", tools: " + strings.Join(run.tools, ", ")
		}
		if run.firstToken > 0 {
			line += fmt.Sprintf(", first token after %s", run.firstToken.Round(time.Millisecond))
		}
		fmt.Println(line)
	}
	fmt.Println("(subagent runs are stored in their own sessions and show up only in the live trace)")
}
