package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"

	"crux.foo"
)

// Triage is what the decider answers about a ticket: one question per field.
type Triage struct {
	Urgent bool   `json:"urgent" description:"Does the customer need help right away?" true:"Money, outages or deadlines are at stake" false:"It can wait"`
	Area   string `json:"area" description:"Which team should handle this?" choices:"billing=Payments, invoices and refunds|technical=Bugs, outages and integrations|sales=Pricing, upgrades and new accounts"`
	Mood   int    `json:"mood" description:"How frustrated is the customer?" levels:"calm|frustrated|very angry"`
}

func main() {
	// Jev needs TYPESAFE_API_KEY; crux.OpenRouterDecisionModelJev1_13 needs
	// OPENROUTER_API_KEY. Any other model answers through structured output.
	model := flag.String("model", crux.Jev, "Decision model, or any generative model")
	ticket := flag.String("ticket", "Help! My payouts have been failing for 3 days.", "Ticket to triage")
	flag.Parse()
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	decider := crux.MustDecider(crux.NewDecider(*model))
	res, err := crux.Decide[Triage](ctx, decider, *ticket)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%+v\n", res.Value)
	for question, confidence := range res.Confidence {
		fmt.Printf("%s: confidence %.2f, probabilities %v\n", question, confidence, res.Probabilities[question])
	}
}
