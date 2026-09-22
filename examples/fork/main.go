package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/apzuk3/crux"
)

func main() {
	openAIKey := os.Getenv("OPENAI_API_KEY")
	geminiKey := os.Getenv("GEMINI_API_KEY")
	if geminiKey == "" {
		geminiKey = os.Getenv("GOOGLE_API_KEY")
	}
	if openAIKey == "" || geminiKey == "" {
		log.Fatal("set OPENAI_API_KEY and GEMINI_API_KEY (or GOOGLE_API_KEY)")
	}

	ctx := context.Background()
	agent := crux.Must(crux.New(
		"travel-planner",
		crux.ChatModelGPT4_1Mini,
		crux.WithProvider(crux.ProviderOpenAI),
		crux.WithAPIKey(openAIKey),
		crux.WithInstructions("You are a helpful travel planner. Keep answers concise."),
	))

	response, err := agent.Run(ctx, "Plan a one-day visit to Rome for two adults with a 150 euro budget. We prefer history and vegetarian food.")
	if err != nil {
		log.Fatalf("OpenAI: %v", err)
	}
	fmt.Printf("OpenAI:\n%s\n\n", response)

	// Fork at the end of the completed conversation, retaining its full history.
	// Connection settings are inherited, so explicitly use Gemini's credentials.
	fork, err := agent.Fork(
		crux.WithProvider(crux.ProviderGoogle),
		crux.WithModel(crux.Gemini2_5Flash),
		crux.WithAPIKey(geminiKey),
		crux.WithBaseURL(""),
	)
	if err != nil {
		log.Fatalf("fork: %v", err)
	}

	// Gemini sees the original request and OpenAI's answer in the copied history.
	response, err = fork.Run(ctx,
		"Revise that itinerary for a rainy day, keeping our original budget and preferences. Explain what you changed.")
	if err != nil {
		log.Fatalf("Gemini: %v", err)
	}
	fmt.Printf("Gemini fork:\n%s\n", response)
}
