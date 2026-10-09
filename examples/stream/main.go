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

func main() {
	provider := flag.String("provider", "openai", "Provider: openai, anthropic, google, openrouter, xai, deepseek, ollama")
	model := flag.String("model", crux.OpenAIGPT5_4, "Provider model ID")
	prompt := flag.String("prompt", "Explain why binary search is logarithmic", "Message to send")
	baseURL := flag.String("base-url", "", "Optional provider base URL")
	flag.Parse()
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	opts := []crux.AgentOption{crux.WithProvider(crux.Provider(*provider))}
	if *baseURL != "" {
		opts = append(opts, crux.WithBaseURL(*baseURL))
	}
	agent := crux.Must(crux.New("stream-demo", *model, opts...))
	session := crux.MustSession(crux.NewSession(ctx, agent))
	var previous crux.Chunk
	for chunk, err := range session.Stream(ctx, *prompt) {
		if err != nil {
			log.Fatal(err)
		}
		if chunk.Kind != previous.Kind || chunk.Turn != previous.Turn {
			fmt.Printf("\n[%s · turn %d]\n", chunk.Kind, chunk.Turn)
		}
		fmt.Print(chunk.Delta)
		previous = chunk
	}
	fmt.Println()
}
