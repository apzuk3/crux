// Command ollama runs the quickstart agent on a local Ollama model, so it
// needs no API key:
//
//	ollama pull qwen3
//	go run ./examples/ollama
//	go run ./examples/ollama -model llama3.2 "Is it warmer in Rome or Oslo?"
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"strings"

	"crux.foo"
)

type weatherArgs struct {
	City string `json:"city" description:"City name, e.g. Paris"`
}

// getWeather returns a made-up forecast; swap in a real API.
func getWeather(ctx context.Context, in weatherArgs) (string, error) {
	temp := 10 + len(in.City)*2
	return fmt.Sprintf("Sunny, %d°C in %s", temp, in.City), nil
}

func main() {
	model := flag.String("model", crux.OllamaQwen3, "An installed Ollama model that supports tool calls")
	flag.Parse()
	prompt := strings.Join(flag.Args(), " ")
	if prompt == "" {
		prompt = "What's the weather in Paris and in Tokyo?"
	}

	crux.RegisterTool("get_weather", "Get the current weather for a city", getWeather)

	agent := crux.Must(crux.New("assistant", *model,
		crux.WithProvider(crux.ProviderOllama),
		crux.WithInstructions("You are a helpful assistant. Use tools for facts."),
		crux.WithTools([]string{"get_weather"}),
	))

	ctx := context.Background()
	session := crux.MustSession(crux.NewSession(ctx, agent))

	answer, err := session.Run(ctx, prompt)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(answer)
}
