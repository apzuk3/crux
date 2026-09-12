package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/apzuk3/crux"
)

type weatherArgs struct {
	Location string `json:"location" description:"The city to report the weather for"`
}

func getWeather(ctx context.Context, args weatherArgs) (string, error) {
	fmt.Printf("get_weather(%q)\n", args.Location)
	return "good weather", nil
}

func init() {
	crux.RegisterTool("get_weather", "Get weather at the given location", getWeather)
}

type Output struct {
	Answer string `json:"answer"`
}

func main() {
	agent := crux.NewAgent(
		crux.ClaudeHaiku4_5,
		crux.WithInstructions("You are a weather assistant. You are given a location and you need to return the weather at that location."),
		crux.WithAllowedTools([]string{"get_weather"}),
		crux.WithOutputSchemaFrom[Output](),
	)

	output, err := crux.Run[Output](context.Background(), agent, "What is the weather in New York City?")
	if err != nil {
		log.Fatal(err)
	}

	if err := json.NewEncoder(os.Stdout).Encode(output); err != nil {
		log.Fatal(err)
	}
}
