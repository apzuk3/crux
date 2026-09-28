package crux_test

import (
	"context"
	"errors"
	"fmt"
	"log"

	"github.com/apzuk3/crux"
)

type WeatherArgs struct {
	City string `json:"city" description:"City name, e.g. Paris"`
}

func Example() {
	// Tools are registered once, anywhere in the program, and referenced by name.
	crux.RegisterTool("get_weather", "Get the current weather for a city",
		func(ctx context.Context, in WeatherArgs) (string, error) {
			return "Sunny, 22°C in " + in.City, nil
		})

	agent := crux.Must(crux.New("assistant", crux.ClaudeHaiku4_5,
		crux.WithInstructions("You are a helpful assistant."),
		crux.WithTools([]string{"get_weather"}),
	))

	ctx := context.Background()
	session := crux.MustSession(crux.NewSession(ctx, agent))
	answer, err := session.Run(ctx, "What's the weather in Paris?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(answer)
}

func ExampleSession_RunInto() {
	type Forecast struct {
		City    string `json:"city"`
		Summary string `json:"summary"`
	}

	agent := crux.Must(crux.New("forecaster", crux.ChatModelGPT5_4,
		crux.WithOutputSchemaFrom[Forecast](),
	))

	ctx := context.Background()
	session := crux.MustSession(crux.NewSession(ctx, agent))

	var forecast Forecast
	if err := session.RunInto(ctx, "Describe a typical spring day in Paris.", &forecast); err != nil {
		log.Fatal(err)
	}
	fmt.Println(forecast.City, forecast.Summary)
}

func ExampleSession_Stream() {
	agent := crux.Must(crux.New("writer", crux.ClaudeHaiku4_5))

	ctx := context.Background()
	session := crux.MustSession(crux.NewSession(ctx, agent))
	for chunk, err := range session.Stream(ctx, "Write a haiku about Go.") {
		if err != nil {
			log.Fatal(err)
		}
		if chunk.Kind == crux.ChunkText {
			fmt.Print(chunk.Delta)
		}
	}
}

func ExampleSession_Approve() {
	crux.RegisterTool("delete_file", "Delete a file",
		func(ctx context.Context, in struct {
			Path string `json:"path"`
		}) (string, error) {
			return "deleted " + in.Path, nil
		},
		crux.WithApprovalNeeded(true),
	)

	agent := crux.Must(crux.New("janitor", crux.ClaudeHaiku4_5,
		crux.WithTools([]string{"delete_file"})))

	ctx := context.Background()
	session := crux.MustSession(crux.NewSession(ctx, agent))

	_, err := session.Run(ctx, "Delete /tmp/cache.txt")
	for errors.Is(err, crux.ErrApprovalNeeded) {
		for _, call := range session.PendingApprovals() {
			fmt.Printf("approving %s(%s)\n", call.Name, call.Args)
			if err := session.Approve(ctx, call.ID); err != nil {
				log.Fatal(err)
			}
		}
		_, err = session.Resume(ctx)
	}
	if err != nil {
		log.Fatal(err)
	}
}
