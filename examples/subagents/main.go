package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/apzuk3/crux"
)

// WithSubAgent uses the child's output schema for its tool arguments too.
// Text contains the task on input and the completed work on output.
type Message struct {
	Text string `json:"text" jsonschema:"description=The task or text to process on input; the completed work on output"`
}

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		return fmt.Errorf("set OPENAI_API_KEY before running this example")
	}

	// Only the researcher can access these fictional product facts.
	registry := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(registry, "get_product_facts", "Get fictional demo product facts",
		func(ctx context.Context, args struct{}) (string, *crux.StateDelta, error) {
			fmt.Println("[researcher] called get_product_facts")
			return "TrailLight is a fictional camping lantern. It weighs 180 grams, lasts 12 hours per charge, costs $39, and is splash-resistant but not waterproof.", nil, nil
		})

	researcher := crux.Must(crux.New("researcher", crux.ChatModelGPT4_1Mini,
		crux.WithAPIKey(key),
		crux.WithToolsRegistry([]string{"get_product_facts"}, registry),
		crux.WithInstructions(`Read the task in the input's text field.
You must call get_product_facts to retrieve the fictional product facts.
Return a concise factual brief in the output's text field, preserving all numbers and limitations.`),
		crux.WithOutputSchemaFrom[Message](),
	))
	editor := crux.Must(crux.New("editor", crux.ChatModelGPT4_1Mini,
		crux.WithAPIKey(key),
		crux.WithInstructions(`Read the brief in the input's text field.
Rewrite it as a friendly two-sentence product description in the output's text field.
Preserve the price and splash-resistant-but-not-waterproof limitation. Do not invent facts.`),
		crux.WithOutputSchemaFrom[Message](),
	))
	mainAgent := crux.Must(crux.New("coordinator", crux.ChatModelGPT4_1Mini,
		crux.WithAPIKey(key),
		crux.WithInstructions(`You coordinate specialists. You must delegate rather than write the description yourself.
First call researcher with a task in its text argument asking for a factual TrailLight brief.
After receiving the researcher's result, call editor with that brief in its text argument.
Finally return the editor's product description to the user.`),
		crux.WithSubAgent(researcher, "Research TrailLight product facts. Pass the research task in text."),
		crux.WithSubAgent(editor, "Edit a factual brief into a product description. Pass the brief in text."),
	))

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	fmt.Println("Running coordinator → researcher → editor...")
	output, err := mainAgent.Run(ctx, "Create a short product description for the fictional TrailLight lantern using both specialists.")
	if err != nil {
		return fmt.Errorf("run coordinator: %w", err)
	}

	// WithSubAgent stores each successful child's output under its name.
	// Check this instead of trusting a claim of delegation in the final answer.
	state := mainAgent.StateSnapshot()
	for _, name := range []string{"researcher", "editor"} {
		raw, ok := state[name].(string)
		if !ok {
			return fmt.Errorf("delegation check failed: no successful %s result in coordinator state", name)
		}
		var result Message
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return fmt.Errorf("decode %s result: %w", name, err)
		}
		if strings.TrimSpace(result.Text) == "" {
			return fmt.Errorf("delegation check failed: %s returned empty text", name)
		}
		fmt.Printf("\n[%s result]\n%s\n", name, result.Text)
	}
	fmt.Printf("\n[coordinator answer]\n%s\n\nPASS: coordinator called both subagents successfully.\n", output)
	return nil
}
