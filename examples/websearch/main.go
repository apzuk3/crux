// Command websearch smoke-tests provider-executed web search.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"time"

	"crux.foo"
)

type providerCase struct {
	provider crux.Provider
	model    string
	keyEnv   []string
	skip     string
}

type outcome int

const (
	caseSucceeded outcome = iota
	caseFailed
	caseSkipped
)

func main() {
	os.Exit(run())
}

func run() int {
	provider := flag.String("provider", "all", "Provider: all, openai, anthropic, google (or gemini), openrouter, deepseek")
	model := flag.String("model", "", "Override the model (requires a single supported provider)")
	baseURL := flag.String("base-url", "", "Override the API base URL (requires a single supported provider)")
	prompt := flag.String("prompt", "Search the web for the latest stable Go release as of "+time.Now().UTC().Format("2006-01-02")+". Give its version, release date, and official source URLs. Use live web search rather than memory.", "Search prompt")
	timeout := flag.Duration("timeout", 2*time.Minute, "Timeout per provider")
	flag.Parse()

	cases, ok := selectCases(*provider)
	if !ok {
		fmt.Fprintf(os.Stderr, "unknown provider %q; use -help for available providers\n", *provider)
		return 2
	}
	if err := validateFlags(cases, *model, *baseURL, *prompt, *timeout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	fmt.Printf("Prompt: %s\n\nSUCCESS means a nonempty response with web search enabled; search execution is not observable through Agent.Run.\n\n", *prompt)
	succeeded, failed, skipped := 0, 0, 0
	for _, c := range cases {
		switch runCase(ctx, c, *model, *baseURL, *prompt, *timeout) {
		case caseSkipped:
			skipped++
			continue
		case caseFailed:
			failed++
		default:
			succeeded++
		}
		if ctx.Err() != nil {
			fmt.Fprintln(os.Stderr, "Interrupted")
			return 1
		}
	}
	fmt.Printf("Summary: %d succeeded, %d failed, %d skipped\n", succeeded, failed, skipped)
	if failed > 0 || succeeded == 0 {
		return 1
	}
	return 0
}

// selectCases returns the providers to run for the -provider flag; ok is
// false when the flag names no known provider.
func selectCases(provider string) (cases []providerCase, ok bool) {
	cases = []providerCase{
		{provider: crux.ProviderOpenAI, model: crux.OpenAIGPT4_1Mini,
			keyEnv: []string{"OPENAI_API_KEY", "OPENAI_APIKEY", "OPENAI_KEY"}},
		{provider: crux.ProviderAnthropic, model: crux.ClaudeHaiku4_5,
			keyEnv: []string{"ANTHROPIC_API_KEY", "ANTHROPIC_APIKEY", "ANTHROPIC_KEY", "ANTHROPIC_AUTH_TOKEN"}},
		{provider: crux.ProviderGoogle, model: crux.Gemini2_5Flash,
			keyEnv: []string{"GOOGLE_API_KEY", "GOOGLE_APIKEY", "GOOGLE_KEY", "GEMINI_API_KEY", "GEMINI_APIKEY", "GEMINI_KEY"}},
		{provider: crux.ProviderOpenrouter, skip: "native web search is not supported by the OpenRouter adapter"},
		{provider: crux.ProviderDeepSeek, skip: "no execution adapter is implemented"},
	}
	selected := strings.ToLower(strings.TrimSpace(provider))
	if selected == "gemini" {
		selected = string(crux.ProviderGoogle)
	}
	if selected == "all" {
		return cases, true
	}
	var matches []providerCase
	for _, c := range cases {
		if string(c.provider) == selected {
			matches = append(matches, c)
		}
	}
	return matches, len(matches) > 0
}

func validateFlags(cases []providerCase, model, baseURL, prompt string, timeout time.Duration) error {
	if flag.NArg() != 0 || timeout <= 0 || strings.TrimSpace(prompt) == "" {
		return errors.New("expected no positional arguments, a positive timeout, and a nonempty prompt")
	}
	if (model != "" || baseURL != "") && (len(cases) != 1 || cases[0].skip != "") {
		return errors.New("-model and -base-url require a single supported -provider")
	}
	return nil
}

// lookupKey returns the first of the environment variables that is set.
func lookupKey(names []string) string {
	for _, name := range names {
		if key := os.Getenv(name); key != "" {
			return key
		}
	}
	return ""
}

// runCase sends the prompt to one provider and prints what happened.
func runCase(ctx context.Context, c providerCase, model, baseURL, prompt string, timeout time.Duration) outcome {
	if c.skip != "" {
		fmt.Printf("[%s] SKIP: %s\n\n", c.provider, c.skip)
		return caseSkipped
	}
	if model != "" {
		c.model = model
	}
	fmt.Printf("[%s] model=%s\n", c.provider, c.model)
	key := lookupKey(c.keyEnv)
	if key == "" {
		fmt.Printf("SKIP: missing credentials; set one of %s\n\n", strings.Join(c.keyEnv, ", "))
		return caseSkipped
	}
	opts := []crux.AgentOption{
		crux.WithProvider(c.provider),
		crux.WithAPIKey(key),
		crux.WithWebSearch(),
		crux.WithInstructions("Use the provider's web search tool to answer the user's question. Cite source URLs and clearly state if search is unavailable."),
	}
	if baseURL != "" {
		opts = append(opts, crux.WithBaseURL(baseURL))
	}
	agent := crux.Must(crux.New(fmt.Sprintf("websearch-%s", c.provider), c.model, opts...))
	session := crux.MustSession(crux.NewSession(ctx, agent))
	requestCtx, cancel := context.WithTimeout(ctx, timeout)
	start := time.Now()
	answer, err := session.Run(requestCtx, prompt)
	elapsed := time.Since(start).Round(time.Millisecond)
	cancel()
	if err == nil && strings.TrimSpace(answer) == "" {
		err = fmt.Errorf("provider returned an empty response")
	}
	if err != nil {
		fmt.Printf("FAIL (%s): %v\n\n", elapsed, err)
		return caseFailed
	}
	fmt.Printf("SUCCESS (%s)\n%s\n\n", elapsed, answer)
	return caseSucceeded
}
