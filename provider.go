package crux

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"
)

type Provider string

const (
	ProviderOpenAI     Provider = "openai"
	ProviderAnthropic  Provider = "anthropic"
	ProviderGoogle     Provider = "google"
	ProviderOpenrouter Provider = "openrouter"
	ProviderDeepSeek   Provider = "deepseek"
	ProviderXAI        Provider = "xai"
	ProviderOllama     Provider = "ollama"
)

// model is a known model name, used to infer a provider.
type model struct {
	Name string
}

// providerSpec is everything crux knows about one provider.
type providerSpec struct {
	models  []model
	envVars []string // checked in order for an API key
	baseURL string   // "" uses the SDK default
	step    func(a *Agent, ctx context.Context, log []Entry, httpClient *http.Client, emit chunkSink) ([]Entry, error)
	schema  func(root map[string]any, provider Provider) (map[string]any, error)
	prepare func(a *Agent) error // optional; provider rules and defaults, run at the end of New
}

var providerSpecs = map[Provider]providerSpec{}

// registerProvider adds a provider. It is called only from init functions.
func registerProvider(provider Provider, spec providerSpec) {
	if _, dup := providerSpecs[provider]; dup || spec.step == nil || spec.schema == nil {
		panic(fmt.Sprintf("crux: invalid registration of provider %q", provider))
	}
	providerSpecs[provider] = spec
}

func inferProvider(modelName string) Provider {
	var matched Provider
	for provider, spec := range providerSpecs {
		for _, m := range spec.models {
			if m.Name == modelName {
				if matched != "" && matched != provider {
					return "" // Ambiguous IDs require WithProvider.
				}
				matched = provider
			}
		}
	}
	return matched
}

func firstEnv(names []string) string {
	for _, name := range names {
		if value := os.Getenv(name); value != "" {
			return value
		}
	}
	return ""
}

// defaultHTTPClient is used when neither the agent nor the session sets a
// client and the SDK would not supply its own. Like the OpenAI SDK's default, it
// gives up on a server that accepts a request but never sends response headers;
// the body is not limited, so long streams are unaffected. It is built on first
// use, so a wrapped http.DefaultTransport (for tracing, say) is kept, though
// then without the timeout.
var defaultHTTPClient = sync.OnceValue(func() *http.Client {
	return newHTTPClient(10 * time.Minute)
})

func newHTTPClient(responseHeaderTimeout time.Duration) *http.Client {
	if transport, ok := http.DefaultTransport.(*http.Transport); ok {
		transport = transport.Clone()
		transport.ResponseHeaderTimeout = responseHeaderTimeout
		return &http.Client{Transport: transport}
	}
	return &http.Client{Transport: http.DefaultTransport}
}
