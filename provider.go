package crux

import (
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

var (
	providers = map[Provider][]model{}

	providerMu *sync.RWMutex = &sync.RWMutex{}
)

func inferProvider(modelName string) Provider {
	providerMu.RLock()
	defer providerMu.RUnlock()
	var matched Provider
	for provider, models := range providers {
		for _, m := range models {
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

func discoverAPIKey(provider Provider) string {
	var envVars = []string{}

	switch provider {
	case ProviderOpenAI:
		envVars = []string{"OPENAI_API_KEY", "OPENAI_APIKEY", "OPENAI_KEY"}
	case ProviderAnthropic:
		// ANTHROPIC_AUTH_TOKEN is a bearer token, not an API key; the SDK's
		// default options send it as Authorization when no key is set.
		// https://github.com/anthropics/anthropic-sdk-go/blob/v1.72.0/client.go
		envVars = []string{"ANTHROPIC_API_KEY", "ANTHROPIC_APIKEY", "ANTHROPIC_KEY"}
	case ProviderGoogle:
		envVars = []string{"GOOGLE_API_KEY", "GOOGLE_APIKEY", "GOOGLE_KEY", "GEMINI_API_KEY", "GEMINI_APIKEY", "GEMINI_KEY"}
	case ProviderOpenrouter:
		envVars = []string{"OPENROUTER_API_KEY", "OPENROUTER_APIKEY", "OPENROUTER_KEY"}
	case ProviderXAI:
		envVars = []string{"XAI_API_KEY", "XAI_APIKEY", "XAI_KEY"}
	case ProviderDeepSeek:
		envVars = []string{"DEEPSEEK_API_KEY", "DEEPSEEK_APIKEY", "DEEPSEEK_KEY"}
	case ProviderOllama:
		envVars = []string{"OLLAMA_API_KEY", "OLLAMA_APIKEY", "OLLAMA_KEY"}
	}

	for _, envVar := range envVars {
		if value := os.Getenv(envVar); value != "" {
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
