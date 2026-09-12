package crux

import (
	"os"
	"sync"
)

type Provider string

const (
	ProviderOpenAI     Provider = "openai"
	ProviderAnthropic  Provider = "anthropic"
	ProviderGoogle     Provider = "google"
	ProviderOpenrouter Provider = "openrouter"
	ProviderDeepSeek   Provider = "deepseek"
)

type Model struct {
	Name string
}

var (
	providers = map[Provider][]Model{}

	providerMu *sync.RWMutex = &sync.RWMutex{}
)

func inferProvider(modelName string) Provider {
	for provider, models := range providers {
		for _, model := range models {
			if model.Name == modelName {
				return provider
			}
		}
	}
	return ""
}

func discoverAPIKey(provider Provider) string {
	var envVars = []string{}

	switch provider {
	case ProviderOpenAI:
		envVars = []string{"OPENAI_API_KEY", "OPENAI_APIKEY", "OPENAI_KEY"}
	case ProviderAnthropic:
		envVars = []string{"ANTHROPIC_API_KEY", "ANTHROPIC_APIKEY", "ANTHROPIC_KEY", "ANTHROPIC_AUTH_TOKEN"}
	case ProviderGoogle:
		envVars = []string{"GOOGLE_API_KEY", "GOOGLE_APIKEY", "GOOGLE_KEY", "GEMINI_API_KEY", "GEMINI_APIKEY", "GEMINI_KEY"}
	case ProviderOpenrouter:
		envVars = []string{"OPENROUTER_API_KEY", "OPENROUTER_APIKEY", "OPENROUTER_KEY"}
	}

	for _, envVar := range envVars {
		if value := os.Getenv(envVar); value != "" {
			return value
		}
	}

	return ""
}
