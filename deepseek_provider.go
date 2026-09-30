package crux

import "errors"

// Native DeepSeek model IDs documented for the Responses endpoint.
// Source: https://api-docs.deepseek.com/guides/responses_api
const (
	DeepSeekFlash = "deepseek-flash"
)

func init() {
	registerProvider(ProviderDeepSeek, providerSpec{
		models: []model{
			{Name: DeepSeekFlash},
		},
		envVars: []string{"DEEPSEEK_API_KEY", "DEEPSEEK_APIKEY", "DEEPSEEK_KEY"},
		baseURL: "https://api.deepseek.com",
		step:    (*Agent).openAIstep, // stateless Responses with plain-text reasoning replay
		schema:  adaptOpenAI,
		prepare: prepareDeepSeek,
	})
}

func prepareDeepSeek(a *Agent) error {
	if a.searchOptions != nil {
		return errors.New("native web search is not supported by this DeepSeek adapter")
	}
	return nil
}
