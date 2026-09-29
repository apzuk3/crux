package crux

// Native DeepSeek model IDs documented for the Responses endpoint.
// Source: https://api-docs.deepseek.com/guides/responses_api
const (
	DeepSeekFlash = "deepseek-flash"
)

func init() {
	providerMu.Lock()
	defer providerMu.Unlock()
	providers[ProviderDeepSeek] = []model{
		{Name: DeepSeekFlash},
	}
}
