package crux

const (
	Gemini3_8Flash                          = "gemini-3.8-flash"
	Gemini3_7Flash                          = "gemini-3.7-flash"
	Gemini3_6Flash                          = "gemini-3.6-flash"
	Gemini3_5Flash                          = "gemini-3.5-flash"
	Gemini3_5FlashLite                      = "gemini-3.5-flash-lite"
	Gemini3_1FlashLite                      = "gemini-3.1-flash-lite"
	Gemini3_1FlashImage                     = "gemini-3.1-flash-image"
	Gemini3_1FlashLiteImage                 = "gemini-3.1-flash-lite-image"
	Gemini3ProImage                         = "gemini-3-pro-image"
	Gemini3_1ProPreview                     = "gemini-3.1-pro-preview"
	Gemini3FlashPreview                     = "gemini-3-flash-preview"
	Gemini3_5LiveTranslatePreview           = "gemini-3.5-live-translate-preview"
	Gemini3_1FlashLivePreview               = "gemini-3.1-flash-live-preview"
	Gemini3_1FlashTTSPreview                = "gemini-3.1-flash-tts-preview"
	GeminiOmni1_1Flash                      = "gemini-omni-1.1-flash"
	Gemini3_5Transcribe                     = "gemini-3.5-transcribe"
	Gemini3_5TranscribeLive                 = "gemini-3.5-transcribe-live"
	Gemini2_5Pro                            = "gemini-2.5-pro"
	Gemini2_5Flash                          = "gemini-2.5-flash"
	Gemini2_5FlashLite                      = "gemini-2.5-flash-lite"
	Gemini2_5FlashImage                     = "gemini-2.5-flash-image"
	Gemini2_5FlashNativeAudioPreview12_2025 = "gemini-2.5-flash-native-audio-preview-12-2025"
	Gemini2_5FlashPreviewTTS                = "gemini-2.5-flash-preview-tts"
	Gemini2_5ProPreviewTTS                  = "gemini-2.5-pro-preview-tts"
	Gemini2_5ComputerUsePreview10_2025      = "gemini-2.5-computer-use-preview-10-2025"
	GeminiEmbedding2Preview                 = "gemini-embedding-2-preview"
	GeminiEmbedding001                      = "gemini-embedding-001"
	GeminiRoboticsER2Preview                = "gemini-robotics-er-2-preview"
	GeminiRoboticsER1_6Preview              = "gemini-robotics-er-1.6-preview"
)

func init() {
	providerMu.Lock()
	defer providerMu.Unlock()

	providers[ProviderGoogle] = []model{
		{Name: Gemini3_8Flash}, {Name: Gemini3_7Flash},
		{Name: Gemini3_6Flash}, {Name: Gemini3_5Flash},
		{Name: Gemini3_5FlashLite}, {Name: Gemini3_1FlashLite},
		{Name: Gemini3_1FlashImage}, {Name: Gemini3_1FlashLiteImage},
		{Name: Gemini3ProImage}, {Name: Gemini3_1ProPreview},
		{Name: Gemini3FlashPreview}, {Name: Gemini3_5LiveTranslatePreview},
		{Name: Gemini3_1FlashLivePreview}, {Name: Gemini3_1FlashTTSPreview},
		{Name: GeminiOmni1_1Flash}, {Name: Gemini3_5Transcribe},
		{Name: Gemini3_5TranscribeLive},
		{Name: Gemini2_5Pro}, {Name: Gemini2_5Flash}, {Name: Gemini2_5FlashLite},
		{Name: Gemini2_5FlashImage}, {Name: Gemini2_5FlashNativeAudioPreview12_2025},
		{Name: Gemini2_5FlashPreviewTTS}, {Name: Gemini2_5ProPreviewTTS},
		{Name: Gemini2_5ComputerUsePreview10_2025},
		{Name: GeminiEmbedding2Preview}, {Name: GeminiEmbedding001},
		{Name: GeminiRoboticsER2Preview}, {Name: GeminiRoboticsER1_6Preview},
	}
}
