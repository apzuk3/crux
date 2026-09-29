package crux

import "github.com/openai/openai-go/v3"

type ChatModel = openai.ChatModel

const (
	ChatModelGPT6Astra                        = openai.ChatModelGPT6Astra
	ChatModelGPT5_6Sol                        = openai.ChatModelGPT5_6Sol
	ChatModelGPT5_6Terra                      = openai.ChatModelGPT5_6Terra
	ChatModelGPT5_6Luna                       = openai.ChatModelGPT5_6Luna
	ChatModelGPT5_5                           = openai.ChatModelGPT5_5
	ChatModelGPT5_5_2026_04_23                = openai.ChatModelGPT5_5_2026_04_23
	ChatModelGPT5_4                           = openai.ChatModelGPT5_4
	ChatModelGPT5_4Mini                       = openai.ChatModelGPT5_4Mini
	ChatModelGPT5_4Nano                       = openai.ChatModelGPT5_4Nano
	ChatModelGPT5_4Mini2026_03_17             = openai.ChatModelGPT5_4Mini2026_03_17
	ChatModelGPT5_4Nano2026_03_17             = openai.ChatModelGPT5_4Nano2026_03_17
	ChatModelGPT5_3ChatLatest                 = openai.ChatModelGPT5_3ChatLatest
	ChatModelGPT5_2                           = openai.ChatModelGPT5_2
	ChatModelGPT5_2_2025_12_11                = openai.ChatModelGPT5_2_2025_12_11
	ChatModelGPT5_2ChatLatest                 = openai.ChatModelGPT5_2ChatLatest
	ChatModelGPT5_2Pro                        = openai.ChatModelGPT5_2Pro
	ChatModelGPT5_2Pro2025_12_11              = openai.ChatModelGPT5_2Pro2025_12_11
	ChatModelGPT5_1                           = openai.ChatModelGPT5_1
	ChatModelGPT5_1_2025_11_13                = openai.ChatModelGPT5_1_2025_11_13
	ChatModelGPT5_1Codex                      = openai.ChatModelGPT5_1Codex
	ChatModelGPT5_1Mini                       = openai.ChatModelGPT5_1Mini
	ChatModelGPT5_1ChatLatest                 = openai.ChatModelGPT5_1ChatLatest
	ChatModelGPT5                             = openai.ChatModelGPT5
	ChatModelGPT5Mini                         = openai.ChatModelGPT5Mini
	ChatModelGPT5Nano                         = openai.ChatModelGPT5Nano
	ChatModelGPT5_2025_08_07                  = openai.ChatModelGPT5_2025_08_07
	ChatModelGPT5Mini2025_08_07               = openai.ChatModelGPT5Mini2025_08_07
	ChatModelGPT5Nano2025_08_07               = openai.ChatModelGPT5Nano2025_08_07
	ChatModelGPT5ChatLatest                   = openai.ChatModelGPT5ChatLatest
	ChatModelGPT4_1                           = openai.ChatModelGPT4_1
	ChatModelGPT4_1Mini                       = openai.ChatModelGPT4_1Mini
	ChatModelGPT4_1Nano                       = openai.ChatModelGPT4_1Nano
	ChatModelGPT4_1_2025_04_14                = openai.ChatModelGPT4_1_2025_04_14
	ChatModelGPT4_1Mini2025_04_14             = openai.ChatModelGPT4_1Mini2025_04_14
	ChatModelGPT4_1Nano2025_04_14             = openai.ChatModelGPT4_1Nano2025_04_14
	ChatModelO4Mini                           = openai.ChatModelO4Mini
	ChatModelO4Mini2025_04_16                 = openai.ChatModelO4Mini2025_04_16
	ChatModelO3                               = openai.ChatModelO3
	ChatModelO3_2025_04_16                    = openai.ChatModelO3_2025_04_16
	ChatModelO3Mini                           = openai.ChatModelO3Mini
	ChatModelO3Mini2025_01_31                 = openai.ChatModelO3Mini2025_01_31
	ChatModelO1                               = openai.ChatModelO1
	ChatModelO1_2024_12_17                    = openai.ChatModelO1_2024_12_17
	ChatModelO1Preview                        = openai.ChatModelO1Preview
	ChatModelO1Preview2024_09_12              = openai.ChatModelO1Preview2024_09_12
	ChatModelO1Mini                           = openai.ChatModelO1Mini
	ChatModelO1Mini2024_09_12                 = openai.ChatModelO1Mini2024_09_12
	ChatModelGPT4o                            = openai.ChatModelGPT4o
	ChatModelGPT4o2024_11_20                  = openai.ChatModelGPT4o2024_11_20
	ChatModelGPT4o2024_08_06                  = openai.ChatModelGPT4o2024_08_06
	ChatModelGPT4o2024_05_13                  = openai.ChatModelGPT4o2024_05_13
	ChatModelGPT4oAudioPreview                = openai.ChatModelGPT4oAudioPreview
	ChatModelGPT4oAudioPreview2024_10_01      = openai.ChatModelGPT4oAudioPreview2024_10_01
	ChatModelGPT4oAudioPreview2024_12_17      = openai.ChatModelGPT4oAudioPreview2024_12_17
	ChatModelGPT4oAudioPreview2025_06_03      = openai.ChatModelGPT4oAudioPreview2025_06_03
	ChatModelGPT4oMiniAudioPreview            = openai.ChatModelGPT4oMiniAudioPreview
	ChatModelGPT4oMiniAudioPreview2024_12_17  = openai.ChatModelGPT4oMiniAudioPreview2024_12_17
	ChatModelGPT4oSearchPreview               = openai.ChatModelGPT4oSearchPreview
	ChatModelGPT4oMiniSearchPreview           = openai.ChatModelGPT4oMiniSearchPreview
	ChatModelGPT4oSearchPreview2025_03_11     = openai.ChatModelGPT4oSearchPreview2025_03_11
	ChatModelGPT4oMiniSearchPreview2025_03_11 = openai.ChatModelGPT4oMiniSearchPreview2025_03_11
	ChatModelChatgpt4oLatest                  = openai.ChatModelChatgpt4oLatest
	ChatModelCodexMiniLatest                  = openai.ChatModelCodexMiniLatest
	ChatModelGPT4oMini                        = openai.ChatModelGPT4oMini
	ChatModelGPT4oMini2024_07_18              = openai.ChatModelGPT4oMini2024_07_18
	ChatModelGPT4Turbo                        = openai.ChatModelGPT4Turbo
	ChatModelGPT4Turbo2024_04_09              = openai.ChatModelGPT4Turbo2024_04_09
	ChatModelGPT4_0125Preview                 = openai.ChatModelGPT4_0125Preview
	ChatModelGPT4TurboPreview                 = openai.ChatModelGPT4TurboPreview
	ChatModelGPT4_1106Preview                 = openai.ChatModelGPT4_1106Preview
	ChatModelGPT4VisionPreview                = openai.ChatModelGPT4VisionPreview
	ChatModelGPT4                             = openai.ChatModelGPT4
	ChatModelGPT4_0314                        = openai.ChatModelGPT4_0314
	ChatModelGPT4_0613                        = openai.ChatModelGPT4_0613
	ChatModelGPT4_32k                         = openai.ChatModelGPT4_32k
	ChatModelGPT4_32k0314                     = openai.ChatModelGPT4_32k0314
	ChatModelGPT4_32k0613                     = openai.ChatModelGPT4_32k0613
	ChatModelGPT3_5Turbo                      = openai.ChatModelGPT3_5Turbo
	ChatModelGPT3_5Turbo16k                   = openai.ChatModelGPT3_5Turbo16k
	ChatModelGPT3_5Turbo0301                  = openai.ChatModelGPT3_5Turbo0301
	ChatModelGPT3_5Turbo0613                  = openai.ChatModelGPT3_5Turbo0613
	ChatModelGPT3_5Turbo1106                  = openai.ChatModelGPT3_5Turbo1106
	ChatModelGPT3_5Turbo0125                  = openai.ChatModelGPT3_5Turbo0125
	ChatModelGPT3_5Turbo16k0613               = openai.ChatModelGPT3_5Turbo16k0613
)

func init() {
	providerMu.Lock()
	defer providerMu.Unlock()

	providers[ProviderOpenAI] = []model{
		{Name: ChatModelGPT6Astra},
		{Name: ChatModelGPT5_6Sol},
		{Name: ChatModelGPT5_6Terra},
		{Name: ChatModelGPT5_6Luna},
		{Name: ChatModelGPT5_5},
		{Name: ChatModelGPT5_5_2026_04_23},
		{Name: ChatModelGPT5_4},
		{Name: ChatModelGPT5_4Mini},
		{Name: ChatModelGPT5_4Nano},
		{Name: ChatModelGPT5_4Mini2026_03_17},
		{Name: ChatModelGPT5_4Nano2026_03_17},
		{Name: ChatModelGPT5_3ChatLatest},
		{Name: ChatModelGPT5_2},
		{Name: ChatModelGPT5_2_2025_12_11},
		{Name: ChatModelGPT5_2ChatLatest},
		{Name: ChatModelGPT5_2Pro},
		{Name: ChatModelGPT5_2Pro2025_12_11},
		{Name: ChatModelGPT5_1},
		{Name: ChatModelGPT5_1_2025_11_13},
		{Name: ChatModelGPT5_1Codex},
		{Name: ChatModelGPT5_1Mini},
		{Name: ChatModelGPT5_1ChatLatest},
		{Name: ChatModelGPT5},
		{Name: ChatModelGPT5Mini},
		{Name: ChatModelGPT5Nano},
		{Name: ChatModelGPT5_2025_08_07},
		{Name: ChatModelGPT5Mini2025_08_07},
		{Name: ChatModelGPT5Nano2025_08_07},
		{Name: ChatModelGPT5ChatLatest},
		{Name: ChatModelGPT4_1},
		{Name: ChatModelGPT4_1Mini},
		{Name: ChatModelGPT4_1Nano},
		{Name: ChatModelGPT4_1_2025_04_14},
		{Name: ChatModelGPT4_1Mini2025_04_14},
		{Name: ChatModelGPT4_1Nano2025_04_14},
		{Name: ChatModelO4Mini},
		{Name: ChatModelO4Mini2025_04_16},
		{Name: ChatModelO3},
		{Name: ChatModelO3_2025_04_16},
		{Name: ChatModelO3Mini},
		{Name: ChatModelO3Mini2025_01_31},
		{Name: ChatModelO1},
		{Name: ChatModelO1_2024_12_17},
		{Name: ChatModelO1Preview},
		{Name: ChatModelO1Preview2024_09_12},
		{Name: ChatModelO1Mini},
		{Name: ChatModelO1Mini2024_09_12},
		{Name: ChatModelGPT4o},
		{Name: ChatModelGPT4o2024_11_20},
		{Name: ChatModelGPT4o2024_08_06},
		{Name: ChatModelGPT4o2024_05_13},
		{Name: ChatModelGPT4oAudioPreview},
		{Name: ChatModelGPT4oAudioPreview2024_10_01},
		{Name: ChatModelGPT4oAudioPreview2024_12_17},
		{Name: ChatModelGPT4oAudioPreview2025_06_03},
		{Name: ChatModelGPT4oMiniAudioPreview},
		{Name: ChatModelGPT4oMiniAudioPreview2024_12_17},
		{Name: ChatModelGPT4oSearchPreview},
		{Name: ChatModelGPT4oMiniSearchPreview},
		{Name: ChatModelGPT4oSearchPreview2025_03_11},
		{Name: ChatModelGPT4oMiniSearchPreview2025_03_11},
		{Name: ChatModelChatgpt4oLatest},
		{Name: ChatModelCodexMiniLatest},
		{Name: ChatModelGPT4oMini},
		{Name: ChatModelGPT4oMini2024_07_18},
		{Name: ChatModelGPT4Turbo},
		{Name: ChatModelGPT4Turbo2024_04_09},
		{Name: ChatModelGPT4_0125Preview},
		{Name: ChatModelGPT4TurboPreview},
		{Name: ChatModelGPT4_1106Preview},
		{Name: ChatModelGPT4VisionPreview},
		{Name: ChatModelGPT4},
		{Name: ChatModelGPT4_0314},
		{Name: ChatModelGPT4_0613},
		{Name: ChatModelGPT4_32k},
		{Name: ChatModelGPT4_32k0314},
		{Name: ChatModelGPT4_32k0613},
		{Name: ChatModelGPT3_5Turbo},
		{Name: ChatModelGPT3_5Turbo16k},
		{Name: ChatModelGPT3_5Turbo0301},
		{Name: ChatModelGPT3_5Turbo0613},
		{Name: ChatModelGPT3_5Turbo1106},
		{Name: ChatModelGPT3_5Turbo0125},
		{Name: ChatModelGPT3_5Turbo16k0613},
	}
}
