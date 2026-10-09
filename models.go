package crux

import (
	"errors"
	"strings"

	"crux.foo/internal/provider"
	"crux.foo/internal/schema"
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go/v3"
)

// Model names for every provider, and the registration that lets New infer
// the provider from them. Any other model works with WithProvider.

// ---------- openai ----------

type OpenAIModel = openai.ChatModel

const (
	OpenAIGPT6Astra                        = openai.ChatModelGPT6Astra
	OpenAIGPT5_6Sol                        = openai.ChatModelGPT5_6Sol
	OpenAIGPT5_6Terra                      = openai.ChatModelGPT5_6Terra
	OpenAIGPT5_6Luna                       = openai.ChatModelGPT5_6Luna
	OpenAIGPT5_5                           = openai.ChatModelGPT5_5
	OpenAIGPT5_5_2026_04_23                = openai.ChatModelGPT5_5_2026_04_23
	OpenAIGPT5_4                           = openai.ChatModelGPT5_4
	OpenAIGPT5_4Mini                       = openai.ChatModelGPT5_4Mini
	OpenAIGPT5_4Nano                       = openai.ChatModelGPT5_4Nano
	OpenAIGPT5_4Mini2026_03_17             = openai.ChatModelGPT5_4Mini2026_03_17
	OpenAIGPT5_4Nano2026_03_17             = openai.ChatModelGPT5_4Nano2026_03_17
	OpenAIGPT5_3ChatLatest                 = openai.ChatModelGPT5_3ChatLatest
	OpenAIGPT5_2                           = openai.ChatModelGPT5_2
	OpenAIGPT5_2_2025_12_11                = openai.ChatModelGPT5_2_2025_12_11
	OpenAIGPT5_2ChatLatest                 = openai.ChatModelGPT5_2ChatLatest
	OpenAIGPT5_2Pro                        = openai.ChatModelGPT5_2Pro
	OpenAIGPT5_2Pro2025_12_11              = openai.ChatModelGPT5_2Pro2025_12_11
	OpenAIGPT5_1                           = openai.ChatModelGPT5_1
	OpenAIGPT5_1_2025_11_13                = openai.ChatModelGPT5_1_2025_11_13
	OpenAIGPT5_1Codex                      = openai.ChatModelGPT5_1Codex
	OpenAIGPT5_1Mini                       = openai.ChatModelGPT5_1Mini
	OpenAIGPT5_1ChatLatest                 = openai.ChatModelGPT5_1ChatLatest
	OpenAIGPT5                             = openai.ChatModelGPT5
	OpenAIGPT5Mini                         = openai.ChatModelGPT5Mini
	OpenAIGPT5Nano                         = openai.ChatModelGPT5Nano
	OpenAIGPT5_2025_08_07                  = openai.ChatModelGPT5_2025_08_07
	OpenAIGPT5Mini2025_08_07               = openai.ChatModelGPT5Mini2025_08_07
	OpenAIGPT5Nano2025_08_07               = openai.ChatModelGPT5Nano2025_08_07
	OpenAIGPT5ChatLatest                   = openai.ChatModelGPT5ChatLatest
	OpenAIGPT4_1                           = openai.ChatModelGPT4_1
	OpenAIGPT4_1Mini                       = openai.ChatModelGPT4_1Mini
	OpenAIGPT4_1Nano                       = openai.ChatModelGPT4_1Nano
	OpenAIGPT4_1_2025_04_14                = openai.ChatModelGPT4_1_2025_04_14
	OpenAIGPT4_1Mini2025_04_14             = openai.ChatModelGPT4_1Mini2025_04_14
	OpenAIGPT4_1Nano2025_04_14             = openai.ChatModelGPT4_1Nano2025_04_14
	OpenAIO4Mini                           = openai.ChatModelO4Mini
	OpenAIO4Mini2025_04_16                 = openai.ChatModelO4Mini2025_04_16
	OpenAIO3                               = openai.ChatModelO3
	OpenAIO3_2025_04_16                    = openai.ChatModelO3_2025_04_16
	OpenAIO3Mini                           = openai.ChatModelO3Mini
	OpenAIO3Mini2025_01_31                 = openai.ChatModelO3Mini2025_01_31
	OpenAIO1                               = openai.ChatModelO1
	OpenAIO1_2024_12_17                    = openai.ChatModelO1_2024_12_17
	OpenAIO1Preview                        = openai.ChatModelO1Preview
	OpenAIO1Preview2024_09_12              = openai.ChatModelO1Preview2024_09_12
	OpenAIO1Mini                           = openai.ChatModelO1Mini
	OpenAIO1Mini2024_09_12                 = openai.ChatModelO1Mini2024_09_12
	OpenAIGPT4o                            = openai.ChatModelGPT4o
	OpenAIGPT4o2024_11_20                  = openai.ChatModelGPT4o2024_11_20
	OpenAIGPT4o2024_08_06                  = openai.ChatModelGPT4o2024_08_06
	OpenAIGPT4o2024_05_13                  = openai.ChatModelGPT4o2024_05_13
	OpenAIGPT4oAudioPreview                = openai.ChatModelGPT4oAudioPreview
	OpenAIGPT4oAudioPreview2024_10_01      = openai.ChatModelGPT4oAudioPreview2024_10_01
	OpenAIGPT4oAudioPreview2024_12_17      = openai.ChatModelGPT4oAudioPreview2024_12_17
	OpenAIGPT4oAudioPreview2025_06_03      = openai.ChatModelGPT4oAudioPreview2025_06_03
	OpenAIGPT4oMiniAudioPreview            = openai.ChatModelGPT4oMiniAudioPreview
	OpenAIGPT4oMiniAudioPreview2024_12_17  = openai.ChatModelGPT4oMiniAudioPreview2024_12_17
	OpenAIGPT4oSearchPreview               = openai.ChatModelGPT4oSearchPreview
	OpenAIGPT4oMiniSearchPreview           = openai.ChatModelGPT4oMiniSearchPreview
	OpenAIGPT4oSearchPreview2025_03_11     = openai.ChatModelGPT4oSearchPreview2025_03_11
	OpenAIGPT4oMiniSearchPreview2025_03_11 = openai.ChatModelGPT4oMiniSearchPreview2025_03_11
	OpenAIChatgpt4oLatest                  = openai.ChatModelChatgpt4oLatest
	OpenAICodexMiniLatest                  = openai.ChatModelCodexMiniLatest
	OpenAIGPT4oMini                        = openai.ChatModelGPT4oMini
	OpenAIGPT4oMini2024_07_18              = openai.ChatModelGPT4oMini2024_07_18
	OpenAIGPT4Turbo                        = openai.ChatModelGPT4Turbo
	OpenAIGPT4Turbo2024_04_09              = openai.ChatModelGPT4Turbo2024_04_09
	OpenAIGPT4_0125Preview                 = openai.ChatModelGPT4_0125Preview
	OpenAIGPT4TurboPreview                 = openai.ChatModelGPT4TurboPreview
	OpenAIGPT4_1106Preview                 = openai.ChatModelGPT4_1106Preview
	OpenAIGPT4VisionPreview                = openai.ChatModelGPT4VisionPreview
	OpenAIGPT4                             = openai.ChatModelGPT4
	OpenAIGPT4_0314                        = openai.ChatModelGPT4_0314
	OpenAIGPT4_0613                        = openai.ChatModelGPT4_0613
	OpenAIGPT4_32k                         = openai.ChatModelGPT4_32k
	OpenAIGPT4_32k0314                     = openai.ChatModelGPT4_32k0314
	OpenAIGPT4_32k0613                     = openai.ChatModelGPT4_32k0613
	OpenAIGPT3_5Turbo                      = openai.ChatModelGPT3_5Turbo
	OpenAIGPT3_5Turbo16k                   = openai.ChatModelGPT3_5Turbo16k
	OpenAIGPT3_5Turbo0301                  = openai.ChatModelGPT3_5Turbo0301
	OpenAIGPT3_5Turbo0613                  = openai.ChatModelGPT3_5Turbo0613
	OpenAIGPT3_5Turbo1106                  = openai.ChatModelGPT3_5Turbo1106
	OpenAIGPT3_5Turbo0125                  = openai.ChatModelGPT3_5Turbo0125
	OpenAIGPT3_5Turbo16k0613               = openai.ChatModelGPT3_5Turbo16k0613
)

// ---------- anthropic ----------

type AnthropicModel = anthropic.Model

// Models newer than the SDK's constants.
const (
	ClaudeOpus5_5   AnthropicModel = "claude-opus-5-5"
	ClaudeSonnet5_5 AnthropicModel = "claude-sonnet-5-5"
)

const (
	ClaudeFable5_1           = anthropic.ModelClaudeFable5_1
	ClaudeMythos5_1          = anthropic.ModelClaudeMythos5_1
	ClaudeSonnet5            = anthropic.ModelClaudeSonnet5
	ClaudeFable5             = anthropic.ModelClaudeFable5
	ClaudeMythos5            = anthropic.ModelClaudeMythos5
	ClaudeOpus5              = anthropic.ModelClaudeOpus5
	ClaudeOpus4_8            = anthropic.ModelClaudeOpus4_8
	ClaudeOpus4_7            = anthropic.ModelClaudeOpus4_7
	ClaudeOpus4_6            = anthropic.ModelClaudeOpus4_6
	ClaudeSonnet4_6          = anthropic.ModelClaudeSonnet4_6
	ClaudeHaiku4_5           = anthropic.ModelClaudeHaiku4_5
	ClaudeHaiku4_5_20251001  = anthropic.ModelClaudeHaiku4_5_20251001
	ClaudeOpus4_5            = anthropic.ModelClaudeOpus4_5
	ClaudeOpus4_5_20251101   = anthropic.ModelClaudeOpus4_5_20251101
	ClaudeSonnet4_5          = anthropic.ModelClaudeSonnet4_5
	ClaudeSonnet4_5_20250929 = anthropic.ModelClaudeSonnet4_5_20250929
)

// ---------- gemini ----------

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

// ---------- xai ----------

// Native xAI text model IDs and all aliases listed on their model pages.
// Verified against https://docs.x.ai/developers/models on 2026-09-12.
// Aliases may move to newer models; dated names are retained as documented.
const (
	XAIGrok4_6         = "grok-4.6"
	XAIGrok4_5         = "grok-4.5"
	XAIGrok4_5Latest   = "grok-4.5-latest"
	XAIGrokBuildLatest = "grok-build-latest"
	XAIGrok4_3         = "grok-4.3"
	XAIGrok4_3Latest   = "grok-4.3-latest"

	XAIGrok4_20_0309Reasoning                  = "grok-4.20-0309-reasoning"
	XAIGrok4_20ReasoningLatest                 = "grok-4.20-reasoning-latest"
	XAIGrok4_20                                = "grok-4.20"
	XAIGrok4_20Reasoning                       = "grok-4.20-reasoning"
	XAIGrok4_20_0309                           = "grok-4.20-0309"
	XAIGrok4_20Beta0309Reasoning               = "grok-4.20-beta-0309-reasoning"
	XAIGrok4_20Beta                            = "grok-4.20-beta"
	XAIGrok4_20Beta0309                        = "grok-4.20-beta-0309"
	XAIGrok4_20BetaLatest                      = "grok-4.20-beta-latest"
	XAIGrok4_20BetaLatestReasoning             = "grok-4.20-beta-latest-reasoning"
	XAIGrok4_20BetaReasoning                   = "grok-4.20-beta-reasoning"
	XAIGrok4_20ExperimentalBeta0304Reasoning   = "grok-4.20-experimental-beta-0304-reasoning"
	XAIGrok4_20ExperimentalBeta0304            = "grok-4.20-experimental-beta-0304"
	XAIGrok4_20ExperimentalBetaReasoningLatest = "grok-4.20-experimental-beta-reasoning-latest"
	XAIGrok4_20ExperimentalBetaLatest          = "grok-4.20-experimental-beta-latest"
	XAIGrok4_20ReasoningGV2                    = "grok-4.20-reasoning-gv2"

	XAIGrok4_20_0309NonReasoning                  = "grok-4.20-0309-non-reasoning"
	XAIGrok4_20NonReasoning                       = "grok-4.20-non-reasoning"
	XAIGrok4_20NonReasoningLatest                 = "grok-4.20-non-reasoning-latest"
	XAIGrok4_20BetaNonReasoning                   = "grok-4.20-beta-non-reasoning"
	XAIGrok4_20BetaLatestNonReasoning             = "grok-4.20-beta-latest-non-reasoning"
	XAIGrok4_20ExperimentalBeta0304NonReasoning   = "grok-4.20-experimental-beta-0304-non-reasoning"
	XAIGrok4_20ExperimentalBetaNonReasoningLatest = "grok-4.20-experimental-beta-non-reasoning-latest"
	XAIGrok4_20Beta0309NonReasoning               = "grok-4.20-beta-0309-non-reasoning"
	XAIGrok4_20NonReasoningGV2                    = "grok-4.20-non-reasoning-gv2"

	XAIGrok4_20MultiAgent0309                   = "grok-4.20-multi-agent-0309"
	XAIGrok4_20MultiAgent                       = "grok-4.20-multi-agent"
	XAIGrok4_20MultiAgentLatest                 = "grok-4.20-multi-agent-latest"
	XAIGrok4_20MultiAgentBetaLatest             = "grok-4.20-multi-agent-beta-latest"
	XAIGrok4_20MultiAgentExperimentalBeta0304   = "grok-4.20-multi-agent-experimental-beta-0304"
	XAIGrok4_20MultiAgentExperimentalBetaLatest = "grok-4.20-multi-agent-experimental-beta-latest"
	XAIGrok4_20MultiAgentBeta0309               = "grok-4.20-multi-agent-beta-0309"

	XAIGrokBuild0_1       = "grok-build-0.1"
	XAIGrokCodeFast1      = "grok-code-fast-1"
	XAIGrokCodeFast       = "grok-code-fast"
	XAIGrokCodeFast1_0825 = "grok-code-fast-1-0825"
)

// ---------- deepseek ----------

// Native DeepSeek model IDs documented for the Responses endpoint.
// Source: https://api-docs.deepseek.com/guides/responses_api
const (
	DeepSeekFlash = "deepseek-flash"
)

// ---------- openrouter ----------

// OpenRouter model IDs, including counterparts of the native provider exports
// and models from additional providers.
// Values are verified against https://openrouter.ai/api/v1/models.
// Only models with a catalog counterpart are included; native snapshot IDs
// are not redirected to rolling aliases. Model availability can change.
const (
	OpenRouterChatModelGPT6Astra           = "openai/gpt-6-astra"
	OpenRouterChatModelGPT5_6Sol           = "openai/gpt-5.6-sol"
	OpenRouterChatModelGPT5_6Terra         = "openai/gpt-5.6-terra"
	OpenRouterChatModelGPT5_6Luna          = "openai/gpt-5.6-luna"
	OpenRouterChatModelGPT5_5              = "openai/gpt-5.5"
	OpenRouterChatModelGPT5_4              = "openai/gpt-5.4"
	OpenRouterChatModelGPT5_4Mini          = "openai/gpt-5.4-mini"
	OpenRouterChatModelGPT5_4Nano          = "openai/gpt-5.4-nano"
	OpenRouterChatModelGPT5_2              = "openai/gpt-5.2"
	OpenRouterChatModelGPT5_2ChatLatest    = "openai/gpt-5.2-chat"
	OpenRouterChatModelGPT5_2Pro           = "openai/gpt-5.2-pro"
	OpenRouterChatModelGPT5_1              = "openai/gpt-5.1"
	OpenRouterChatModelGPT5_1Codex         = "openai/gpt-5.1-codex"
	OpenRouterChatModelGPT5                = "openai/gpt-5"
	OpenRouterChatModelGPT5Mini            = "openai/gpt-5-mini"
	OpenRouterChatModelGPT5Nano            = "openai/gpt-5-nano"
	OpenRouterChatModelGPT4_1              = "openai/gpt-4.1"
	OpenRouterChatModelGPT4_1Mini          = "openai/gpt-4.1-mini"
	OpenRouterChatModelGPT4_1Nano          = "openai/gpt-4.1-nano"
	OpenRouterChatModelO4Mini              = "openai/o4-mini"
	OpenRouterChatModelO3                  = "openai/o3"
	OpenRouterChatModelO3Mini              = "openai/o3-mini"
	OpenRouterChatModelO1                  = "openai/o1"
	OpenRouterChatModelGPT4o               = "openai/gpt-4o"
	OpenRouterChatModelGPT4o2024_11_20     = "openai/gpt-4o-2024-11-20"
	OpenRouterChatModelGPT4o2024_08_06     = "openai/gpt-4o-2024-08-06"
	OpenRouterChatModelGPT4o2024_05_13     = "openai/gpt-4o-2024-05-13"
	OpenRouterChatModelGPT4oMini           = "openai/gpt-4o-mini"
	OpenRouterChatModelGPT4oMini2024_07_18 = "openai/gpt-4o-mini-2024-07-18"
	OpenRouterChatModelGPT4Turbo           = "openai/gpt-4-turbo"
	OpenRouterChatModelGPT4TurboPreview    = "openai/gpt-4-turbo-preview"
	OpenRouterChatModelGPT4                = "openai/gpt-4"
	OpenRouterChatModelGPT3_5Turbo         = "openai/gpt-3.5-turbo"
	OpenRouterChatModelGPT3_5Turbo16k      = "openai/gpt-3.5-turbo-16k"
	OpenRouterChatModelGPT3_5Turbo0613     = "openai/gpt-3.5-turbo-0613"

	OpenRouterGemini3_8Flash          = "google/gemini-3.8-flash"
	OpenRouterGemini3_7Flash          = "google/gemini-3.7-flash"
	OpenRouterGemini3_6Flash          = "google/gemini-3.6-flash"
	OpenRouterGemini3_5Flash          = "google/gemini-3.5-flash"
	OpenRouterGemini3_5FlashLite      = "google/gemini-3.5-flash-lite"
	OpenRouterGemini3_1FlashLite      = "google/gemini-3.1-flash-lite"
	OpenRouterGemini3_1FlashImage     = "google/gemini-3.1-flash-image"
	OpenRouterGemini3_1FlashLiteImage = "google/gemini-3.1-flash-lite-image"
	OpenRouterGemini3ProImage         = "google/gemini-3-pro-image"
	OpenRouterGemini3_1ProPreview     = "google/gemini-3.1-pro-preview"
	OpenRouterGemini3FlashPreview     = "google/gemini-3-flash-preview"
	OpenRouterGemini2_5Pro            = "google/gemini-2.5-pro"
	OpenRouterGemini2_5Flash          = "google/gemini-2.5-flash"
	OpenRouterGemini2_5FlashLite      = "google/gemini-2.5-flash-lite"
	OpenRouterGemini2_5FlashImage     = "google/gemini-2.5-flash-image"

	OpenRouterAnthropicClaudeFable5_1  = "anthropic/claude-fable-5.1"
	OpenRouterAnthropicClaudeSonnet5   = "anthropic/claude-sonnet-5"
	OpenRouterAnthropicClaudeFable5    = "anthropic/claude-fable-5"
	OpenRouterAnthropicClaudeOpus5     = "anthropic/claude-opus-5"
	OpenRouterAnthropicClaudeOpus4_8   = "anthropic/claude-opus-4.8"
	OpenRouterAnthropicClaudeOpus4_7   = "anthropic/claude-opus-4.7"
	OpenRouterAnthropicClaudeOpus4_6   = "anthropic/claude-opus-4.6"
	OpenRouterAnthropicClaudeSonnet4_6 = "anthropic/claude-sonnet-4.6"
	OpenRouterAnthropicClaudeHaiku4_5  = "anthropic/claude-haiku-4.5"
	OpenRouterAnthropicClaudeOpus4_5   = "anthropic/claude-opus-4.5"
	OpenRouterAnthropicClaudeSonnet4_5 = "anthropic/claude-sonnet-4.5"

	OpenRouterDeepSeekV4_1Flash = "deepseek/deepseek-v4.1-flash"
	OpenRouterDeepSeekV4Pro     = "deepseek/deepseek-v4-pro"
	OpenRouterDeepSeekV4Flash   = "deepseek/deepseek-v4-flash"
	OpenRouterDeepSeekV3_2      = "deepseek/deepseek-v3.2"
	OpenRouterDeepSeekR1        = "deepseek/deepseek-r1"
	OpenRouterDeepSeekR1_0528   = "deepseek/deepseek-r1-0528"
	OpenRouterDeepSeekChat      = "deepseek/deepseek-chat"

	OpenRouterQwen3_8Max0902   = "qwen/qwen3.8-max-0902"
	OpenRouterQwen3_8Flash     = "qwen/qwen3.8-flash"
	OpenRouterQwen3_8_27B      = "qwen/qwen3.8-27b"
	OpenRouterQwen3_8_2_4TA95B = "qwen/qwen3.8-2.4t-a95b"
	OpenRouterQwen3_7Max       = "qwen/qwen3.7-max"
	OpenRouterQwen3_7Plus      = "qwen/qwen3.7-plus"
	OpenRouterQwen3_7Flash     = "qwen/qwen3.7-flash"
	OpenRouterQwen3Coder       = "qwen/qwen3-coder"
	OpenRouterQwen3CoderNext   = "qwen/qwen3-coder-next"
	OpenRouterQwen3CoderPlus   = "qwen/qwen3-coder-plus"
	OpenRouterQwen3CoderFlash  = "qwen/qwen3-coder-flash"

	OpenRouterMetaMuseSpark1_3            = "meta/muse-spark-1.3"
	OpenRouterMetaMuseSpark1_3Contributor = "meta/muse-spark-1.3-contributor"
	OpenRouterMetaMuseGlimmer30B          = "meta/muse-glimmer-30b"
	OpenRouterMetaLlama4Maverick          = "meta-llama/llama-4-maverick"
	OpenRouterMetaLlama4Scout             = "meta-llama/llama-4-scout"
	OpenRouterMetaLlama3_3_70BInstruct    = "meta-llama/llama-3.3-70b-instruct"
	OpenRouterMetaLlama3_1_8BInstruct     = "meta-llama/llama-3.1-8b-instruct"

	OpenRouterMistralMedium3_5        = "mistralai/mistral-medium-3-5"
	OpenRouterMistralSmall2603        = "mistralai/mistral-small-2603"
	OpenRouterMistralLarge2512        = "mistralai/mistral-large-2512"
	OpenRouterMistralDevstral2512     = "mistralai/devstral-2512"
	OpenRouterMistralCodestral2508    = "mistralai/codestral-2508"
	OpenRouterMistralMinistral14B2512 = "mistralai/ministral-14b-2512"
	OpenRouterMistralMinistral8B2512  = "mistralai/ministral-8b-2512"
	OpenRouterMistralMinistral3B2512  = "mistralai/ministral-3b-2512"

	OpenRouterXAIGrok4_6            = "x-ai/grok-4.6"
	OpenRouterXAIGrok4_5            = "x-ai/grok-4.5"
	OpenRouterXAIGrok4_3            = "x-ai/grok-4.3"
	OpenRouterXAIGrok4_20           = "x-ai/grok-4.20"
	OpenRouterXAIGrok4_20MultiAgent = "x-ai/grok-4.20-multi-agent"
	OpenRouterXAIGrokBuild0_1       = "x-ai/grok-build-0.1"

	OpenRouterMoonshotKimiK3         = "moonshotai/kimi-k3"
	OpenRouterMoonshotKimiK2_7Code   = "moonshotai/kimi-k2.7-code"
	OpenRouterMoonshotKimiK2_6       = "moonshotai/kimi-k2.6"
	OpenRouterMoonshotKimiK2_5       = "moonshotai/kimi-k2.5"
	OpenRouterMoonshotKimiK2Thinking = "moonshotai/kimi-k2-thinking"
	OpenRouterMiniMaxM3              = "minimax/minimax-m3"
	OpenRouterMiniMaxM2_7            = "minimax/minimax-m2.7"
	OpenRouterMiniMaxM2_5            = "minimax/minimax-m2.5"
	OpenRouterZAIGLM5_3              = "z-ai/glm-5.3"
	OpenRouterZAIGLM5_3Flash         = "z-ai/glm-5.3-flash"
	OpenRouterZAIGLM5_2              = "z-ai/glm-5.2"
	OpenRouterZAIGLM5_1              = "z-ai/glm-5.1"
	OpenRouterZAIGLM5VTurbo          = "z-ai/glm-5v-turbo"

	OpenRouterAmazonNova2LiteV1           = "amazon/nova-2-lite-v1"
	OpenRouterAmazonNovaPremierV1         = "amazon/nova-premier-v1"
	OpenRouterAmazonNovaProV1             = "amazon/nova-pro-v1"
	OpenRouterAmazonNovaLiteV1            = "amazon/nova-lite-v1"
	OpenRouterAmazonNovaMicroV1           = "amazon/nova-micro-v1"
	OpenRouterCohereCommandA              = "cohere/command-a"
	OpenRouterCohereCommandR08_2024       = "cohere/command-r-08-2024"
	OpenRouterCohereCommandRPlus08_2024   = "cohere/command-r-plus-08-2024"
	OpenRouterCohereNorthMiniCodeFree     = "cohere/north-mini-code:free"
	OpenRouterPerplexitySonar             = "perplexity/sonar"
	OpenRouterPerplexitySonarPro          = "perplexity/sonar-pro"
	OpenRouterPerplexitySonarProSearch    = "perplexity/sonar-pro-search"
	OpenRouterPerplexitySonarReasoningPro = "perplexity/sonar-reasoning-pro"
	OpenRouterPerplexitySonarDeepResearch = "perplexity/sonar-deep-research"

	OpenRouterNVIDIANemotron3_5Lightning   = "nvidia/nemotron-3.5-lightning"
	OpenRouterNVIDIANemotron3Ultra550BA55B = "nvidia/nemotron-3-ultra-550b-a55b"
	OpenRouterNVIDIANemotron3Super120BA12B = "nvidia/nemotron-3-super-120b-a12b"
	OpenRouterNVIDIANemotron3Nano30BA3B    = "nvidia/nemotron-3-nano-30b-a3b"
	OpenRouterMicrosoftPhi4                = "microsoft/phi-4"
	OpenRouterMicrosoftWizardLM2_8x22B     = "microsoft/wizardlm-2-8x22b"
	OpenRouterIBMGranite4_2_8B             = "ibm-granite/granite-4.2-8b"
	OpenRouterIBMGranite4_0HMicro          = "ibm-granite/granite-4.0-h-micro"
	OpenRouterTencentHY4Preview            = "tencent/hy4-preview"
	OpenRouterTencentHY3                   = "tencent/hy3"
	OpenRouterByteDanceSeed2_1Turbo        = "bytedance-seed/seed-2-1-turbo"
	OpenRouterByteDanceSeed2_0Code         = "bytedance-seed/seed-2.0-code"
	OpenRouterByteDanceSeed2_0Lite         = "bytedance-seed/seed-2.0-lite"
	OpenRouterByteDanceSeed2_0Mini         = "bytedance-seed/seed-2.0-mini"
	OpenRouterByteDanceUITars1_5_7B        = "bytedance/ui-tars-1.5-7b"

	OpenRouterSakanaFuguUltraV2                                   = "sakana/fugu-ultra-v2"
	OpenRouterSakanaFuguMax                                       = "sakana/fugu-max"
	OpenRouterInclusionAILing3_0Flash                             = "inclusionai/ling-3.0-flash"
	OpenRouterInclusionAILing3_0FlashVL                           = "inclusionai/ling-3.0-flash-vl"
	OpenRouterInceptionMercury2_5                                 = "inception/mercury-2.5"
	OpenRouterNexAGINexN2_5MiniFree                               = "nex-agi/nex-n2.5-mini:free"
	OpenRouterNexAGINexN2_5ProFree                                = "nex-agi/nex-n2.5-pro:free"
	OpenRouterDotsStudioDots3NotePreviewFree                      = "dots-studio/dots-3-note-preview:free"
	OpenRouterLiquidLFM2_5_2_6BFree                               = "liquid/lfm-2.5-2.6b:free"
	OpenRouterUpstageSolarPro4                                    = "upstage/solar-pro4"
	OpenRouterThinkingMachinesInkling                             = "thinkingmachines/inkling"
	OpenRouterThinkingMachinesInklingSmall                        = "thinkingmachines/inkling-small"
	OpenRouterPoolsideLagunaS2_1                                  = "poolside/laguna-s-2.1"
	OpenRouterPoolsideLagunaXS2_1                                 = "poolside/laguna-xs-2.1"
	OpenRouterMeituanLongCat2_0                                   = "meituan/longcat-2.0"
	OpenRouterKwaiPilotKatCoderProV2_5                            = "kwaipilot/kat-coder-pro-v2.5"
	OpenRouterAionLabsAion3_0                                     = "aion-labs/aion-3.0"
	OpenRouterAionLabsAion3_0Mini                                 = "aion-labs/aion-3.0-mini"
	OpenRouterStepFunStep3_7Flash                                 = "stepfun/step-3.7-flash"
	OpenRouterPerceptronMK1                                       = "perceptron/perceptron-mk1"
	OpenRouterXiaomiMiMoV2_5Pro                                   = "xiaomi/mimo-v2.5-pro"
	OpenRouterXiaomiMiMoV2_5                                      = "xiaomi/mimo-v2.5"
	OpenRouterArceeAITrinityLargeThinking                         = "arcee-ai/trinity-large-thinking"
	OpenRouterRekaAIEdge                                          = "rekaai/reka-edge"
	OpenRouterRekaAIFlash3                                        = "rekaai/reka-flash-3"
	OpenRouterWriterPalmyraX5                                     = "writer/palmyra-x5"
	OpenRouterRelaceSearch                                        = "relace/relace-search"
	OpenRouterRelaceApply3                                        = "relace/relace-apply-3"
	OpenRouterTheDrummerCydonia24BV4_1                            = "thedrummer/cydonia-24b-v4.1"
	OpenRouterNousResearchHermes4_405B                            = "nousresearch/hermes-4-405b"
	OpenRouterCognitiveComputationsDolphinMistral24BVeniceEdition = "cognitivecomputations/dolphin-mistral-24b-venice-edition"
	OpenRouterMorphV3Large                                        = "morph/morph-v3-large"
	OpenRouterMorphV3Fast                                         = "morph/morph-v3-fast"
	OpenRouterBaiduErnie4_5VL424BA47B                             = "baidu/ernie-4.5-vl-424b-a47b"
	OpenRouterSao10KL3_3Euryale70B                                = "sao10k/l3.3-euryale-70b"
	OpenRouterAnthraciteMagnumV4_72B                              = "anthracite-org/magnum-v4-72b"
	OpenRouterMancerWeaver                                        = "mancer/weaver"
	OpenRouterUndi95RemmSlerpL2_13B                               = "undi95/remm-slerp-l2-13b"
	OpenRouterGrypheMythoMaxL2_13B                                = "gryphe/mythomax-l2-13b"
	OpenRouterInferenceNetSchematronV2Turbo                       = "inference-net/schematron-v2-turbo"
	OpenRouterInferenceNetSchematronV2Small                       = "inference-net/schematron-v2-small"
)

// OpenRouter decision model IDs, for NewDecider. They answer typed questions
// through OpenRouter's Decisions API instead of generating text.
const (
	OpenRouterDecisionModelJev1_13   = "typesafe/jev-1.13"
	OpenRouterDecisionModelJevLatest = "typesafe/jev-latest"
)

// ---------- typesafe ----------

// TypeSafe decision model IDs, for NewDecider. Jev answers typed questions with
// calibrated probabilities and generates no text, so agents can't use it.
// Source: https://docs.typesafe.ai/api
const (
	Jev     = "jev-latest"
	Jev1_13 = "jev-1.13"
)

// ---------- ollama ----------

// Official Ollama generation families from https://ollama.com/library,
// checked on 2026-09-12. Each value follows its model page's default run command.
// Families without a latest tag use the published explicit tag instead.
// Embedding-only families are excluded. Tool and modality support varies by model.
// Local models must be pulled; cloud-tagged models require Ollama cloud access.
// Custom models and other tags work with WithProvider(ProviderOllama).
const (
	OllamaAlfred                 = "alfred"
	OllamaAtheneV2               = "athene-v2"
	OllamaAya                    = "aya"
	OllamaAyaExpanse             = "aya-expanse"
	OllamaBakllava               = "bakllava"
	OllamaBespokeMinicheck       = "bespoke-minicheck"
	OllamaCodebooga              = "codebooga"
	OllamaCodeGeeX4              = "codegeex4"
	OllamaCodeGemma              = "codegemma"
	OllamaCodeLlama              = "codellama"
	OllamaCodeQwen               = "codeqwen"
	OllamaCodestral              = "codestral"
	OllamaCodeup                 = "codeup"
	OllamaCogito                 = "cogito"
	OllamaCogito2_1              = "cogito-2.1"
	OllamaCommandA               = "command-a"
	OllamaCommandR               = "command-r"
	OllamaCommandR7B             = "command-r7b"
	OllamaCommandR7BArabic       = "command-r7b-arabic"
	OllamaCommandRPlus           = "command-r-plus"
	OllamaDBRX                   = "dbrx"
	OllamaDeepCoder              = "deepcoder"
	OllamaDeepScaler             = "deepscaler"
	OllamaDeepSeekCoder          = "deepseek-coder"
	OllamaDeepSeekCoderV2        = "deepseek-coder-v2"
	OllamaDeepSeekLLM            = "deepseek-llm"
	OllamaDeepSeekOCR            = "deepseek-ocr"
	OllamaDeepSeekR1             = "deepseek-r1"
	OllamaDeepSeekV2             = "deepseek-v2"
	OllamaDeepSeekV2_5           = "deepseek-v2.5"
	OllamaDeepSeekV3             = "deepseek-v3"
	OllamaDeepSeekV3_1           = "deepseek-v3.1"
	OllamaDeepSeekV4_1Flash      = "deepseek-v4.1-flash:cloud"
	OllamaDeepSeekV4Flash        = "deepseek-v4-flash:cloud"
	OllamaDeepSeekV4Pro          = "deepseek-v4-pro:cloud"
	OllamaDevstral               = "devstral"
	OllamaDevstral2              = "devstral-2"
	OllamaDevstralSmall2         = "devstral-small-2"
	OllamaDolphin3               = "dolphin3"
	OllamaDolphinCoder           = "dolphincoder"
	OllamaDolphinLlama3          = "dolphin-llama3"
	OllamaDolphinMistral         = "dolphin-mistral"
	OllamaDolphinMixtral         = "dolphin-mixtral"
	OllamaDolphinPhi             = "dolphin-phi"
	OllamaDuckDBNSQL             = "duckdb-nsql"
	OllamaEverythingLM           = "everythinglm"
	OllamaExaone3_5              = "exaone3.5"
	OllamaExaoneDeep             = "exaone-deep"
	OllamaFalcon                 = "falcon"
	OllamaFalcon2                = "falcon2"
	OllamaFalcon3                = "falcon3"
	OllamaFireFunctionV2         = "firefunction-v2"
	OllamaFunctionGemma          = "functiongemma"
	OllamaGemma                  = "gemma"
	OllamaGemma2                 = "gemma2"
	OllamaGemma3                 = "gemma3"
	OllamaGemma3n                = "gemma3n"
	OllamaGemma4                 = "gemma4"
	OllamaGLM4                   = "glm4"
	OllamaGLM4_7Flash            = "glm-4.7-flash"
	OllamaGLM5_1                 = "glm-5.1:cloud"
	OllamaGLM5_2                 = "glm-5.2:cloud"
	OllamaGLM5_3                 = "glm-5.3:cloud"
	OllamaGLM5_3Flash            = "glm-5.3-flash:cloud"
	OllamaGLMOCR                 = "glm-ocr"
	OllamaGoliath                = "goliath"
	OllamaGPTOSS                 = "gpt-oss"
	OllamaGPTOSSSafeguard        = "gpt-oss-safeguard"
	OllamaGranite3_1Dense        = "granite3.1-dense"
	OllamaGranite3_1MoE          = "granite3.1-moe"
	OllamaGranite3_2             = "granite3.2"
	OllamaGranite3_2Vision       = "granite3.2-vision"
	OllamaGranite3_3             = "granite3.3"
	OllamaGranite3Dense          = "granite3-dense"
	OllamaGranite3Guardian       = "granite3-guardian"
	OllamaGranite3MoE            = "granite3-moe"
	OllamaGranite4               = "granite4"
	OllamaGranite4_1             = "granite4.1:3b"
	OllamaGranite4_1Guardian     = "granite4.1-guardian:8b"
	OllamaGranite4_2             = "granite4.2"
	OllamaGraniteCode            = "granite-code"
	OllamaHermes3                = "hermes3"
	OllamaInternLM2              = "internlm2"
	OllamaKimiK2_6               = "kimi-k2.6:cloud"
	OllamaKimiK2_7Code           = "kimi-k2.7-code:cloud"
	OllamaKimiK3                 = "kimi-k3:cloud"
	OllamaLagunaS2_1             = "laguna-s-2.1"
	OllamaLagunaXS_2             = "laguna-xs.2"
	OllamaLagunaXS2_1            = "laguna-xs-2.1"
	OllamaLFM2                   = "lfm2"
	OllamaLFM2_5                 = "lfm2.5"
	OllamaLFM2_5Thinking         = "lfm2.5-thinking"
	OllamaLlama2                 = "llama2"
	OllamaLlama2Chinese          = "llama2-chinese"
	OllamaLlama2Uncensored       = "llama2-uncensored"
	OllamaLlama3                 = "llama3"
	OllamaLlama3_1               = "llama3.1"
	OllamaLlama3_2               = "llama3.2"
	OllamaLlama3_2Vision         = "llama3.2-vision"
	OllamaLlama3_3               = "llama3.3"
	OllamaLlama3ChatQA           = "llama3-chatqa"
	OllamaLlama3Gradient         = "llama3-gradient"
	OllamaLlama3GroqToolUse      = "llama3-groq-tool-use"
	OllamaLlama4                 = "llama4"
	OllamaLlamaGuard3            = "llama-guard3"
	OllamaLlamaPro               = "llama-pro"
	OllamaLlava                  = "llava"
	OllamaLlavaLlama3            = "llava-llama3"
	OllamaLlavaPhi3              = "llava-phi3"
	OllamaMagicoder              = "magicoder"
	OllamaMagistral              = "magistral"
	OllamaMarcoO1                = "marco-o1"
	OllamaMathstral              = "mathstral"
	OllamaMedGemma               = "medgemma"
	OllamaMedGemma1_5            = "medgemma1.5"
	OllamaMeditron               = "meditron"
	OllamaMedLlama2              = "medllama2"
	OllamaMegadolphin            = "megadolphin"
	OllamaMiniCPMV               = "minicpm-v"
	OllamaMiniCPMV4_5            = "minicpm-v4.5"
	OllamaMiniCPMV4_6            = "minicpm-v4.6"
	OllamaMiniMaxM2_7            = "minimax-m2.7:cloud"
	OllamaMiniMaxM3              = "minimax-m3:cloud"
	OllamaMinistral3             = "ministral-3"
	OllamaMistral                = "mistral"
	OllamaMistralLarge           = "mistral-large"
	OllamaMistralLarge3          = "mistral-large-3:675b-cloud"
	OllamaMistrallite            = "mistrallite"
	OllamaMistralMedium3_5       = "mistral-medium-3.5"
	OllamaMistralNemo            = "mistral-nemo"
	OllamaMistralOpenOrca        = "mistral-openorca"
	OllamaMistralSmall           = "mistral-small"
	OllamaMistralSmall3_1        = "mistral-small3.1"
	OllamaMistralSmall3_2        = "mistral-small3.2"
	OllamaMixtral                = "mixtral"
	OllamaMoondream              = "moondream"
	OllamaMuseGlimmer            = "muse-glimmer"
	OllamaNemotron               = "nemotron"
	OllamaNemotron3              = "nemotron3:33b"
	OllamaNemotron3_5Lightning   = "nemotron-3.5-lightning"
	OllamaNemotron3Nano          = "nemotron-3-nano"
	OllamaNemotron3Super         = "nemotron-3-super"
	OllamaNemotron3Ultra         = "nemotron-3-ultra:cloud"
	OllamaNemotronCascade2       = "nemotron-cascade-2"
	OllamaNemotronMini           = "nemotron-mini"
	OllamaNeuralChat             = "neural-chat"
	OllamaNexusRaven             = "nexusraven"
	OllamaNorthMiniCode1_0       = "north-mini-code-1.0"
	OllamaNotus                  = "notus"
	OllamaNotux                  = "notux"
	OllamaNousHermes             = "nous-hermes"
	OllamaNousHermes2            = "nous-hermes2"
	OllamaNousHermes2Mixtral     = "nous-hermes2-mixtral"
	OllamaNuExtract              = "nuextract"
	OllamaOLMo2                  = "olmo2"
	OllamaOLMo3                  = "olmo-3"
	OllamaOLMo3_1                = "olmo-3.1"
	OllamaOpenChat               = "openchat"
	OllamaOpenCoder              = "opencoder"
	OllamaOpenHermes             = "openhermes"
	OllamaOpenOrcaPlatypus2      = "open-orca-platypus2"
	OllamaOpenThinker            = "openthinker"
	OllamaOrca2                  = "orca2"
	OllamaOrcaMini               = "orca-mini"
	OllamaOrnith                 = "ornith"
	OllamaOrnith1_5              = "ornith-1.5:9b"
	OllamaPhi                    = "phi"
	OllamaPhi3                   = "phi3"
	OllamaPhi3_5                 = "phi3.5"
	OllamaPhi4                   = "phi4"
	OllamaPhi4Mini               = "phi4-mini"
	OllamaPhi4MiniReasoning      = "phi4-mini-reasoning"
	OllamaPhi4Reasoning          = "phi4-reasoning"
	OllamaPhindCodeLlama         = "phind-codellama"
	OllamaQwen                   = "qwen"
	OllamaQwen2                  = "qwen2"
	OllamaQwen2_5                = "qwen2.5"
	OllamaQwen2_5Coder           = "qwen2.5-coder"
	OllamaQwen2_5VL              = "qwen2.5vl"
	OllamaQwen2Math              = "qwen2-math"
	OllamaQwen3                  = "qwen3"
	OllamaQwen3_5                = "qwen3.5"
	OllamaQwen3_6                = "qwen3.6"
	OllamaQwen3_8                = "qwen3.8"
	OllamaQwen3_8FlashNext       = "qwen3.8-flash-next:125b-mlx"
	OllamaQwen3Coder             = "qwen3-coder"
	OllamaQwen3CoderNext         = "qwen3-coder-next"
	OllamaQwen3Next              = "qwen3-next"
	OllamaQwen3VL                = "qwen3-vl"
	OllamaQwQ                    = "qwq"
	OllamaR1_1776                = "r1-1776"
	OllamaReaderLM               = "reader-lm"
	OllamaReflection             = "reflection"
	OllamaRNJ1                   = "rnj-1"
	OllamaSailor2                = "sailor2"
	OllamaSamanthaMistral        = "samantha-mistral"
	OllamaShieldGemma            = "shieldgemma"
	OllamaSmallThinker           = "smallthinker"
	OllamaSmolLM                 = "smollm"
	OllamaSmolLM2                = "smollm2"
	OllamaSolar                  = "solar"
	OllamaSolarPro               = "solar-pro"
	OllamaSQLCoder               = "sqlcoder"
	OllamaStableBeluga           = "stable-beluga"
	OllamaStableCode             = "stable-code"
	OllamaStableLM2              = "stablelm2"
	OllamaStableLMZephyr         = "stablelm-zephyr"
	OllamaStarCoder              = "starcoder"
	OllamaStarCoder2             = "starcoder2"
	OllamaStarlingLM             = "starling-lm"
	OllamaTinyDolphin            = "tinydolphin"
	OllamaTinyLlama              = "tinyllama"
	OllamaTranslateGemma         = "translategemma"
	OllamaTulu3                  = "tulu3"
	OllamaVicuna                 = "vicuna"
	OllamaWizardCoder            = "wizardcoder"
	OllamaWizardLM               = "wizardlm:7b-q2_K"
	OllamaWizardLM2              = "wizardlm2"
	OllamaWizardLMUncensored     = "wizardlm-uncensored"
	OllamaWizardMath             = "wizard-math"
	OllamaWizardVicuna           = "wizard-vicuna"
	OllamaWizardVicunaUncensored = "wizard-vicuna-uncensored"
	OllamaXWinLM                 = "xwinlm"
	OllamaYarnLlama2             = "yarn-llama2"
	OllamaYarnMistral            = "yarn-mistral"
	OllamaYi                     = "yi"
	OllamaYiCoder                = "yi-coder"
	OllamaZephyr                 = "zephyr"

	// Existing size-specific exports.
	OllamaQwen3_8B  = "qwen3:8b"
	OllamaGPTOSS20B = "gpt-oss:20b"
)

func init() {
	registerProvider(ProviderOpenAI, providerSpec{
		models: []string{
			OpenAIGPT6Astra,
			OpenAIGPT5_6Sol,
			OpenAIGPT5_6Terra,
			OpenAIGPT5_6Luna,
			OpenAIGPT5_5,
			OpenAIGPT5_5_2026_04_23,
			OpenAIGPT5_4,
			OpenAIGPT5_4Mini,
			OpenAIGPT5_4Nano,
			OpenAIGPT5_4Mini2026_03_17,
			OpenAIGPT5_4Nano2026_03_17,
			OpenAIGPT5_3ChatLatest,
			OpenAIGPT5_2,
			OpenAIGPT5_2_2025_12_11,
			OpenAIGPT5_2ChatLatest,
			OpenAIGPT5_2Pro,
			OpenAIGPT5_2Pro2025_12_11,
			OpenAIGPT5_1,
			OpenAIGPT5_1_2025_11_13,
			OpenAIGPT5_1Codex,
			OpenAIGPT5_1Mini,
			OpenAIGPT5_1ChatLatest,
			OpenAIGPT5,
			OpenAIGPT5Mini,
			OpenAIGPT5Nano,
			OpenAIGPT5_2025_08_07,
			OpenAIGPT5Mini2025_08_07,
			OpenAIGPT5Nano2025_08_07,
			OpenAIGPT5ChatLatest,
			OpenAIGPT4_1,
			OpenAIGPT4_1Mini,
			OpenAIGPT4_1Nano,
			OpenAIGPT4_1_2025_04_14,
			OpenAIGPT4_1Mini2025_04_14,
			OpenAIGPT4_1Nano2025_04_14,
			OpenAIO4Mini,
			OpenAIO4Mini2025_04_16,
			OpenAIO3,
			OpenAIO3_2025_04_16,
			OpenAIO3Mini,
			OpenAIO3Mini2025_01_31,
			OpenAIO1,
			OpenAIO1_2024_12_17,
			OpenAIO1Preview,
			OpenAIO1Preview2024_09_12,
			OpenAIO1Mini,
			OpenAIO1Mini2024_09_12,
			OpenAIGPT4o,
			OpenAIGPT4o2024_11_20,
			OpenAIGPT4o2024_08_06,
			OpenAIGPT4o2024_05_13,
			OpenAIGPT4oAudioPreview,
			OpenAIGPT4oAudioPreview2024_10_01,
			OpenAIGPT4oAudioPreview2024_12_17,
			OpenAIGPT4oAudioPreview2025_06_03,
			OpenAIGPT4oMiniAudioPreview,
			OpenAIGPT4oMiniAudioPreview2024_12_17,
			OpenAIGPT4oSearchPreview,
			OpenAIGPT4oMiniSearchPreview,
			OpenAIGPT4oSearchPreview2025_03_11,
			OpenAIGPT4oMiniSearchPreview2025_03_11,
			OpenAIChatgpt4oLatest,
			OpenAICodexMiniLatest,
			OpenAIGPT4oMini,
			OpenAIGPT4oMini2024_07_18,
			OpenAIGPT4Turbo,
			OpenAIGPT4Turbo2024_04_09,
			OpenAIGPT4_0125Preview,
			OpenAIGPT4TurboPreview,
			OpenAIGPT4_1106Preview,
			OpenAIGPT4VisionPreview,
			OpenAIGPT4,
			OpenAIGPT4_0314,
			OpenAIGPT4_0613,
			OpenAIGPT4_32k,
			OpenAIGPT4_32k0314,
			OpenAIGPT4_32k0613,
			OpenAIGPT3_5Turbo,
			OpenAIGPT3_5Turbo16k,
			OpenAIGPT3_5Turbo0301,
			OpenAIGPT3_5Turbo0613,
			OpenAIGPT3_5Turbo1106,
			OpenAIGPT3_5Turbo0125,
			OpenAIGPT3_5Turbo16k0613,
		},
		envVars:       []string{"OPENAI_API_KEY", "OPENAI_APIKEY", "OPENAI_KEY"},
		step:          provider.OpenAI,
		schema:        schema.AdaptOpenAI,
		contextWindow: openAIWindow,
	})

	registerProvider(ProviderAnthropic, providerSpec{
		models: []string{
			ClaudeOpus5_5, ClaudeSonnet5_5,
			ClaudeFable5_1, ClaudeMythos5_1,
			ClaudeSonnet5, ClaudeFable5,
			ClaudeMythos5, ClaudeOpus5,
			ClaudeOpus4_8, ClaudeOpus4_7,
			ClaudeOpus4_6,
			ClaudeSonnet4_6, ClaudeHaiku4_5,
			ClaudeHaiku4_5_20251001, ClaudeOpus4_5,
			ClaudeOpus4_5_20251101, ClaudeSonnet4_5,
			ClaudeSonnet4_5_20250929,
		},
		// ANTHROPIC_AUTH_TOKEN is a bearer token, not an API key; the SDK's
		// default options send it as Authorization when no key is set.
		// https://github.com/anthropics/anthropic-sdk-go/blob/v1.72.0/client.go
		envVars:       []string{"ANTHROPIC_API_KEY", "ANTHROPIC_APIKEY", "ANTHROPIC_KEY"},
		step:          provider.Anthropic,
		schema:        schema.AdaptAnthropic,
		contextWindow: anthropicWindow,
		prepare:       prepareAnthropic,
	})

	registerProvider(ProviderGoogle, providerSpec{
		models: []string{
			Gemini3_8Flash, Gemini3_7Flash,
			Gemini3_6Flash, Gemini3_5Flash,
			Gemini3_5FlashLite, Gemini3_1FlashLite,
			Gemini3_1FlashImage, Gemini3_1FlashLiteImage,
			Gemini3ProImage, Gemini3_1ProPreview,
			Gemini3FlashPreview, Gemini3_5LiveTranslatePreview,
			Gemini3_1FlashLivePreview, Gemini3_1FlashTTSPreview,
			GeminiOmni1_1Flash, Gemini3_5Transcribe,
			Gemini3_5TranscribeLive,
			Gemini2_5Pro, Gemini2_5Flash, Gemini2_5FlashLite,
			Gemini2_5FlashImage, Gemini2_5FlashNativeAudioPreview12_2025,
			Gemini2_5FlashPreviewTTS, Gemini2_5ProPreviewTTS,
			Gemini2_5ComputerUsePreview10_2025,
			GeminiEmbedding2Preview, GeminiEmbedding001,
			GeminiRoboticsER2Preview, GeminiRoboticsER1_6Preview,
		},
		envVars:       []string{"GOOGLE_API_KEY", "GOOGLE_APIKEY", "GOOGLE_KEY", "GEMINI_API_KEY", "GEMINI_APIKEY", "GEMINI_KEY"},
		step:          provider.Gemini,
		prepare:       prepareGoogle,
		schema:        schema.AdaptPermissive,
		contextWindow: geminiWindow,
	})

	registerProvider(ProviderXAI, providerSpec{
		models: []string{
			XAIGrok4_6,
			XAIGrok4_5,
			XAIGrok4_5Latest,
			XAIGrokBuildLatest,
			XAIGrok4_3,
			XAIGrok4_3Latest,
			XAIGrok4_20_0309Reasoning,
			XAIGrok4_20ReasoningLatest,
			XAIGrok4_20,
			XAIGrok4_20Reasoning,
			XAIGrok4_20_0309,
			XAIGrok4_20Beta0309Reasoning,
			XAIGrok4_20Beta,
			XAIGrok4_20Beta0309,
			XAIGrok4_20BetaLatest,
			XAIGrok4_20BetaLatestReasoning,
			XAIGrok4_20BetaReasoning,
			XAIGrok4_20ExperimentalBeta0304Reasoning,
			XAIGrok4_20ExperimentalBeta0304,
			XAIGrok4_20ExperimentalBetaReasoningLatest,
			XAIGrok4_20ExperimentalBetaLatest,
			XAIGrok4_20ReasoningGV2,
			XAIGrok4_20_0309NonReasoning,
			XAIGrok4_20NonReasoning,
			XAIGrok4_20NonReasoningLatest,
			XAIGrok4_20BetaNonReasoning,
			XAIGrok4_20BetaLatestNonReasoning,
			XAIGrok4_20ExperimentalBeta0304NonReasoning,
			XAIGrok4_20ExperimentalBetaNonReasoningLatest,
			XAIGrok4_20Beta0309NonReasoning,
			XAIGrok4_20NonReasoningGV2,
			XAIGrok4_20MultiAgent0309,
			XAIGrok4_20MultiAgent,
			XAIGrok4_20MultiAgentLatest,
			XAIGrok4_20MultiAgentBetaLatest,
			XAIGrok4_20MultiAgentExperimentalBeta0304,
			XAIGrok4_20MultiAgentExperimentalBetaLatest,
			XAIGrok4_20MultiAgentBeta0309,
			XAIGrokBuild0_1,
			XAIGrokCodeFast1,
			XAIGrokCodeFast,
			XAIGrokCodeFast1_0825,
		},
		envVars:       []string{"XAI_API_KEY", "XAI_APIKEY", "XAI_KEY"},
		baseURL:       "https://api.x.ai/v1",
		step:          provider.OpenAI,
		schema:        schema.AdaptOpenAI,
		contextWindow: xAIWindow,
	})

	registerProvider(ProviderDeepSeek, providerSpec{
		models: []string{
			DeepSeekFlash,
		},
		envVars:       []string{"DEEPSEEK_API_KEY", "DEEPSEEK_APIKEY", "DEEPSEEK_KEY"},
		baseURL:       "https://api.deepseek.com",
		step:          provider.OpenAI, // stateless Responses with plain-text reasoning replay
		schema:        schema.AdaptOpenAI,
		contextWindow: deepSeekWindow,
		prepare:       prepareDeepSeek,
	})

	registerProvider(ProviderOpenrouter, providerSpec{
		models: []string{
			OpenRouterChatModelGPT6Astra,
			OpenRouterChatModelGPT5_6Sol,
			OpenRouterChatModelGPT5_6Terra,
			OpenRouterChatModelGPT5_6Luna,
			OpenRouterChatModelGPT5_5,
			OpenRouterChatModelGPT5_4,
			OpenRouterChatModelGPT5_4Mini,
			OpenRouterChatModelGPT5_4Nano,
			OpenRouterChatModelGPT5_2,
			OpenRouterChatModelGPT5_2ChatLatest,
			OpenRouterChatModelGPT5_2Pro,
			OpenRouterChatModelGPT5_1,
			OpenRouterChatModelGPT5_1Codex,
			OpenRouterChatModelGPT5,
			OpenRouterChatModelGPT5Mini,
			OpenRouterChatModelGPT5Nano,
			OpenRouterChatModelGPT4_1,
			OpenRouterChatModelGPT4_1Mini,
			OpenRouterChatModelGPT4_1Nano,
			OpenRouterChatModelO4Mini,
			OpenRouterChatModelO3,
			OpenRouterChatModelO3Mini,
			OpenRouterChatModelO1,
			OpenRouterChatModelGPT4o,
			OpenRouterChatModelGPT4o2024_11_20,
			OpenRouterChatModelGPT4o2024_08_06,
			OpenRouterChatModelGPT4o2024_05_13,
			OpenRouterChatModelGPT4oMini,
			OpenRouterChatModelGPT4oMini2024_07_18,
			OpenRouterChatModelGPT4Turbo,
			OpenRouterChatModelGPT4TurboPreview,
			OpenRouterChatModelGPT4,
			OpenRouterChatModelGPT3_5Turbo,
			OpenRouterChatModelGPT3_5Turbo16k,
			OpenRouterChatModelGPT3_5Turbo0613,
			OpenRouterGemini3_8Flash,
			OpenRouterGemini3_7Flash,
			OpenRouterGemini3_6Flash,
			OpenRouterGemini3_5Flash,
			OpenRouterGemini3_5FlashLite,
			OpenRouterGemini3_1FlashLite,
			OpenRouterGemini3_1FlashImage,
			OpenRouterGemini3_1FlashLiteImage,
			OpenRouterGemini3ProImage,
			OpenRouterGemini3_1ProPreview,
			OpenRouterGemini3FlashPreview,
			OpenRouterGemini2_5Pro,
			OpenRouterGemini2_5Flash,
			OpenRouterGemini2_5FlashLite,
			OpenRouterGemini2_5FlashImage,
			OpenRouterAnthropicClaudeFable5_1,
			OpenRouterAnthropicClaudeSonnet5,
			OpenRouterAnthropicClaudeFable5,
			OpenRouterAnthropicClaudeOpus5,
			OpenRouterAnthropicClaudeOpus4_8,
			OpenRouterAnthropicClaudeOpus4_7,
			OpenRouterAnthropicClaudeOpus4_6,
			OpenRouterAnthropicClaudeSonnet4_6,
			OpenRouterAnthropicClaudeHaiku4_5,
			OpenRouterAnthropicClaudeOpus4_5,
			OpenRouterAnthropicClaudeSonnet4_5,
			OpenRouterDeepSeekV4_1Flash,
			OpenRouterDeepSeekV4Pro,
			OpenRouterDeepSeekV4Flash,
			OpenRouterDeepSeekV3_2,
			OpenRouterDeepSeekR1,
			OpenRouterDeepSeekR1_0528,
			OpenRouterDeepSeekChat,
			OpenRouterQwen3_8Max0902,
			OpenRouterQwen3_8Flash,
			OpenRouterQwen3_8_27B,
			OpenRouterQwen3_8_2_4TA95B,
			OpenRouterQwen3_7Max,
			OpenRouterQwen3_7Plus,
			OpenRouterQwen3_7Flash,
			OpenRouterQwen3Coder,
			OpenRouterQwen3CoderNext,
			OpenRouterQwen3CoderPlus,
			OpenRouterQwen3CoderFlash,
			OpenRouterMetaMuseSpark1_3,
			OpenRouterMetaMuseSpark1_3Contributor,
			OpenRouterMetaMuseGlimmer30B,
			OpenRouterMetaLlama4Maverick,
			OpenRouterMetaLlama4Scout,
			OpenRouterMetaLlama3_3_70BInstruct,
			OpenRouterMetaLlama3_1_8BInstruct,
			OpenRouterMistralMedium3_5,
			OpenRouterMistralSmall2603,
			OpenRouterMistralLarge2512,
			OpenRouterMistralDevstral2512,
			OpenRouterMistralCodestral2508,
			OpenRouterMistralMinistral14B2512,
			OpenRouterMistralMinistral8B2512,
			OpenRouterMistralMinistral3B2512,
			OpenRouterXAIGrok4_6,
			OpenRouterXAIGrok4_5,
			OpenRouterXAIGrok4_3,
			OpenRouterXAIGrok4_20,
			OpenRouterXAIGrok4_20MultiAgent,
			OpenRouterXAIGrokBuild0_1,
			OpenRouterMoonshotKimiK3,
			OpenRouterMoonshotKimiK2_7Code,
			OpenRouterMoonshotKimiK2_6,
			OpenRouterMoonshotKimiK2_5,
			OpenRouterMoonshotKimiK2Thinking,
			OpenRouterMiniMaxM3,
			OpenRouterMiniMaxM2_7,
			OpenRouterMiniMaxM2_5,
			OpenRouterZAIGLM5_3,
			OpenRouterZAIGLM5_3Flash,
			OpenRouterZAIGLM5_2,
			OpenRouterZAIGLM5_1,
			OpenRouterZAIGLM5VTurbo,
			OpenRouterAmazonNova2LiteV1,
			OpenRouterAmazonNovaPremierV1,
			OpenRouterAmazonNovaProV1,
			OpenRouterAmazonNovaLiteV1,
			OpenRouterAmazonNovaMicroV1,
			OpenRouterCohereCommandA,
			OpenRouterCohereCommandR08_2024,
			OpenRouterCohereCommandRPlus08_2024,
			OpenRouterCohereNorthMiniCodeFree,
			OpenRouterPerplexitySonar,
			OpenRouterPerplexitySonarPro,
			OpenRouterPerplexitySonarProSearch,
			OpenRouterPerplexitySonarReasoningPro,
			OpenRouterPerplexitySonarDeepResearch,
			OpenRouterNVIDIANemotron3_5Lightning,
			OpenRouterNVIDIANemotron3Ultra550BA55B,
			OpenRouterNVIDIANemotron3Super120BA12B,
			OpenRouterNVIDIANemotron3Nano30BA3B,
			OpenRouterMicrosoftPhi4,
			OpenRouterMicrosoftWizardLM2_8x22B,
			OpenRouterIBMGranite4_2_8B,
			OpenRouterIBMGranite4_0HMicro,
			OpenRouterTencentHY4Preview,
			OpenRouterTencentHY3,
			OpenRouterByteDanceSeed2_1Turbo,
			OpenRouterByteDanceSeed2_0Code,
			OpenRouterByteDanceSeed2_0Lite,
			OpenRouterByteDanceSeed2_0Mini,
			OpenRouterByteDanceUITars1_5_7B,
			OpenRouterSakanaFuguUltraV2,
			OpenRouterSakanaFuguMax,
			OpenRouterInclusionAILing3_0Flash,
			OpenRouterInclusionAILing3_0FlashVL,
			OpenRouterInceptionMercury2_5,
			OpenRouterNexAGINexN2_5MiniFree,
			OpenRouterNexAGINexN2_5ProFree,
			OpenRouterDotsStudioDots3NotePreviewFree,
			OpenRouterLiquidLFM2_5_2_6BFree,
			OpenRouterUpstageSolarPro4,
			OpenRouterThinkingMachinesInkling,
			OpenRouterThinkingMachinesInklingSmall,
			OpenRouterPoolsideLagunaS2_1,
			OpenRouterPoolsideLagunaXS2_1,
			OpenRouterMeituanLongCat2_0,
			OpenRouterKwaiPilotKatCoderProV2_5,
			OpenRouterAionLabsAion3_0,
			OpenRouterAionLabsAion3_0Mini,
			OpenRouterStepFunStep3_7Flash,
			OpenRouterPerceptronMK1,
			OpenRouterXiaomiMiMoV2_5Pro,
			OpenRouterXiaomiMiMoV2_5,
			OpenRouterArceeAITrinityLargeThinking,
			OpenRouterRekaAIEdge,
			OpenRouterRekaAIFlash3,
			OpenRouterWriterPalmyraX5,
			OpenRouterRelaceSearch,
			OpenRouterRelaceApply3,
			OpenRouterTheDrummerCydonia24BV4_1,
			OpenRouterNousResearchHermes4_405B,
			OpenRouterCognitiveComputationsDolphinMistral24BVeniceEdition,
			OpenRouterMorphV3Large,
			OpenRouterMorphV3Fast,
			OpenRouterBaiduErnie4_5VL424BA47B,
			OpenRouterSao10KL3_3Euryale70B,
			OpenRouterAnthraciteMagnumV4_72B,
			OpenRouterMancerWeaver,
			OpenRouterUndi95RemmSlerpL2_13B,
			OpenRouterGrypheMythoMaxL2_13B,
			OpenRouterInferenceNetSchematronV2Turbo,
			OpenRouterInferenceNetSchematronV2Small,
			OpenRouterDecisionModelJev1_13,
			OpenRouterDecisionModelJevLatest,
		},
		envVars:       []string{"OPENROUTER_API_KEY", "OPENROUTER_APIKEY", "OPENROUTER_KEY"},
		baseURL:       "https://openrouter.ai/api/v1",
		step:          provider.OpenAI,
		schema:        schema.AdaptOpenAI,
		contextWindow: openRouterWindow,
		prepare:       prepareOpenRouter,
		decide:        provider.Decisions("../alpha/decisions"),
		isDecisionModel: func(model string) bool {
			return strings.HasPrefix(model, "typesafe/")
		},
	})

	registerProvider(ProviderTypeSafe, providerSpec{
		models:  []string{Jev, Jev1_13},
		envVars: []string{"TYPESAFE_API_KEY", "TYPESAFE_APIKEY", "TYPESAFE_KEY"},
		baseURL: "https://api.typesafe.ai",
		decide:  provider.Decisions("v1/systemone"),
	})

	registerProvider(ProviderOllama, providerSpec{
		models: []string{
			OllamaAlfred,
			OllamaAtheneV2,
			OllamaAya,
			OllamaAyaExpanse,
			OllamaBakllava,
			OllamaBespokeMinicheck,
			OllamaCodebooga,
			OllamaCodeGeeX4,
			OllamaCodeGemma,
			OllamaCodeLlama,
			OllamaCodeQwen,
			OllamaCodestral,
			OllamaCodeup,
			OllamaCogito,
			OllamaCogito2_1,
			OllamaCommandA,
			OllamaCommandR,
			OllamaCommandR7B,
			OllamaCommandR7BArabic,
			OllamaCommandRPlus,
			OllamaDBRX,
			OllamaDeepCoder,
			OllamaDeepScaler,
			OllamaDeepSeekCoder,
			OllamaDeepSeekCoderV2,
			OllamaDeepSeekLLM,
			OllamaDeepSeekOCR,
			OllamaDeepSeekR1,
			OllamaDeepSeekV2,
			OllamaDeepSeekV2_5,
			OllamaDeepSeekV3,
			OllamaDeepSeekV3_1,
			OllamaDeepSeekV4_1Flash,
			OllamaDeepSeekV4Flash,
			OllamaDeepSeekV4Pro,
			OllamaDevstral,
			OllamaDevstral2,
			OllamaDevstralSmall2,
			OllamaDolphin3,
			OllamaDolphinCoder,
			OllamaDolphinLlama3,
			OllamaDolphinMistral,
			OllamaDolphinMixtral,
			OllamaDolphinPhi,
			OllamaDuckDBNSQL,
			OllamaEverythingLM,
			OllamaExaone3_5,
			OllamaExaoneDeep,
			OllamaFalcon,
			OllamaFalcon2,
			OllamaFalcon3,
			OllamaFireFunctionV2,
			OllamaFunctionGemma,
			OllamaGemma,
			OllamaGemma2,
			OllamaGemma3,
			OllamaGemma3n,
			OllamaGemma4,
			OllamaGLM4,
			OllamaGLM4_7Flash,
			OllamaGLM5_1,
			OllamaGLM5_2,
			OllamaGLM5_3,
			OllamaGLM5_3Flash,
			OllamaGLMOCR,
			OllamaGoliath,
			OllamaGPTOSS,
			OllamaGPTOSSSafeguard,
			OllamaGranite3_1Dense,
			OllamaGranite3_1MoE,
			OllamaGranite3_2,
			OllamaGranite3_2Vision,
			OllamaGranite3_3,
			OllamaGranite3Dense,
			OllamaGranite3Guardian,
			OllamaGranite3MoE,
			OllamaGranite4,
			OllamaGranite4_1,
			OllamaGranite4_1Guardian,
			OllamaGranite4_2,
			OllamaGraniteCode,
			OllamaHermes3,
			OllamaInternLM2,
			OllamaKimiK2_6,
			OllamaKimiK2_7Code,
			OllamaKimiK3,
			OllamaLagunaS2_1,
			OllamaLagunaXS_2,
			OllamaLagunaXS2_1,
			OllamaLFM2,
			OllamaLFM2_5,
			OllamaLFM2_5Thinking,
			OllamaLlama2,
			OllamaLlama2Chinese,
			OllamaLlama2Uncensored,
			OllamaLlama3,
			OllamaLlama3_1,
			OllamaLlama3_2,
			OllamaLlama3_2Vision,
			OllamaLlama3_3,
			OllamaLlama3ChatQA,
			OllamaLlama3Gradient,
			OllamaLlama3GroqToolUse,
			OllamaLlama4,
			OllamaLlamaGuard3,
			OllamaLlamaPro,
			OllamaLlava,
			OllamaLlavaLlama3,
			OllamaLlavaPhi3,
			OllamaMagicoder,
			OllamaMagistral,
			OllamaMarcoO1,
			OllamaMathstral,
			OllamaMedGemma,
			OllamaMedGemma1_5,
			OllamaMeditron,
			OllamaMedLlama2,
			OllamaMegadolphin,
			OllamaMiniCPMV,
			OllamaMiniCPMV4_5,
			OllamaMiniCPMV4_6,
			OllamaMiniMaxM2_7,
			OllamaMiniMaxM3,
			OllamaMinistral3,
			OllamaMistral,
			OllamaMistralLarge,
			OllamaMistralLarge3,
			OllamaMistrallite,
			OllamaMistralMedium3_5,
			OllamaMistralNemo,
			OllamaMistralOpenOrca,
			OllamaMistralSmall,
			OllamaMistralSmall3_1,
			OllamaMistralSmall3_2,
			OllamaMixtral,
			OllamaMoondream,
			OllamaMuseGlimmer,
			OllamaNemotron,
			OllamaNemotron3,
			OllamaNemotron3_5Lightning,
			OllamaNemotron3Nano,
			OllamaNemotron3Super,
			OllamaNemotron3Ultra,
			OllamaNemotronCascade2,
			OllamaNemotronMini,
			OllamaNeuralChat,
			OllamaNexusRaven,
			OllamaNorthMiniCode1_0,
			OllamaNotus,
			OllamaNotux,
			OllamaNousHermes,
			OllamaNousHermes2,
			OllamaNousHermes2Mixtral,
			OllamaNuExtract,
			OllamaOLMo2,
			OllamaOLMo3,
			OllamaOLMo3_1,
			OllamaOpenChat,
			OllamaOpenCoder,
			OllamaOpenHermes,
			OllamaOpenOrcaPlatypus2,
			OllamaOpenThinker,
			OllamaOrca2,
			OllamaOrcaMini,
			OllamaOrnith,
			OllamaOrnith1_5,
			OllamaPhi,
			OllamaPhi3,
			OllamaPhi3_5,
			OllamaPhi4,
			OllamaPhi4Mini,
			OllamaPhi4MiniReasoning,
			OllamaPhi4Reasoning,
			OllamaPhindCodeLlama,
			OllamaQwen,
			OllamaQwen2,
			OllamaQwen2_5,
			OllamaQwen2_5Coder,
			OllamaQwen2_5VL,
			OllamaQwen2Math,
			OllamaQwen3,
			OllamaQwen3_5,
			OllamaQwen3_6,
			OllamaQwen3_8,
			OllamaQwen3_8FlashNext,
			OllamaQwen3Coder,
			OllamaQwen3CoderNext,
			OllamaQwen3Next,
			OllamaQwen3VL,
			OllamaQwQ,
			OllamaR1_1776,
			OllamaReaderLM,
			OllamaReflection,
			OllamaRNJ1,
			OllamaSailor2,
			OllamaSamanthaMistral,
			OllamaShieldGemma,
			OllamaSmallThinker,
			OllamaSmolLM,
			OllamaSmolLM2,
			OllamaSolar,
			OllamaSolarPro,
			OllamaSQLCoder,
			OllamaStableBeluga,
			OllamaStableCode,
			OllamaStableLM2,
			OllamaStableLMZephyr,
			OllamaStarCoder,
			OllamaStarCoder2,
			OllamaStarlingLM,
			OllamaTinyDolphin,
			OllamaTinyLlama,
			OllamaTranslateGemma,
			OllamaTulu3,
			OllamaVicuna,
			OllamaWizardCoder,
			OllamaWizardLM,
			OllamaWizardLM2,
			OllamaWizardLMUncensored,
			OllamaWizardMath,
			OllamaWizardVicuna,
			OllamaWizardVicunaUncensored,
			OllamaXWinLM,
			OllamaYarnLlama2,
			OllamaYarnMistral,
			OllamaYi,
			OllamaYiCoder,
			OllamaZephyr,
			OllamaQwen3_8B,
			OllamaGPTOSS20B,
		},
		envVars: []string{"OLLAMA_API_KEY", "OLLAMA_APIKEY", "OLLAMA_KEY"},
		baseURL: "http://localhost:11434/v1",
		step:    provider.OpenAI,
		schema:  schema.AdaptOpenAI,
		prepare: prepareOllama,
	})
}

// prepareAnthropic rejects settings the Messages API refuses.
// Context windows in input tokens, used to decide when to compact. They are
// deliberately conservative, by model family, so a newer model compacts a
// little early rather than late; checked on 2026-10-01. A wrong guess is
// recovered from when the provider rejects a request as too long.

func openAIWindow(model string) int {
	for _, w := range []struct {
		prefix string
		tokens int
	}{
		{"gpt-3.5", 16_385},
		{"gpt-4-32k", 32_768},
		{"gpt-4.1", 1_000_000},
		{"gpt-4o", 128_000}, {"chatgpt-4o", 128_000}, {"gpt-4-turbo", 128_000},
		{"gpt-4-1106", 128_000}, {"gpt-4-0125", 128_000}, {"gpt-4-vision", 128_000},
		{"o1-mini", 128_000}, {"o1-preview", 128_000}, {"codex-mini", 128_000},
		{"gpt-4", 8_192},
		{"o1", 200_000}, {"o3", 200_000}, {"o4", 200_000},
		{"gpt-5", 272_000}, {"gpt-6", 272_000},
	} {
		if strings.HasPrefix(model, w.prefix) {
			return w.tokens
		}
	}
	return 0
}

func anthropicWindow(model string) int {
	if strings.HasPrefix(model, "claude-") {
		return 200_000
	}
	return 0
}

func geminiWindow(model string) int {
	switch {
	case !strings.HasPrefix(model, "gemini-"), strings.Contains(model, "embedding"):
		return 0
	case strings.Contains(model, "-tts"):
		return 8_192
	case strings.Contains(model, "-image"):
		return 32_768
	case strings.Contains(model, "live"), strings.Contains(model, "audio"), strings.Contains(model, "computer-use"):
		return 128_000
	}
	return 1_048_576
}

func xAIWindow(model string) int {
	switch {
	case strings.HasPrefix(model, "grok-4"), strings.HasPrefix(model, "grok-code"), strings.HasPrefix(model, "grok-build"):
		return 256_000
	case strings.HasPrefix(model, "grok-3"):
		return 131_072
	}
	return 0
}

func deepSeekWindow(string) int { return 128_000 }

// openRouterWindow uses the window of the model's own provider.
func openRouterWindow(model string) int {
	vendor, name, _ := strings.Cut(model, "/")
	switch vendor {
	case "openai":
		return openAIWindow(name)
	case "anthropic":
		return anthropicWindow(name)
	case "google":
		return geminiWindow(name)
	case "x-ai":
		return xAIWindow(name)
	case "deepseek":
		return deepSeekWindow(name)
	}
	return 0
}

func prepareAnthropic(a *Agent) error {
	reasons := a.reasoning != "" && a.reasoning != ReasoningOff
	if a.temperature != nil && reasons {
		return errors.New("anthropic does not accept a temperature while the model reasons; remove WithTemperature or use WithReasoning(ReasoningOff)")
	}
	_, forcesTool := a.toolChoice.tool()
	if (forcesTool || a.toolChoice == ToolChoiceRequired) && reasons {
		return errors.New("anthropic cannot force a tool call while the model reasons; use WithReasoning(ReasoningOff) or ToolChoiceAuto")
	}
	return nil
}

// prepareGoogle rejects settings the Gemini API has no equivalent for.
func prepareGoogle(a *Agent) error {
	if a.parallel != nil && !*a.parallel {
		return errors.New("gemini has no setting to turn off parallel tool calls; remove WithParallelToolCalls(false)")
	}
	return nil
}

func prepareDeepSeek(a *Agent) error {
	if a.searchOptions != nil {
		return errors.New("native web search is not supported by this DeepSeek adapter")
	}
	return nil
}

func prepareOpenRouter(a *Agent) error {
	if a.searchOptions != nil {
		return errors.New("native web search is not supported by this OpenRouter adapter")
	}
	return nil
}

// prepareOllama sets a placeholder key for local servers, which ignore
// authentication. Ollama v0.13.3+ is required for stateless Responses support.
// JSON Schema via text.format is verified in local Ollama v0.34.0;
// Ollama Cloud does not currently support structured outputs.
func prepareOllama(a *Agent) error {
	if a.apiKey == "" || a.apiKey == "ollama" {
		// Hosted web search needs a real key: https://docs.ollama.com/web-search
		if a.searchOptions != nil {
			return errors.New("Ollama API key is required for Ollama web search")
		}
		a.apiKey = "ollama"
	}
	return nil
}
