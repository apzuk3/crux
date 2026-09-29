package crux

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

func init() {
	providerMu.Lock()
	defer providerMu.Unlock()

	providers[ProviderOpenrouter] = []model{
		{Name: OpenRouterChatModelGPT6Astra},
		{Name: OpenRouterChatModelGPT5_6Sol},
		{Name: OpenRouterChatModelGPT5_6Terra},
		{Name: OpenRouterChatModelGPT5_6Luna},
		{Name: OpenRouterChatModelGPT5_5},
		{Name: OpenRouterChatModelGPT5_4},
		{Name: OpenRouterChatModelGPT5_4Mini},
		{Name: OpenRouterChatModelGPT5_4Nano},
		{Name: OpenRouterChatModelGPT5_2},
		{Name: OpenRouterChatModelGPT5_2ChatLatest},
		{Name: OpenRouterChatModelGPT5_2Pro},
		{Name: OpenRouterChatModelGPT5_1},
		{Name: OpenRouterChatModelGPT5_1Codex},
		{Name: OpenRouterChatModelGPT5},
		{Name: OpenRouterChatModelGPT5Mini},
		{Name: OpenRouterChatModelGPT5Nano},
		{Name: OpenRouterChatModelGPT4_1},
		{Name: OpenRouterChatModelGPT4_1Mini},
		{Name: OpenRouterChatModelGPT4_1Nano},
		{Name: OpenRouterChatModelO4Mini},
		{Name: OpenRouterChatModelO3},
		{Name: OpenRouterChatModelO3Mini},
		{Name: OpenRouterChatModelO1},
		{Name: OpenRouterChatModelGPT4o},
		{Name: OpenRouterChatModelGPT4o2024_11_20},
		{Name: OpenRouterChatModelGPT4o2024_08_06},
		{Name: OpenRouterChatModelGPT4o2024_05_13},
		{Name: OpenRouterChatModelGPT4oMini},
		{Name: OpenRouterChatModelGPT4oMini2024_07_18},
		{Name: OpenRouterChatModelGPT4Turbo},
		{Name: OpenRouterChatModelGPT4TurboPreview},
		{Name: OpenRouterChatModelGPT4},
		{Name: OpenRouterChatModelGPT3_5Turbo},
		{Name: OpenRouterChatModelGPT3_5Turbo16k},
		{Name: OpenRouterChatModelGPT3_5Turbo0613},
		{Name: OpenRouterGemini3_8Flash},
		{Name: OpenRouterGemini3_7Flash},
		{Name: OpenRouterGemini3_6Flash},
		{Name: OpenRouterGemini3_5Flash},
		{Name: OpenRouterGemini3_5FlashLite},
		{Name: OpenRouterGemini3_1FlashLite},
		{Name: OpenRouterGemini3_1FlashImage},
		{Name: OpenRouterGemini3_1FlashLiteImage},
		{Name: OpenRouterGemini3ProImage},
		{Name: OpenRouterGemini3_1ProPreview},
		{Name: OpenRouterGemini3FlashPreview},
		{Name: OpenRouterGemini2_5Pro},
		{Name: OpenRouterGemini2_5Flash},
		{Name: OpenRouterGemini2_5FlashLite},
		{Name: OpenRouterGemini2_5FlashImage},
		{Name: OpenRouterAnthropicClaudeFable5_1},
		{Name: OpenRouterAnthropicClaudeSonnet5},
		{Name: OpenRouterAnthropicClaudeFable5},
		{Name: OpenRouterAnthropicClaudeOpus5},
		{Name: OpenRouterAnthropicClaudeOpus4_8},
		{Name: OpenRouterAnthropicClaudeOpus4_7},
		{Name: OpenRouterAnthropicClaudeOpus4_6},
		{Name: OpenRouterAnthropicClaudeSonnet4_6},
		{Name: OpenRouterAnthropicClaudeHaiku4_5},
		{Name: OpenRouterAnthropicClaudeOpus4_5},
		{Name: OpenRouterAnthropicClaudeSonnet4_5},
		{Name: OpenRouterDeepSeekV4_1Flash},
		{Name: OpenRouterDeepSeekV4Pro},
		{Name: OpenRouterDeepSeekV4Flash},
		{Name: OpenRouterDeepSeekV3_2},
		{Name: OpenRouterDeepSeekR1},
		{Name: OpenRouterDeepSeekR1_0528},
		{Name: OpenRouterDeepSeekChat},
		{Name: OpenRouterQwen3_8Max0902},
		{Name: OpenRouterQwen3_8Flash},
		{Name: OpenRouterQwen3_8_27B},
		{Name: OpenRouterQwen3_8_2_4TA95B},
		{Name: OpenRouterQwen3_7Max},
		{Name: OpenRouterQwen3_7Plus},
		{Name: OpenRouterQwen3_7Flash},
		{Name: OpenRouterQwen3Coder},
		{Name: OpenRouterQwen3CoderNext},
		{Name: OpenRouterQwen3CoderPlus},
		{Name: OpenRouterQwen3CoderFlash},
		{Name: OpenRouterMetaMuseSpark1_3},
		{Name: OpenRouterMetaMuseSpark1_3Contributor},
		{Name: OpenRouterMetaMuseGlimmer30B},
		{Name: OpenRouterMetaLlama4Maverick},
		{Name: OpenRouterMetaLlama4Scout},
		{Name: OpenRouterMetaLlama3_3_70BInstruct},
		{Name: OpenRouterMetaLlama3_1_8BInstruct},
		{Name: OpenRouterMistralMedium3_5},
		{Name: OpenRouterMistralSmall2603},
		{Name: OpenRouterMistralLarge2512},
		{Name: OpenRouterMistralDevstral2512},
		{Name: OpenRouterMistralCodestral2508},
		{Name: OpenRouterMistralMinistral14B2512},
		{Name: OpenRouterMistralMinistral8B2512},
		{Name: OpenRouterMistralMinistral3B2512},
		{Name: OpenRouterXAIGrok4_6},
		{Name: OpenRouterXAIGrok4_5},
		{Name: OpenRouterXAIGrok4_3},
		{Name: OpenRouterXAIGrok4_20},
		{Name: OpenRouterXAIGrok4_20MultiAgent},
		{Name: OpenRouterXAIGrokBuild0_1},
		{Name: OpenRouterMoonshotKimiK3},
		{Name: OpenRouterMoonshotKimiK2_7Code},
		{Name: OpenRouterMoonshotKimiK2_6},
		{Name: OpenRouterMoonshotKimiK2_5},
		{Name: OpenRouterMoonshotKimiK2Thinking},
		{Name: OpenRouterMiniMaxM3},
		{Name: OpenRouterMiniMaxM2_7},
		{Name: OpenRouterMiniMaxM2_5},
		{Name: OpenRouterZAIGLM5_3},
		{Name: OpenRouterZAIGLM5_3Flash},
		{Name: OpenRouterZAIGLM5_2},
		{Name: OpenRouterZAIGLM5_1},
		{Name: OpenRouterZAIGLM5VTurbo},
		{Name: OpenRouterAmazonNova2LiteV1},
		{Name: OpenRouterAmazonNovaPremierV1},
		{Name: OpenRouterAmazonNovaProV1},
		{Name: OpenRouterAmazonNovaLiteV1},
		{Name: OpenRouterAmazonNovaMicroV1},
		{Name: OpenRouterCohereCommandA},
		{Name: OpenRouterCohereCommandR08_2024},
		{Name: OpenRouterCohereCommandRPlus08_2024},
		{Name: OpenRouterCohereNorthMiniCodeFree},
		{Name: OpenRouterPerplexitySonar},
		{Name: OpenRouterPerplexitySonarPro},
		{Name: OpenRouterPerplexitySonarProSearch},
		{Name: OpenRouterPerplexitySonarReasoningPro},
		{Name: OpenRouterPerplexitySonarDeepResearch},
		{Name: OpenRouterNVIDIANemotron3_5Lightning},
		{Name: OpenRouterNVIDIANemotron3Ultra550BA55B},
		{Name: OpenRouterNVIDIANemotron3Super120BA12B},
		{Name: OpenRouterNVIDIANemotron3Nano30BA3B},
		{Name: OpenRouterMicrosoftPhi4},
		{Name: OpenRouterMicrosoftWizardLM2_8x22B},
		{Name: OpenRouterIBMGranite4_2_8B},
		{Name: OpenRouterIBMGranite4_0HMicro},
		{Name: OpenRouterTencentHY4Preview},
		{Name: OpenRouterTencentHY3},
		{Name: OpenRouterByteDanceSeed2_1Turbo},
		{Name: OpenRouterByteDanceSeed2_0Code},
		{Name: OpenRouterByteDanceSeed2_0Lite},
		{Name: OpenRouterByteDanceSeed2_0Mini},
		{Name: OpenRouterByteDanceUITars1_5_7B},
		{Name: OpenRouterSakanaFuguUltraV2},
		{Name: OpenRouterSakanaFuguMax},
		{Name: OpenRouterInclusionAILing3_0Flash},
		{Name: OpenRouterInclusionAILing3_0FlashVL},
		{Name: OpenRouterInceptionMercury2_5},
		{Name: OpenRouterNexAGINexN2_5MiniFree},
		{Name: OpenRouterNexAGINexN2_5ProFree},
		{Name: OpenRouterDotsStudioDots3NotePreviewFree},
		{Name: OpenRouterLiquidLFM2_5_2_6BFree},
		{Name: OpenRouterUpstageSolarPro4},
		{Name: OpenRouterThinkingMachinesInkling},
		{Name: OpenRouterThinkingMachinesInklingSmall},
		{Name: OpenRouterPoolsideLagunaS2_1},
		{Name: OpenRouterPoolsideLagunaXS2_1},
		{Name: OpenRouterMeituanLongCat2_0},
		{Name: OpenRouterKwaiPilotKatCoderProV2_5},
		{Name: OpenRouterAionLabsAion3_0},
		{Name: OpenRouterAionLabsAion3_0Mini},
		{Name: OpenRouterStepFunStep3_7Flash},
		{Name: OpenRouterPerceptronMK1},
		{Name: OpenRouterXiaomiMiMoV2_5Pro},
		{Name: OpenRouterXiaomiMiMoV2_5},
		{Name: OpenRouterArceeAITrinityLargeThinking},
		{Name: OpenRouterRekaAIEdge},
		{Name: OpenRouterRekaAIFlash3},
		{Name: OpenRouterWriterPalmyraX5},
		{Name: OpenRouterRelaceSearch},
		{Name: OpenRouterRelaceApply3},
		{Name: OpenRouterTheDrummerCydonia24BV4_1},
		{Name: OpenRouterNousResearchHermes4_405B},
		{Name: OpenRouterCognitiveComputationsDolphinMistral24BVeniceEdition},
		{Name: OpenRouterMorphV3Large},
		{Name: OpenRouterMorphV3Fast},
		{Name: OpenRouterBaiduErnie4_5VL424BA47B},
		{Name: OpenRouterSao10KL3_3Euryale70B},
		{Name: OpenRouterAnthraciteMagnumV4_72B},
		{Name: OpenRouterMancerWeaver},
		{Name: OpenRouterUndi95RemmSlerpL2_13B},
		{Name: OpenRouterGrypheMythoMaxL2_13B},
		{Name: OpenRouterInferenceNetSchematronV2Turbo},
		{Name: OpenRouterInferenceNetSchematronV2Small},
	}
}
