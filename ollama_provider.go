package crux

import "errors"

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
	registerProvider(ProviderOllama, providerSpec{
		models: []model{
			{Name: OllamaAlfred},
			{Name: OllamaAtheneV2},
			{Name: OllamaAya},
			{Name: OllamaAyaExpanse},
			{Name: OllamaBakllava},
			{Name: OllamaBespokeMinicheck},
			{Name: OllamaCodebooga},
			{Name: OllamaCodeGeeX4},
			{Name: OllamaCodeGemma},
			{Name: OllamaCodeLlama},
			{Name: OllamaCodeQwen},
			{Name: OllamaCodestral},
			{Name: OllamaCodeup},
			{Name: OllamaCogito},
			{Name: OllamaCogito2_1},
			{Name: OllamaCommandA},
			{Name: OllamaCommandR},
			{Name: OllamaCommandR7B},
			{Name: OllamaCommandR7BArabic},
			{Name: OllamaCommandRPlus},
			{Name: OllamaDBRX},
			{Name: OllamaDeepCoder},
			{Name: OllamaDeepScaler},
			{Name: OllamaDeepSeekCoder},
			{Name: OllamaDeepSeekCoderV2},
			{Name: OllamaDeepSeekLLM},
			{Name: OllamaDeepSeekOCR},
			{Name: OllamaDeepSeekR1},
			{Name: OllamaDeepSeekV2},
			{Name: OllamaDeepSeekV2_5},
			{Name: OllamaDeepSeekV3},
			{Name: OllamaDeepSeekV3_1},
			{Name: OllamaDeepSeekV4_1Flash},
			{Name: OllamaDeepSeekV4Flash},
			{Name: OllamaDeepSeekV4Pro},
			{Name: OllamaDevstral},
			{Name: OllamaDevstral2},
			{Name: OllamaDevstralSmall2},
			{Name: OllamaDolphin3},
			{Name: OllamaDolphinCoder},
			{Name: OllamaDolphinLlama3},
			{Name: OllamaDolphinMistral},
			{Name: OllamaDolphinMixtral},
			{Name: OllamaDolphinPhi},
			{Name: OllamaDuckDBNSQL},
			{Name: OllamaEverythingLM},
			{Name: OllamaExaone3_5},
			{Name: OllamaExaoneDeep},
			{Name: OllamaFalcon},
			{Name: OllamaFalcon2},
			{Name: OllamaFalcon3},
			{Name: OllamaFireFunctionV2},
			{Name: OllamaFunctionGemma},
			{Name: OllamaGemma},
			{Name: OllamaGemma2},
			{Name: OllamaGemma3},
			{Name: OllamaGemma3n},
			{Name: OllamaGemma4},
			{Name: OllamaGLM4},
			{Name: OllamaGLM4_7Flash},
			{Name: OllamaGLM5_1},
			{Name: OllamaGLM5_2},
			{Name: OllamaGLM5_3},
			{Name: OllamaGLM5_3Flash},
			{Name: OllamaGLMOCR},
			{Name: OllamaGoliath},
			{Name: OllamaGPTOSS},
			{Name: OllamaGPTOSSSafeguard},
			{Name: OllamaGranite3_1Dense},
			{Name: OllamaGranite3_1MoE},
			{Name: OllamaGranite3_2},
			{Name: OllamaGranite3_2Vision},
			{Name: OllamaGranite3_3},
			{Name: OllamaGranite3Dense},
			{Name: OllamaGranite3Guardian},
			{Name: OllamaGranite3MoE},
			{Name: OllamaGranite4},
			{Name: OllamaGranite4_1},
			{Name: OllamaGranite4_1Guardian},
			{Name: OllamaGranite4_2},
			{Name: OllamaGraniteCode},
			{Name: OllamaHermes3},
			{Name: OllamaInternLM2},
			{Name: OllamaKimiK2_6},
			{Name: OllamaKimiK2_7Code},
			{Name: OllamaKimiK3},
			{Name: OllamaLagunaS2_1},
			{Name: OllamaLagunaXS_2},
			{Name: OllamaLagunaXS2_1},
			{Name: OllamaLFM2},
			{Name: OllamaLFM2_5},
			{Name: OllamaLFM2_5Thinking},
			{Name: OllamaLlama2},
			{Name: OllamaLlama2Chinese},
			{Name: OllamaLlama2Uncensored},
			{Name: OllamaLlama3},
			{Name: OllamaLlama3_1},
			{Name: OllamaLlama3_2},
			{Name: OllamaLlama3_2Vision},
			{Name: OllamaLlama3_3},
			{Name: OllamaLlama3ChatQA},
			{Name: OllamaLlama3Gradient},
			{Name: OllamaLlama3GroqToolUse},
			{Name: OllamaLlama4},
			{Name: OllamaLlamaGuard3},
			{Name: OllamaLlamaPro},
			{Name: OllamaLlava},
			{Name: OllamaLlavaLlama3},
			{Name: OllamaLlavaPhi3},
			{Name: OllamaMagicoder},
			{Name: OllamaMagistral},
			{Name: OllamaMarcoO1},
			{Name: OllamaMathstral},
			{Name: OllamaMedGemma},
			{Name: OllamaMedGemma1_5},
			{Name: OllamaMeditron},
			{Name: OllamaMedLlama2},
			{Name: OllamaMegadolphin},
			{Name: OllamaMiniCPMV},
			{Name: OllamaMiniCPMV4_5},
			{Name: OllamaMiniCPMV4_6},
			{Name: OllamaMiniMaxM2_7},
			{Name: OllamaMiniMaxM3},
			{Name: OllamaMinistral3},
			{Name: OllamaMistral},
			{Name: OllamaMistralLarge},
			{Name: OllamaMistralLarge3},
			{Name: OllamaMistrallite},
			{Name: OllamaMistralMedium3_5},
			{Name: OllamaMistralNemo},
			{Name: OllamaMistralOpenOrca},
			{Name: OllamaMistralSmall},
			{Name: OllamaMistralSmall3_1},
			{Name: OllamaMistralSmall3_2},
			{Name: OllamaMixtral},
			{Name: OllamaMoondream},
			{Name: OllamaMuseGlimmer},
			{Name: OllamaNemotron},
			{Name: OllamaNemotron3},
			{Name: OllamaNemotron3_5Lightning},
			{Name: OllamaNemotron3Nano},
			{Name: OllamaNemotron3Super},
			{Name: OllamaNemotron3Ultra},
			{Name: OllamaNemotronCascade2},
			{Name: OllamaNemotronMini},
			{Name: OllamaNeuralChat},
			{Name: OllamaNexusRaven},
			{Name: OllamaNorthMiniCode1_0},
			{Name: OllamaNotus},
			{Name: OllamaNotux},
			{Name: OllamaNousHermes},
			{Name: OllamaNousHermes2},
			{Name: OllamaNousHermes2Mixtral},
			{Name: OllamaNuExtract},
			{Name: OllamaOLMo2},
			{Name: OllamaOLMo3},
			{Name: OllamaOLMo3_1},
			{Name: OllamaOpenChat},
			{Name: OllamaOpenCoder},
			{Name: OllamaOpenHermes},
			{Name: OllamaOpenOrcaPlatypus2},
			{Name: OllamaOpenThinker},
			{Name: OllamaOrca2},
			{Name: OllamaOrcaMini},
			{Name: OllamaOrnith},
			{Name: OllamaOrnith1_5},
			{Name: OllamaPhi},
			{Name: OllamaPhi3},
			{Name: OllamaPhi3_5},
			{Name: OllamaPhi4},
			{Name: OllamaPhi4Mini},
			{Name: OllamaPhi4MiniReasoning},
			{Name: OllamaPhi4Reasoning},
			{Name: OllamaPhindCodeLlama},
			{Name: OllamaQwen},
			{Name: OllamaQwen2},
			{Name: OllamaQwen2_5},
			{Name: OllamaQwen2_5Coder},
			{Name: OllamaQwen2_5VL},
			{Name: OllamaQwen2Math},
			{Name: OllamaQwen3},
			{Name: OllamaQwen3_5},
			{Name: OllamaQwen3_6},
			{Name: OllamaQwen3_8},
			{Name: OllamaQwen3_8FlashNext},
			{Name: OllamaQwen3Coder},
			{Name: OllamaQwen3CoderNext},
			{Name: OllamaQwen3Next},
			{Name: OllamaQwen3VL},
			{Name: OllamaQwQ},
			{Name: OllamaR1_1776},
			{Name: OllamaReaderLM},
			{Name: OllamaReflection},
			{Name: OllamaRNJ1},
			{Name: OllamaSailor2},
			{Name: OllamaSamanthaMistral},
			{Name: OllamaShieldGemma},
			{Name: OllamaSmallThinker},
			{Name: OllamaSmolLM},
			{Name: OllamaSmolLM2},
			{Name: OllamaSolar},
			{Name: OllamaSolarPro},
			{Name: OllamaSQLCoder},
			{Name: OllamaStableBeluga},
			{Name: OllamaStableCode},
			{Name: OllamaStableLM2},
			{Name: OllamaStableLMZephyr},
			{Name: OllamaStarCoder},
			{Name: OllamaStarCoder2},
			{Name: OllamaStarlingLM},
			{Name: OllamaTinyDolphin},
			{Name: OllamaTinyLlama},
			{Name: OllamaTranslateGemma},
			{Name: OllamaTulu3},
			{Name: OllamaVicuna},
			{Name: OllamaWizardCoder},
			{Name: OllamaWizardLM},
			{Name: OllamaWizardLM2},
			{Name: OllamaWizardLMUncensored},
			{Name: OllamaWizardMath},
			{Name: OllamaWizardVicuna},
			{Name: OllamaWizardVicunaUncensored},
			{Name: OllamaXWinLM},
			{Name: OllamaYarnLlama2},
			{Name: OllamaYarnMistral},
			{Name: OllamaYi},
			{Name: OllamaYiCoder},
			{Name: OllamaZephyr},
			{Name: OllamaQwen3_8B},
			{Name: OllamaGPTOSS20B},
		},
		envVars: []string{"OLLAMA_API_KEY", "OLLAMA_APIKEY", "OLLAMA_KEY"},
		baseURL: "http://localhost:11434/v1",
		step:    (*Agent).openAIstep,
		schema:  adaptOpenAI,
		prepare: prepareOllama,
	})
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
