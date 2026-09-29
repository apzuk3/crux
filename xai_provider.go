package crux

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

func init() {
	providerMu.Lock()
	defer providerMu.Unlock()

	providers[ProviderXAI] = []model{
		{Name: XAIGrok4_6},
		{Name: XAIGrok4_5},
		{Name: XAIGrok4_5Latest},
		{Name: XAIGrokBuildLatest},
		{Name: XAIGrok4_3},
		{Name: XAIGrok4_3Latest},
		{Name: XAIGrok4_20_0309Reasoning},
		{Name: XAIGrok4_20ReasoningLatest},
		{Name: XAIGrok4_20},
		{Name: XAIGrok4_20Reasoning},
		{Name: XAIGrok4_20_0309},
		{Name: XAIGrok4_20Beta0309Reasoning},
		{Name: XAIGrok4_20Beta},
		{Name: XAIGrok4_20Beta0309},
		{Name: XAIGrok4_20BetaLatest},
		{Name: XAIGrok4_20BetaLatestReasoning},
		{Name: XAIGrok4_20BetaReasoning},
		{Name: XAIGrok4_20ExperimentalBeta0304Reasoning},
		{Name: XAIGrok4_20ExperimentalBeta0304},
		{Name: XAIGrok4_20ExperimentalBetaReasoningLatest},
		{Name: XAIGrok4_20ExperimentalBetaLatest},
		{Name: XAIGrok4_20ReasoningGV2},
		{Name: XAIGrok4_20_0309NonReasoning},
		{Name: XAIGrok4_20NonReasoning},
		{Name: XAIGrok4_20NonReasoningLatest},
		{Name: XAIGrok4_20BetaNonReasoning},
		{Name: XAIGrok4_20BetaLatestNonReasoning},
		{Name: XAIGrok4_20ExperimentalBeta0304NonReasoning},
		{Name: XAIGrok4_20ExperimentalBetaNonReasoningLatest},
		{Name: XAIGrok4_20Beta0309NonReasoning},
		{Name: XAIGrok4_20NonReasoningGV2},
		{Name: XAIGrok4_20MultiAgent0309},
		{Name: XAIGrok4_20MultiAgent},
		{Name: XAIGrok4_20MultiAgentLatest},
		{Name: XAIGrok4_20MultiAgentBetaLatest},
		{Name: XAIGrok4_20MultiAgentExperimentalBeta0304},
		{Name: XAIGrok4_20MultiAgentExperimentalBetaLatest},
		{Name: XAIGrok4_20MultiAgentBeta0309},
		{Name: XAIGrokBuild0_1},
		{Name: XAIGrokCodeFast1},
		{Name: XAIGrokCodeFast},
		{Name: XAIGrokCodeFast1_0825},
	}
}
