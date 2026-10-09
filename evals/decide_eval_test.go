//go:build evals

// Scenario: typed decisions with crux.Decide on Jev (TypeSafe and OpenRouter)
// and on generative models through structured output.

package evals

import (
	"context"
	"os"
	"testing"
	"time"

	"crux.foo"
	"github.com/stretchr/testify/require"
)

type ticketTriage struct {
	Urgent bool   `json:"urgent" description:"Does the customer need help right away?" true:"Money, outages or deadlines are at stake" false:"It can wait"`
	Area   string `json:"area" description:"Which team should handle this?" choices:"billing=Payments, invoices and refunds|technical=Bugs, outages and integrations|sales=Pricing, upgrades and new accounts"`
	Mood   int    `json:"mood" description:"How frustrated is the customer?" levels:"calm|frustrated|very angry"`
}

func Test_Decide(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name   string
		model  string
		envVar string
		native bool
	}{
		{"TypeSafe_Jev", crux.Jev, "TYPESAFE_API_KEY", true},
		{"OpenRouter_Jev", crux.OpenRouterDecisionModelJev1_13, "OPENROUTER_API_KEY", true},
		{"OpenAI_GPT5_4Nano", crux.OpenAIGPT5_4Nano, "OPENAI_API_KEY", false},
		{"Anthropic_ClaudeHaiku4_5", crux.ClaudeHaiku4_5, "ANTHROPIC_API_KEY", false},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if os.Getenv(tc.envVar) == "" {
				t.Skipf("Skipping %s: %s not set", tc.name, tc.envVar)
			}
			ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
			defer cancel()

			d, err := crux.NewDecider(tc.model)
			require.NoError(t, err)
			res, err := crux.Decide[ticketTriage](ctx, d, "This is the THIRD day my payouts have failed. I can't pay my staff. Fix it now!")
			require.NoError(t, err)
			require.True(t, res.Value.Urgent)
			require.Equal(t, "billing", res.Value.Area)
			require.Equal(t, 2, res.Value.Mood)
			if tc.native {
				require.Greater(t, res.Confidence["area"], 0.5)
				require.Len(t, res.Probabilities["mood"], 3)
			} else {
				require.Nil(t, res.Confidence)
			}
		})
	}
}
