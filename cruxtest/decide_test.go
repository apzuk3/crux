package cruxtest_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"testing"

	"github.com/apzuk3/crux"
	"github.com/apzuk3/crux/cruxtest"
	"github.com/stretchr/testify/require"
)

type triage struct {
	Urgent   bool    `json:"urgent" description:"Must a human act within an hour?" true:"Time-sensitive" false:"Can wait"`
	Area     string  `json:"area" description:"Which team owns this" choices:"billing=Payments and invoices|bug=Something is broken|other"`
	Severity int     `json:"severity" description:"Impact on the customer" levels:"none|minor|major|outage"`
	Anger    float64 `json:"anger" levels:"calm|annoyed|furious"`
}

type team string

func (team) Choices() map[string]string {
	return map[string]string{"infra": "Servers and networks", "web": "The website"}
}

type routed struct {
	Team team `json:"team"`
}

func decisionBody(t *testing.T, req *cruxtest.CapturedRequest) map[string]any {
	t.Helper()
	var body map[string]any
	require.NoError(t, json.Unmarshal(req.Body, &body))
	return body
}

func TestDecideJev(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnDecision(map[string]cruxtest.Answer{
		"urgent":   cruxtest.Noul(0.9),
		"area":     cruxtest.Choice("billing", 0.81, map[string]float64{"billing": 0.88, "bug": 0.12, "other": 0}),
		"severity": cruxtest.Score(1.6, 0.7, 0, 0.4, 0.6, 0),
		"anger":    cruxtest.Score(1.25, 0.9, 0, 0.75, 0.25),
	}).WithUsage(cruxtest.TokenUsage{InputTokens: 300, OutputTokens: 20})

	d, err := crux.NewDecider(crux.Jev, append(mock.AgentOptions(), crux.WithInstructions("You triage support tickets."))...)
	require.NoError(t, err)
	require.Equal(t, crux.ProviderTypeSafe, d.Provider())

	res, err := crux.Decide[triage](t.Context(), d, "My payouts have failed for 3 days!")
	require.NoError(t, err)
	require.Equal(t, triage{Urgent: true, Area: "billing", Severity: 2, Anger: 1.25}, res.Value)
	require.InDelta(t, 0.9, res.Confidence["urgent"], 1e-9)
	require.InDelta(t, 0.1, res.Probabilities["urgent"]["false"], 1e-9)
	require.Equal(t, 0.81, res.Confidence["area"])
	require.Equal(t, map[string]float64{"none": 0, "minor": 0.4, "major": 0.6, "outage": 0}, res.Probabilities["severity"])
	require.Equal(t, 300, res.Usage.InputTokens)

	reqs := mock.Requests()
	require.Len(t, reqs, 1)
	require.Equal(t, "https://api.typesafe.ai/v1/systemone", reqs[0].URL.String())
	require.Equal(t, "Bearer cruxtest-mock-key", reqs[0].Header.Get("Authorization"))
	body := decisionBody(t, reqs[0])
	require.Equal(t, "jev-latest", body["model"])
	require.Equal(t, "My payouts have failed for 3 days!", body["state"])
	questions := body["questions"].(map[string]any)
	require.Equal(t, map[string]any{
		"type":         "noul",
		"instructions": "You triage support tickets.\n\nMust a human act within an hour?",
		"criteria":     map[string]any{"true": "Time-sensitive", "false": "Can wait"},
	}, questions["urgent"])
	require.Equal(t, map[string]any{"billing": "Payments and invoices", "bug": "Something is broken", "other": "other"}, questions["area"].(map[string]any)["criteria"])
	require.Equal(t, []any{"none", "minor", "major", "outage"}, questions["severity"].(map[string]any)["criteria"])
	require.Equal(t, "score", questions["anger"].(map[string]any)["type"])
}

func TestDecideStateShapes(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnDecision(map[string]cruxtest.Answer{"team": cruxtest.Choice("web", 0.9, nil)})
	mock.Expect().ReturnDecision(map[string]cruxtest.Answer{"team": cruxtest.Choice("infra", 0.9, nil)})
	d := crux.MustDecider(crux.NewDecider(crux.Jev1_13, mock.AgentOptions()...))

	res, err := crux.Decide[routed](t.Context(), d, map[string]any{"page": "/checkout", "status": 500})
	require.NoError(t, err)
	require.Equal(t, team("web"), res.Value.Team)
	_, err = crux.Decide[routed](t.Context(), d, "first", "second")
	require.NoError(t, err)

	reqs := mock.Requests()
	require.Equal(t, map[string]any{"page": "/checkout", "status": float64(500)}, decisionBody(t, reqs[0])["state"])
	require.Equal(t, []any{"first", "second"}, decisionBody(t, reqs[1])["state"])
	criteria := decisionBody(t, reqs[0])["questions"].(map[string]any)["team"].(map[string]any)["criteria"]
	require.Equal(t, map[string]any{"infra": "Servers and networks", "web": "The website"}, criteria)
}

func TestDecideOpenRouterJev(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnRaw(http.StatusOK, []byte(`{"id":"dec_1","model":"typesafe/jev-1.13","provider":"TypeSafe",
		"answers":{"team":{"type":"choice","choice":"infra","confidence":0.7,"probabilities":{"infra":0.8,"web":0.2}}},
		"usage":{"input_tokens":12,"output_tokens":3,"cost":0.0000005}}`))
	d, err := crux.NewDecider(crux.OpenRouterDecisionModelJev1_13, mock.AgentOptions()...)
	require.NoError(t, err)
	require.Equal(t, crux.ProviderOpenrouter, d.Provider())

	res, err := crux.Decide[routed](t.Context(), d, "disk full on db-3")
	require.NoError(t, err)
	require.Equal(t, team("infra"), res.Value.Team)
	require.Equal(t, 0.0000005, res.Cost)
	require.Equal(t, "typesafe/jev-1.13", res.Model)
	require.Equal(t, "https://openrouter.ai/api/alpha/decisions", mock.Requests()[0].URL.String())
}

func TestDecideOpenRouterChatModelUsesOutputSchema(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnJSON(map[string]any{"team": "web"})
	d, err := crux.NewDecider(crux.OpenRouterChatModelGPT5_4Mini, mock.AgentOptions()...)
	require.NoError(t, err)

	res, err := crux.Decide[routed](t.Context(), d, "the checkout page is blank")
	require.NoError(t, err)
	require.Equal(t, team("web"), res.Value.Team)
	require.Nil(t, res.Confidence)
	require.Equal(t, "/api/v1/responses", mock.Requests()[0].URL.Path)
}

func TestDecideWithLLMs(t *testing.T) {
	for _, tt := range []struct {
		provider crux.Provider
		model    string
	}{
		{crux.ProviderOpenAI, crux.OpenAIGPT5_4Nano},
		{crux.ProviderAnthropic, crux.ClaudeHaiku4_5},
		{crux.ProviderGoogle, crux.Gemini3_5Flash},
	} {
		t.Run(string(tt.provider), func(t *testing.T) {
			mock := cruxtest.NewMock()
			mock.Expect().ReturnJSON(map[string]any{"urgent": true, "area": "bug", "severity": 3, "anger": 2})
			d, err := crux.NewDecider(tt.model, mock.AgentOptions()...)
			require.NoError(t, err)

			res, err := crux.Decide[triage](t.Context(), d, "Everything is down")
			require.NoError(t, err)
			require.Equal(t, triage{Urgent: true, Area: "bug", Severity: 3, Anger: 2}, res.Value)
			require.Nil(t, res.Confidence)
			require.Nil(t, res.Probabilities)
			require.Contains(t, mock.Requests()[0].BodyString(), `3: outage`)
		})
	}
}

func TestDecideLLMAnswerOutsideSchema(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnJSON(map[string]any{"team": "sales"})
	d := crux.MustDecider(crux.NewDecider(crux.OpenAIGPT5_4Nano, mock.AgentOptions()...))
	_, err := crux.Decide[routed](t.Context(), d, "pricing question")
	require.ErrorIs(t, err, crux.ErrOutputValidation)
}

func TestDecideContextTooLong(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnError(http.StatusUnprocessableEntity, `{"detail":[{"loc":["body","state"],"msg":"state is too long","type":"max_tokens_exceeded"}]}`)
	d := crux.MustDecider(crux.NewDecider(crux.Jev, mock.AgentOptions()...))
	_, err := crux.Decide[routed](t.Context(), d, "a very long ticket")
	require.ErrorIs(t, err, crux.ErrContextTooLong)
	mock.AssertTurnCount(t, 1) // 422 is not retried
}

func TestDecideRetriesOverload(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnError(529, `{"detail":"overloaded"}`)
	mock.Expect().ReturnDecision(map[string]cruxtest.Answer{"team": cruxtest.Choice("web", 0.9, nil)})
	d := crux.MustDecider(crux.NewDecider(crux.Jev, mock.AgentOptions()...))
	res, err := crux.Decide[routed](t.Context(), d, "blank page")
	require.NoError(t, err)
	require.Equal(t, team("web"), res.Value.Team)
	mock.AssertAllConsumed(t)
}

func TestDecideRejectsUndecidableTypes(t *testing.T) {
	mock := cruxtest.NewMock()
	d := crux.MustDecider(crux.NewDecider(crux.Jev, mock.AgentOptions()...))

	_, err := crux.Decide[struct {
		Summary string `json:"summary"`
	}](t.Context(), d, "x")
	require.ErrorContains(t, err, "free text can't be decided")

	_, err = crux.Decide[struct {
		Level int `json:"level"`
	}](t.Context(), d, "x")
	require.ErrorContains(t, err, "levels")

	_, err = crux.Decide[struct {
		Pick string `json:"pick" choices:"only"`
	}](t.Context(), d, "x")
	require.ErrorContains(t, err, "1 choices")

	_, err = crux.Decide[routed](t.Context(), d, crux.Data([]byte("x")))
	require.ErrorContains(t, err, "attachments")

	_, err = crux.Decide[routed](t.Context(), d)
	require.ErrorContains(t, err, "no state")

	require.Zero(t, mock.Calls())
}

func TestDecisionModelsCantRunAgents(t *testing.T) {
	_, err := crux.New("triage", crux.Jev, crux.WithAPIKey("k"))
	require.ErrorContains(t, err, "crux.NewDecider")
	_, err = crux.New("triage", crux.OpenRouterDecisionModelJev1_13, crux.WithAPIKey("k"))
	require.ErrorContains(t, err, "crux.NewDecider")
}

func TestNewDeciderRejectsAgentOptions(t *testing.T) {
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "noop", "Does nothing", func(ctx context.Context, in struct{}) (string, *crux.StateDelta, error) {
		return "", nil, nil
	})
	for name, opt := range map[string]crux.AgentOption{
		"tools":       crux.WithToolsRegistry([]string{"noop"}, reg),
		"temperature": crux.WithTemperature(0),
		"output":      crux.WithOutputSchemaFrom[routed](),
	} {
		t.Run(name, func(t *testing.T) {
			_, err := crux.NewDecider(crux.Jev, crux.WithAPIKey("k"), opt)
			require.Error(t, err)
		})
	}
	// A generative model takes sampling settings.
	_, err := crux.NewDecider(crux.OpenAIGPT5_4Nano, crux.WithAPIKey("k"), crux.WithTemperature(0))
	require.NoError(t, err)
}

func TestDecideRejectsAnswerOutsideChoices(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnDecision(map[string]cruxtest.Answer{"team": cruxtest.Choice("sales", 0.9, nil)})
	d := crux.MustDecider(crux.NewDecider(crux.Jev, mock.AgentOptions()...))
	_, err := crux.Decide[routed](t.Context(), d, "x")
	require.Error(t, err)
	require.False(t, errors.Is(err, crux.ErrContextTooLong))
	require.ErrorContains(t, err, `"sales"`)
}

func TestDecideLLMScoreOutsideLevels(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnJSON(map[string]any{"urgent": false, "area": "other", "severity": 7, "anger": 0})
	d := crux.MustDecider(crux.NewDecider(crux.OpenAIGPT5_4Nano, mock.AgentOptions()...))
	_, err := crux.Decide[triage](t.Context(), d, "x")
	require.ErrorIs(t, err, crux.ErrOutputValidation)
}
