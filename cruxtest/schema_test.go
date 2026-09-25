package cruxtest_test

import (
	"context"
	"testing"

	"github.com/apzuk3/crux"
	"github.com/apzuk3/crux/cruxtest"
	"github.com/stretchr/testify/require"
)

type structuredAnswer struct {
	Status string  `json:"status" jsonschema:"enum=ok,enum=error"`
	Score  int     `json:"score" jsonschema:"minimum=1"`
	Note   *string `json:"note,omitempty"`
}

func TestStructuredOutputWireAndValidation(t *testing.T) {
	for _, provider := range []crux.Provider{crux.ProviderOpenAI, crux.ProviderAnthropic, crux.ProviderGoogle, crux.ProviderXAI, crux.ProviderOpenrouter, crux.ProviderDeepSeek, crux.ProviderOllama} {
		t.Run(string(provider), func(t *testing.T) {
			mock := cruxtest.NewMock(cruxtest.WithProvider(provider))
			mock.Expect().ReturnText(`{"status":"ok","score":2,"note":null}`)
			mock.Expect().ReturnText(`{"status":"wrong","score":2,"note":null}`)
			a, err := crux.New("structured", "test-model",
				crux.WithProvider(provider), crux.WithAPIKey("mock"), crux.WithHTTPClient(mock.Client()),
				crux.WithOutputSchemaFrom[structuredAnswer]())
			require.NoError(t, err)
			var answer structuredAnswer
			require.NoError(t, a.RunInto(context.Background(), "extract", &answer))
			require.Equal(t, "ok", answer.Status)
			require.Equal(t, 2, answer.Score)
			_, err = a.Run(context.Background(), "extract again")
			require.ErrorIs(t, err, crux.ErrOutputValidation)
			// Retained invalid output must not bypass validation on Resume.
			_, err = a.Run(context.Background(), nil)
			require.ErrorIs(t, err, crux.ErrOutputValidation)
			var body map[string]any
			require.NoError(t, mock.Requests()[0].UnmarshalBody(&body))
			var schema map[string]any
			switch provider {
			case crux.ProviderAnthropic:
				schema = body["output_config"].(map[string]any)["format"].(map[string]any)["schema"].(map[string]any)
				props := schema["properties"].(map[string]any)
				require.NotContains(t, props["score"], "minimum")
				require.Contains(t, props["score"].(map[string]any)["description"], "minimum=1")
				require.NotContains(t, schema["required"], "note")
			case crux.ProviderGoogle:
				config := body["generationConfig"].(map[string]any)
				require.Equal(t, "application/json", config["responseMimeType"])
				schema = config["responseJsonSchema"].(map[string]any)
				require.NotContains(t, schema["required"], "note")
			default:
				format := body["text"].(map[string]any)["format"].(map[string]any)
				require.Equal(t, true, format["strict"])
				schema = format["schema"].(map[string]any)
				require.Contains(t, schema["required"], "note")
			}
			require.Equal(t, false, schema["additionalProperties"])
			require.NotContains(t, schema, "$schema")
			require.NotContains(t, schema, "$id")
			mock.AssertTurnCount(t, 2)
			mock.AssertAllConsumed(t)
		})
	}
}

func TestStructuredOutputRepair(t *testing.T) {
	for _, tc := range []struct {
		name      string
		retries   int
		turns     int32
		responses []string
		wantError bool
	}{
		{"disabled", 0, 10, []string{`{}`}, true},
		{"corrected", 1, 10, []string{`{}`, `{"status":"ok","score":1,"note":null}`}, false},
		{"exhausted", 1, 10, []string{`{}`, `{}`}, true},
		{"turn limit", 3, 1, []string{`{}`}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mock := cruxtest.NewMock()
			for _, response := range tc.responses {
				mock.Expect().ReturnText(response)
			}
			a, err := crux.New("repair", "test-model", crux.WithProvider(crux.ProviderOpenAI),
				crux.WithHTTPClient(mock.Client()), crux.WithAPIKey("mock"),
				crux.WithOutputSchemaFrom[structuredAnswer](), crux.WithMaxRepairs(tc.retries), crux.WithMaxTurns(tc.turns))
			require.NoError(t, err)
			_, err = a.Run(context.Background(), "extract")
			if tc.wantError {
				require.ErrorIs(t, err, crux.ErrOutputValidation)
			} else {
				require.NoError(t, err)
			}
			if len(tc.responses) > 1 {
				require.Contains(t, mock.Requests()[1].BodyString(), "Return corrected JSON")
			}
			mock.AssertTurnCount(t, len(tc.responses))
			mock.AssertAllConsumed(t)
		})
	}
}

func TestStructuredOutputAcrossToolTurns(t *testing.T) {
	reg := registerMockTools(t)
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("get_weather", WeatherArgs{City: "Paris"})
	mock.Expect().ReturnText(`{"status":"ok","score":1,"note":null}`)
	a, err := crux.New("tools", "test-model", crux.WithProvider(crux.ProviderOpenAI),
		crux.WithAPIKey("mock"), crux.WithHTTPClient(mock.Client()),
		crux.WithOutputSchemaFrom[structuredAnswer](), crux.WithToolsRegistry([]string{"get_weather"}, reg))
	require.NoError(t, err)
	_, err = a.Run(context.Background(), "check the weather")
	require.NoError(t, err)
	var first, second map[string]any
	require.NoError(t, mock.Requests()[0].UnmarshalBody(&first))
	require.NoError(t, mock.Requests()[1].UnmarshalBody(&second))
	require.Equal(t, first["text"], second["text"])
	require.Equal(t, false, first["tools"].([]any)[0].(map[string]any)["strict"])
	mock.AssertAllConsumed(t)
}

func TestRunIntoValidatesBeforeMutatingTarget(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnText(`{"status":"ok","score":0,"note":null}`)
	a, err := crux.New("validate", "test-model", crux.WithProvider(crux.ProviderAnthropic),
		crux.WithHTTPClient(mock.Client()), crux.WithAPIKey("mock"), crux.WithOutputSchemaFrom[structuredAnswer]())
	require.NoError(t, err)
	require.Error(t, a.RunInto(context.Background(), "extract", nil))
	mock.AssertTurnCount(t, 0)
	answer := structuredAnswer{Status: "original", Score: 42}
	err = a.RunInto(context.Background(), "extract", &answer)
	require.ErrorIs(t, err, crux.ErrOutputValidation)
	require.Equal(t, "original", answer.Status)
	require.Equal(t, 42, answer.Score)
	mock.AssertAllConsumed(t)
}
