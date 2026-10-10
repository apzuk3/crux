package cruxtest_test

import (
	"context"
	"testing"

	"crux.foo"
	"crux.foo/cruxtest"
	"github.com/invopop/jsonschema"
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
				crux.WithProvider(provider),
				crux.WithOutputSchemaFrom[structuredAnswer]())
			require.NoError(t, err)
			sess, err := crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client()))
			require.NoError(t, err)
			var answer structuredAnswer
			require.NoError(t, sess.RunInto(context.Background(), &answer, "extract"))
			require.Equal(t, "ok", answer.Status)
			require.Equal(t, 2, answer.Score)
			_, err = sess.Run(context.Background(), "extract again")
			require.ErrorIs(t, err, crux.ErrOutputValidation)
			// Retained invalid output must not bypass validation on Resume.
			_, err = sess.Run(context.Background(), nil)
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
		turns     int
		responses []string
		wantError bool
	}{
		{"disabled", 0, 10, []string{`{}`}, true},
		{"corrected", 1, 10, []string{`{}`, `{"status":"ok","score":1,"note":null}`}, false},
		{"exhausted", 1, 10, []string{`{}`, `{}`}, true},
		{"not counted as turns", 1, 1, []string{`{}`, `{"status":"ok","score":1,"note":null}`}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mock := cruxtest.NewMock()
			for _, response := range tc.responses {
				mock.Expect().ReturnText(response)
			}
			a, err := crux.New("repair", "test-model", crux.WithProvider(crux.ProviderOpenAI),
				crux.WithOutputSchemaFrom[structuredAnswer](), crux.WithMaxRepairs(tc.retries), crux.WithMaxTurns(tc.turns))
			require.NoError(t, err)
			sess, err := crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client()))
			require.NoError(t, err)
			_, err = sess.Run(context.Background(), "extract")
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
		crux.WithOutputSchemaFrom[structuredAnswer](), crux.WithToolsRegistry([]string{"get_weather"}, reg))
	require.NoError(t, err)
	sess, err := crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client()))
	require.NoError(t, err)
	_, err = sess.Run(context.Background(), "check the weather")
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
	a, err := crux.New("validate", "test-model", crux.WithProvider(crux.ProviderAnthropic), crux.WithOutputSchemaFrom[structuredAnswer]())
	require.NoError(t, err)
	sess, err := crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client()))
	require.NoError(t, err)
	require.Error(t, sess.RunInto(context.Background(), nil, "extract"))
	mock.AssertTurnCount(t, 0)
	answer := structuredAnswer{Status: "original", Score: 42}
	err = sess.RunInto(context.Background(), &answer, "extract")
	require.ErrorIs(t, err, crux.ErrOutputValidation)
	require.Equal(t, "original", answer.Status)
	require.Equal(t, 42, answer.Score)
	mock.AssertAllConsumed(t)
}

type withMapAnswer struct {
	Labels map[string]string `json:"labels"`
}

func TestMapOutputSchemaCompatibility(t *testing.T) {
	t.Run("gemini preserves map schema", func(t *testing.T) {
		mock := cruxtest.NewMock(cruxtest.WithProvider(crux.ProviderGoogle))
		mock.Expect().ReturnText(`{"labels":{"env":"prod"}}`)
		a, err := crux.New("map-gemini", "test-model",
			crux.WithProvider(crux.ProviderGoogle),
			crux.WithOutputSchemaFrom[withMapAnswer]())
		require.NoError(t, err)
		sess, err := crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client()))
		require.NoError(t, err)
		var res withMapAnswer
		require.NoError(t, sess.RunInto(context.Background(), &res, "get labels"))
		require.Equal(t, "prod", res.Labels["env"])

		var body map[string]any
		require.NoError(t, mock.Requests()[0].UnmarshalBody(&body))
		schema := body["generationConfig"].(map[string]any)["responseJsonSchema"].(map[string]any)
		props := schema["properties"].(map[string]any)
		labelsProp := props["labels"].(map[string]any)
		require.NotEqual(t, false, labelsProp["additionalProperties"])
	})

	t.Run("openai-style providers reject a dynamic map schema in New", func(t *testing.T) {
		for _, provider := range []crux.Provider{crux.ProviderOpenAI, crux.ProviderOpenrouter, crux.ProviderXAI} {
			_, err := crux.New("map-"+string(provider), "test-model",
				crux.WithProvider(provider),
				crux.WithOutputSchemaFrom[withMapAnswer]())
			require.ErrorContains(t, err, "does not support dynamic map schemas", provider)
			require.ErrorContains(t, err, "output schema", provider)
		}
		_, err := crux.New("map-google", crux.Gemini2_5Flash, crux.WithOutputSchemaFrom[withMapAnswer]())
		require.NoError(t, err)
	})
}

func TestInvalidSchemaFailsEarlyZeroRequests(t *testing.T) {
	mock := cruxtest.NewMock()
	// Directly provide a jsonschema.Schema with a bad ref
	schema := &jsonschema.Schema{
		Ref: "#/$defs/DoesNotExist",
	}
	a, err := crux.New("bad-schema", "test-model",
		crux.WithProvider(crux.ProviderOpenAI),
		crux.WithOutputSchema(schema))
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client()))
	require.NoError(t, err)
	_, err = sess.Run(context.Background(), "run")
	require.Error(t, err)
	require.Contains(t, err.Error(), "invalid output schema")
	mock.AssertTurnCount(t, 0)
}

type largeNumberAnswer struct {
	ID uint64 `json:"id" jsonschema:"enum=9007199254740993"`
}

func TestLargeNumberPrecisionPreserved(t *testing.T) {
	mock := cruxtest.NewMock()
	// 9007199254740993 (2^53 + 1) would be rounded to 9007199254740992 if parsed as float64
	mock.Expect().ReturnText(`{"id": 9007199254740993}`)
	a, err := crux.New("large-num", "test-model",
		crux.WithProvider(crux.ProviderOpenAI),
		crux.WithOutputSchemaFrom[largeNumberAnswer]())
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client()))
	require.NoError(t, err)
	var res largeNumberAnswer
	require.NoError(t, sess.RunInto(context.Background(), &res, "get id"))
	require.Equal(t, uint64(9007199254740993), res.ID)

	// Off-by-one rounded number must fail validation
	mock.Expect().ReturnText(`{"id": 9007199254740992}`)
	_, err = sess.Run(context.Background(), "get id again")
	require.ErrorIs(t, err, crux.ErrOutputValidation)
}

type rawSchemaAnswer struct {
	Status string `json:"status"`
	Note   string `json:"note,omitempty"`
}

func TestRawOutputSchemaAcceptsNullForOptional(t *testing.T) {
	// Strict providers are told optional fields are nullable, so a null must validate
	// even when the schema was not built with WithOutputSchemaFrom.
	schema := (&jsonschema.Reflector{ExpandedStruct: true}).Reflect(&rawSchemaAnswer{})
	mock := cruxtest.NewMock()
	mock.Expect().ReturnText(`{"status":"ok","note":null}`)
	a, err := crux.New("raw-schema", "test-model",
		crux.WithProvider(crux.ProviderOpenAI),
		crux.WithOutputSchema(schema))
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client()))
	require.NoError(t, err)
	var res rawSchemaAnswer
	require.NoError(t, sess.RunInto(context.Background(), &res, "status"))
	require.Equal(t, "ok", res.Status)

	mock.Expect().ReturnText(`{"status":null}`)
	_, err = sess.Run(context.Background(), "status again")
	require.ErrorIs(t, err, crux.ErrOutputValidation, "required fields must stay non-null")
}

func TestResumeRepairsStoredInvalidOutput(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnText(`{}`)
	mock.Expect().ReturnText(`{"status":"ok","score":1,"note":null}`)

	strict := newMockAgent(t, mock, crux.OpenAIGPT5_6Sol, crux.WithOutputSchemaFrom[structuredAnswer]())
	session := crux.MustSession(crux.NewSession(t.Context(), strict, crux.WithHTTPClient(mock.Client())))
	_, err := session.Run(t.Context(), "extract")
	require.ErrorIs(t, err, crux.ErrOutputValidation)

	repairing := newMockAgent(t, mock, crux.OpenAIGPT5_6Sol,
		crux.WithOutputSchemaFrom[structuredAnswer](), crux.WithMaxRepairs(1), crux.WithMaxTurns(1))
	resumed := crux.MustSession(crux.NewSession(t.Context(), repairing,
		crux.WithSessionID(session.ID()), crux.WithStore(session.Store()), crux.WithHTTPClient(mock.Client())))
	out, err := resumed.Resume(t.Context())
	require.NoError(t, err)
	require.JSONEq(t, `{"status":"ok","score":1,"note":null}`, out)
	require.Contains(t, mock.Requests()[1].BodyString(), "Return corrected JSON")
	mock.AssertAllConsumed(t)
}

func TestOutputSchemaNumbersStayNumbersOnOpenAI(t *testing.T) {
	type rating struct {
		Stars int `json:"stars" jsonschema:"minimum=1,maximum=5,enum=1,enum=3,enum=5"`
	}
	mock := cruxtest.NewMock()
	mock.Expect().ReturnJSON(map[string]any{"stars": 5})
	a, err := crux.New("rater", crux.OpenAIGPT5_4Nano, crux.WithOutputSchemaFrom[rating]())
	require.NoError(t, err)
	s, err := crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client()))
	require.NoError(t, err)
	_, err = s.Run(t.Context(), "rate it")
	require.NoError(t, err)
	body := mock.Requests()[0].BodyString()
	require.Contains(t, body, `"maximum":5`)
	require.Contains(t, body, `"minimum":1`)
	require.Contains(t, body, `"enum":[1,3,5]`)
}
