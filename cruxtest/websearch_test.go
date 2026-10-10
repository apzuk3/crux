package cruxtest_test

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"crux.foo"
	"crux.foo/cruxtest"
	"github.com/stretchr/testify/require"
)

// Gemini refuses Google Search next to function tools unless the request asks
// for server-side tool invocations in the response.
func TestGeminiSearchWithToolsIncludesServerSideInvocations(t *testing.T) {
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "lookup", "Look something up", func(ctx context.Context, in lookupArgs) (string, *crux.StateDelta, error) {
		return "found " + in.Query, nil, nil
	})
	body := func(t *testing.T, opts ...crux.AgentOption) map[string]any {
		t.Helper()
		mock := cruxtest.NewMock()
		mock.Expect().ReturnText("ok")
		a, err := crux.New("s", crux.Gemini2_5Flash, append([]crux.AgentOption{crux.WithWebSearch()}, opts...)...)
		require.NoError(t, err)
		s, err := crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client()))
		require.NoError(t, err)
		_, err = s.Run(t.Context(), "hi")
		require.NoError(t, err)
		var body map[string]any
		require.NoError(t, json.Unmarshal([]byte(mock.Requests()[0].BodyString()), &body))
		return body
	}

	with := body(t, crux.WithToolsRegistry([]string{"lookup"}, reg))
	config, _ := with["toolConfig"].(map[string]any)
	require.Equal(t, true, config["includeServerSideToolInvocations"])
	require.Len(t, with["tools"], 2)

	without := body(t)
	require.NotContains(t, without, "toolConfig")
	require.Len(t, without["tools"], 1)

	lat, lng := 48.85, 2.35
	located := body(t, crux.WithToolsRegistry([]string{"lookup"}, reg),
		crux.WithWebSearch(crux.WithUserLocation(crux.UserLocation{Latitude: &lat, Longitude: &lng})))
	config, _ = located["toolConfig"].(map[string]any)
	require.Equal(t, true, config["includeServerSideToolInvocations"])
	require.NotNil(t, config["retrievalConfig"], "the location must survive next to the flag")
}

// The server-side search call and result come back as parts the client has
// to echo on the next request; they are kept in Opaque and replayed.
func TestGeminiServerSideToolPartsAreReplayed(t *testing.T) {
	mock := cruxtest.NewMock(cruxtest.WithProvider(crux.ProviderGoogle))
	mock.Expect().ReturnRaw(http.StatusOK, []byte(`{"candidates":[{"content":{"role":"model","parts":[`+
		`{"toolCall":{"id":"tc_1","toolType":"GOOGLE_SEARCH_WEB","args":{"query":"go release"}}},`+
		`{"toolResponse":{"id":"tc_1","toolType":"GOOGLE_SEARCH_WEB","response":{"results":[{"url":"https://go.dev/doc/devel/release"}]}}},`+
		`{"text":"Go 1.27 is current."}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`))
	mock.Expect().ReturnText("you're welcome")
	a, err := crux.New("s", crux.Gemini2_5Flash, crux.WithWebSearch())
	require.NoError(t, err)
	s, err := crux.NewSession(t.Context(), a, crux.WithHTTPClient(mock.Client()))
	require.NoError(t, err)
	out, err := s.Run(t.Context(), "latest go?")
	require.NoError(t, err)
	require.Equal(t, "Go 1.27 is current.", out)
	_, err = s.Run(t.Context(), "thanks")
	require.NoError(t, err)
	second := mock.Requests()[1].BodyString()
	require.Contains(t, second, `"toolCall"`)
	require.Contains(t, second, `"toolResponse"`)
	require.Contains(t, second, "Go 1.27 is current.")
}
