package cruxtest_test

import (
	"encoding/json"
	"testing"

	"crux.foo"
	"crux.foo/cruxtest"
	"github.com/stretchr/testify/require"
)

func TestAnthropicRequestsCacheThePrompt(t *testing.T) {
	type request struct {
		CacheControl *struct {
			Type string `json:"type"`
		} `json:"cache_control"`
		System []struct {
			CacheControl *json.RawMessage `json:"cache_control"`
		} `json:"system"`
		Tools []struct {
			CacheControl *json.RawMessage `json:"cache_control"`
		} `json:"tools"`
	}
	decode := func(t *testing.T, mock *cruxtest.Mock) request {
		t.Helper()
		var req request
		require.NoError(t, json.Unmarshal([]byte(mock.Requests()[0].BodyString()), &req))
		require.NotNil(t, req.CacheControl, "top-level cache_control")
		require.Equal(t, "ephemeral", req.CacheControl.Type)
		return req
	}

	t.Run("system prompt", func(t *testing.T) {
		mock := cruxtest.NewMock()
		mock.Expect().ReturnText("ok")
		s := rulesSession(t, crux.ProviderAnthropic, crux.ClaudeHaiku4_5, mock, crux.WithInstructions("Be brief."))
		_, err := s.Run(t.Context(), "hi")
		require.NoError(t, err)

		req := decode(t, mock)
		require.Len(t, req.System, 1)
		require.NotNil(t, req.System[0].CacheControl)
		require.Nil(t, req.Tools[len(req.Tools)-1].CacheControl, "one fixed breakpoint is enough")
	})

	t.Run("tools only", func(t *testing.T) {
		mock := cruxtest.NewMock()
		mock.Expect().ReturnText("ok")
		s := rulesSession(t, crux.ProviderAnthropic, crux.ClaudeHaiku4_5, mock)
		_, err := s.Run(t.Context(), "hi")
		require.NoError(t, err)

		req := decode(t, mock)
		require.Empty(t, req.System)
		require.NotNil(t, req.Tools[len(req.Tools)-1].CacheControl)
	})
}
