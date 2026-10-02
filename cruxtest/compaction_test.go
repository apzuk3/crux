package cruxtest_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/apzuk3/crux"
	"github.com/apzuk3/crux/cruxtest"
	"github.com/stretchr/testify/require"
)

type PageArgs struct {
	Page int `json:"page"`
}

// compactionSession returns a session whose read_page tool returns pages of
// 6000 bytes, each starting with "PAGE n:".
func compactionSession(t *testing.T, provider crux.Provider, mock *cruxtest.Mock, opts ...crux.AgentOption) *crux.Session {
	t.Helper()
	reg := crux.NewToolsRegistry()
	crux.RegisterToolWithRegistry(reg, "read_page", "Read a page", func(ctx context.Context, in PageArgs) (string, *crux.StateDelta, error) {
		return fmt.Sprintf("PAGE %d:", in.Page) + strings.Repeat("x", 6000), nil, nil
	})
	options := append(mock.AgentOptions(), crux.WithProvider(provider), crux.WithToolsRegistry([]string{"read_page"}, reg))
	a, err := crux.New("reader", "test-model", append(options, opts...)...)
	require.NoError(t, err)
	s, err := crux.NewSession(t.Context(), a)
	require.NoError(t, err)
	return s
}

func compactions(s *crux.Session) []crux.Compaction {
	var out []crux.Compaction
	for _, e := range s.Logs() {
		if e.Kind == crux.KindCompaction {
			out = append(out, *e.Compaction)
		}
	}
	return out
}

func TestCompactionOmitsReadToolOutputs(t *testing.T) {
	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("read_page", PageArgs{Page: 1}).WithUsage(cruxtest.TokenUsage{InputTokens: 100, OutputTokens: 10})
	mock.Expect().ReturnToolCall("read_page", PageArgs{Page: 2}).WithUsage(cruxtest.TokenUsage{InputTokens: 2000, OutputTokens: 10})
	mock.Expect().ReturnText("done")
	// 1500 tokens per page: the second page passes 80% of 4000.
	s := compactionSession(t, crux.ProviderOpenAI, mock, crux.WithContextWindow(4000))

	answer, err := s.Run(t.Context(), "read both pages")
	require.NoError(t, err)
	require.Equal(t, "done", answer)
	mock.AssertAllConsumed(t) // no summary was requested

	got := compactions(s)
	require.Len(t, got, 1)
	require.Empty(t, got[0].Summary)

	last := mock.Requests()[2].BodyString()
	require.NotContains(t, last, "PAGE 1:")
	require.Contains(t, last, "[output omitted to save context: 6007 bytes")
	require.Contains(t, last, "PAGE 2:", "the page the model has not read yet stays")

	full := 0
	for _, e := range s.Logs() {
		if e.ToolResult != nil && strings.HasPrefix(e.ToolResult.Output, "PAGE") {
			full++
		}
	}
	require.Equal(t, 2, full, "the log keeps every output")
}

func TestCompactSummarisesOlderTurns(t *testing.T) {
	// With a large known window, everything would fit; an explicit Compact
	// still summarises all but the latest exchange.
	for name, window := range map[string][]crux.AgentOption{
		"unknown window": nil,
		"large window":   {crux.WithContextWindow(200_000)},
	} {
		t.Run(name, func(t *testing.T) {
			mock := cruxtest.NewMock()
			mock.Expect().ReturnText("first answer")
			mock.Expect().ReturnText("second answer")
			mock.Expect().ReturnText("SUMMARY: an earlier exchange.").WithUsage(cruxtest.TokenUsage{InputTokens: 50, OutputTokens: 9})
			mock.Expect().ReturnText("third answer")
			s := compactionSession(t, crux.ProviderOpenAI, mock, window...)

			for _, q := range []string{"first question", "second question"} {
				_, err := s.Run(t.Context(), q)
				require.NoError(t, err)
			}
			require.NoError(t, s.Compact(t.Context()))
			answer, ok := s.FinalOutput()
			require.True(t, ok, "compaction does not hide the final answer")
			require.Equal(t, "second answer", answer)

			_, err := s.Run(t.Context(), "third question")
			require.NoError(t, err)

			summaryRequest := mock.Requests()[2].BodyString()
			require.Contains(t, summaryRequest, "first question")
			require.NotContains(t, summaryRequest, "second question", "the latest question is kept, not summarised")
			require.NotContains(t, summaryRequest, "read_page", "the summary request has no tools")

			last := mock.Requests()[3].BodyString()
			require.Contains(t, last, "SUMMARY: an earlier exchange.")
			require.NotContains(t, last, "first question")
			require.Contains(t, last, "second question")
			require.Contains(t, last, "third question")

			got := compactions(s)
			require.Len(t, got, 1)
			require.NotZero(t, got[0].Through)
			for _, e := range s.Logs() {
				if e.Kind == crux.KindCompaction {
					require.Equal(t, 50, e.Usage.InputTokens, "the summary request is counted")
				}
			}
		})
	}
}

func TestCompactWithNothingToSummarise(t *testing.T) {
	mock := cruxtest.NewMock()
	s := compactionSession(t, crux.ProviderOpenAI, mock)
	require.NoError(t, s.Compact(t.Context()))
	require.Empty(t, s.Logs())
	require.Empty(t, mock.Requests())
}

func TestContextTooLongIsRecovered(t *testing.T) {
	errors := map[crux.Provider]string{
		crux.ProviderOpenAI:    `{"error":{"message":"Your input exceeds the context window of this model.","type":"invalid_request_error","code":"context_length_exceeded"}}`,
		crux.ProviderAnthropic: `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 210000 tokens > 200000 maximum"}}`,
		crux.ProviderGoogle:    `{"error":{"code":400,"message":"The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).","status":"INVALID_ARGUMENT"}}`,
	}
	for provider, body := range errors {
		t.Run(string(provider), func(t *testing.T) {
			mock := cruxtest.NewMock()
			mock.Expect().ReturnText("first answer")
			mock.Expect().ReturnError(400, body)
			mock.Expect().ReturnText("SUMMARY of the first exchange")
			mock.Expect().ReturnText("second answer")
			s := compactionSession(t, provider, mock)

			_, err := s.Run(t.Context(), "first question")
			require.NoError(t, err)
			answer, err := s.Run(t.Context(), "second question")
			require.NoError(t, err)
			require.Equal(t, "second answer", answer)
			mock.AssertAllConsumed(t)

			retried := mock.Requests()[3].BodyString()
			require.Contains(t, retried, "SUMMARY of the first exchange")
			require.Contains(t, retried, "second question")
			require.NotContains(t, retried, "first question")

			turns := 0
			for _, e := range s.Logs() {
				if e.Kind == crux.KindTurnStarted {
					turns++
				}
			}
			require.Equal(t, 3, turns, "the failed request and its retry are both recorded")
		})
	}

	t.Run("without compaction", func(t *testing.T) {
		mock := cruxtest.NewMock()
		mock.Expect().ReturnError(400, errors[crux.ProviderOpenAI])
		s := compactionSession(t, crux.ProviderOpenAI, mock, crux.WithoutCompaction())
		_, err := s.Run(t.Context(), "question")
		require.ErrorIs(t, err, crux.ErrContextTooLong)
		require.Len(t, mock.Requests(), 1)
	})

	t.Run("nothing to compact", func(t *testing.T) {
		mock := cruxtest.NewMock()
		mock.Expect().ReturnError(400, errors[crux.ProviderOpenAI])
		s := compactionSession(t, crux.ProviderOpenAI, mock)
		_, err := s.Run(t.Context(), "a single question that is too long")
		require.ErrorIs(t, err, crux.ErrContextTooLong)
		require.Len(t, mock.Requests(), 1)
	})
}

func TestCompactionOptions(t *testing.T) {
	mock := cruxtest.NewMock()
	_, err := crux.New("a", "test-model", append(mock.AgentOptions(), crux.WithProvider(crux.ProviderOpenAI),
		crux.WithCompaction(crux.CompactAt(1.5)))...)
	require.ErrorContains(t, err, "between 0 and 1")

	plain := crux.Must(crux.New("a", crux.ClaudeHaiku4_5, crux.WithAPIKey("k")))
	defaults := crux.Must(crux.New("a", crux.ClaudeHaiku4_5, crux.WithAPIKey("k"), crux.WithCompaction()))
	tuned := crux.Must(crux.New("a", crux.ClaudeHaiku4_5, crux.WithAPIKey("k"), crux.WithCompaction(crux.CompactAt(0.6))))
	require.Equal(t, plain.ID(), defaults.ID(), "default settings keep the agent's ID")
	require.NotEqual(t, plain.ID(), tuned.ID())
}

func TestCompactionEntryRoundTrips(t *testing.T) {
	e := crux.Entry{Kind: crux.KindCompaction, Compaction: &crux.Compaction{Through: 7, Summary: "s"}}
	raw, err := json.Marshal(e)
	require.NoError(t, err)
	var back crux.Entry
	require.NoError(t, json.Unmarshal(raw, &back))
	require.Equal(t, crux.Kind(7), back.Kind, "KindCompaction keeps its reserved value")
	require.Equal(t, e.Compaction, back.Compaction)
}

func TestFailedCompactionDoesNotStopTheRun(t *testing.T) {
	const failure = `{"error":{"message":"the summary model is unavailable","type":"invalid_request_error"}}`
	start := func(mock *cruxtest.Mock) *crux.Session {
		// 3500 of a 4000-token window passes the threshold, and there are no
		// outputs to omit, so the next request starts with a summary.
		mock.Expect().ReturnText("first answer").WithUsage(cruxtest.TokenUsage{InputTokens: 3500, OutputTokens: 10})
		mock.Expect().ReturnError(400, failure)
		s := compactionSession(t, crux.ProviderOpenAI, mock, crux.WithContextWindow(4000))
		_, err := s.Run(t.Context(), "first question")
		require.NoError(t, err)
		return s
	}

	t.Run("the request is sent anyway", func(t *testing.T) {
		mock := cruxtest.NewMock()
		s := start(mock)
		mock.Expect().ReturnText("second answer")
		answer, err := s.Run(t.Context(), "second question")
		require.NoError(t, err)
		require.Equal(t, "second answer", answer)
		mock.AssertAllConsumed(t)
		require.Empty(t, compactions(s))
		require.Contains(t, mock.Requests()[2].BodyString(), "first question", "nothing was summarised")
	})

	t.Run("the failure is reported with the request's", func(t *testing.T) {
		mock := cruxtest.NewMock()
		s := start(mock)
		mock.Expect().ReturnError(400, `{"error":{"message":"bad request","type":"invalid_request_error"}}`)
		_, err := s.Run(t.Context(), "second question")
		require.ErrorContains(t, err, "bad request")
		require.ErrorContains(t, err, "the summary model is unavailable")
	})
}
