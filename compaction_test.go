package crux

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func transcriptEntries(n, size int) []Entry {
	var entries []Entry
	for i := range n {
		id := fmt.Sprint("call", i)
		entries = append(entries,
			Entry{Kind: KindUser, Content: []ContentPart{{Kind: ContentKindText, Text: fmt.Sprintf("question %d %s", i, strings.Repeat("q", size))}}},
			Entry{Kind: KindToolCall, ToolCall: &ToolCall{ID: id, Name: "read_page", Args: json.RawMessage(`{"page":1}`)}},
			Entry{Kind: KindToolResult, ToolResult: &ToolResult{CallID: id, Output: fmt.Sprintf("page %d %s", i, strings.Repeat("x", size))}},
		)
	}
	return entries
}

func TestTranscriptFitsBudget(t *testing.T) {
	t.Run("within budget", func(t *testing.T) {
		got := transcript("", transcriptEntries(3, 100), transcriptMax)
		require.Contains(t, got, "question 0")
		require.NotContains(t, got, "more bytes")
		require.NotContains(t, got, "left out")
	})

	t.Run("long outputs are cut", func(t *testing.T) {
		got := transcript("", transcriptEntries(1, 5000), transcriptMax)
		require.Contains(t, got, strings.Repeat("q", 5000), "messages stay whole while they fit")
		require.Contains(t, got, "… [3007 more bytes]")
	})

	t.Run("every part shrinks to a share", func(t *testing.T) {
		got := transcript("EARLIER SUMMARY", transcriptEntries(10, 5000), 20_000)
		require.LessOrEqual(t, len(got), 20_000)
		require.Contains(t, got, "EARLIER SUMMARY")
		require.Contains(t, got, "question 0", "the oldest messages are kept, only shorter")
		require.Contains(t, got, "question 9")
		require.NotContains(t, got, "left out")
	})

	t.Run("oldest entries are left out last", func(t *testing.T) {
		got := transcript("EARLIER SUMMARY", transcriptEntries(100, 5000), 10_000)
		require.LessOrEqual(t, len(got), 10_000)
		require.Contains(t, got, "EARLIER SUMMARY")
		require.Contains(t, got, "earlier messages left out]")
		require.NotContains(t, got, "question 0 ")
		require.Contains(t, got, "page 99", "the latest entries are kept")
	})

	t.Run("budget follows a small window", func(t *testing.T) {
		require.Equal(t, transcriptMax, transcriptBudget(0))
		require.Equal(t, transcriptMax, transcriptBudget(1_000_000))
		require.Equal(t, 16_000, transcriptBudget(8000))
	})
}
