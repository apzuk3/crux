package crux

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestSearchRejectsOversizedQueries(t *testing.T) {
	registry, _ := newFilesystemRegistry(t, map[string]string{"a.txt": "needle\n"})

	slow := strings.Repeat(`([\w\W]{1000})`, 20) + "X"
	wantFSError(t, registry, "search_files_content", searchFilesContentInput{Query: slow, IsRegex: true}, "too complex")
	wantFSError(t, registry, "search_files_content", searchFilesContentInput{Query: `[\w\W]{1000}`, IsRegex: true}, "too complex")
	wantFSError(t, registry, "search_files_content", searchFilesContentInput{Query: strings.Repeat("a", fsMaxQueryLength+1)}, "longer than")

	got := mustFSTool(t, registry, "search_files_content", searchFilesContentInput{Query: `ne+dle\b`, IsRegex: true})
	require.Equal(t, "a.txt:1:1: needle", got)
}

func TestRegexSearchReadsOnlyTheStartOfLongLines(t *testing.T) {
	old := fsMaxRegexLine
	fsMaxRegexLine = 1024
	t.Cleanup(func() { fsMaxRegexLine = old })

	registry, _ := newFilesystemRegistry(t, map[string]string{
		"a.txt": strings.Repeat("x", 10) + "needle" + strings.Repeat("x", 2000) + "\n" +
			strings.Repeat("x", 1500) + "needle\n",
	})
	got := mustFSTool(t, registry, "search_files_content", searchFilesContentInput{Query: "needle", IsRegex: true})
	require.True(t, strings.HasPrefix(got, "a.txt:1:11: "), got)
	require.NotContains(t, got, "a.txt:2:")
	require.Contains(t, got, "[Lines longer than 1 KB were searched only in their first 1 KB.]")

	// Plain text search is linear, so it still reads whole lines.
	got = mustFSTool(t, registry, "search_files_content", searchFilesContentInput{Query: "needle"})
	require.Contains(t, got, "a.txt:2:1501: ")
	require.NotContains(t, got, "Lines longer than")
}

func TestToolCallsFromOneTurnRunAtMostEightAtATime(t *testing.T) {
	var running, peak atomic.Int32
	slow := Tool{
		name: "slow",
		kind: toolKindTool,
		invoke: func(ctx context.Context, args json.RawMessage) (string, *StateDelta, error) {
			n := running.Add(1)
			for {
				p := peak.Load()
				if n <= p || peak.CompareAndSwap(p, n) {
					break
				}
			}
			time.Sleep(20 * time.Millisecond)
			running.Add(-1)
			return string(args), nil, nil
		},
	}
	agent := &Agent{tools: []Tool{slow}}
	var logs []Entry
	for i := range 20 {
		logs = append(logs, Entry{Kind: KindToolCall, ToolCall: &ToolCall{ID: fmt.Sprintf("c%d", i), Name: "slow", Args: json.RawMessage(fmt.Sprintf(`{"i":%d}`, i))}})
	}
	session := MustSession(NewSession(t.Context(), agent, WithSessionLogs(logs)))

	results := session.executeUnexecutedToolCalls(t.Context())
	require.Len(t, results, 20)
	for i, result := range results {
		require.Equal(t, fmt.Sprintf("c%d", i), result.ToolResult.CallID)
	}
	require.Equal(t, int32(8), peak.Load())
}
