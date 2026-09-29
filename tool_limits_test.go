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

type customDecoded struct{ raw map[string]any }

func (c *customDecoded) UnmarshalJSON(data []byte) error { return json.Unmarshal(data, &c.raw) }

type selfDecodingInput struct {
	Path string `json:"path"`
}

func (s *selfDecodingInput) UnmarshalJSON(data []byte) error {
	type plain selfDecodingInput
	return json.Unmarshal(data, (*plain)(s))
}

func TestToolArgsRejectUnknownKeys(t *testing.T) {
	type Embedded struct {
		Mode string `json:"mode,omitempty"`
	}
	type input struct {
		Embedded
		Path   string           `json:"path"`
		Hidden string           `json:"-"`
		Dash   string           `json:"-,omitempty"`
		Inner  *struct{ N int } `json:"inner,omitempty"`
		Extra  map[string]any   `json:"extra,omitempty"`
		Custom *customDecoded   `json:"custom,omitempty"`
	}
	reg := NewToolsRegistry()
	var got input
	RegisterToolWithRegistry(reg, "t", "t", func(ctx context.Context, in input) (string, *StateDelta, error) {
		got = in
		return "", nil, nil
	})
	RegisterToolWithRegistry(reg, "self", "s", func(ctx context.Context, in selfDecodingInput) (string, *StateDelta, error) {
		return "", nil, nil
	})
	RegisterToolWithRegistry(reg, "m", "m", func(ctx context.Context, in map[string]string) (string, *StateDelta, error) {
		return "", nil, nil
	})
	call := func(name, args string) error {
		_, _, err := reg.tools[name].invoke(t.Context(), json.RawMessage(args))
		return err
	}

	require.ErrorContains(t, call("t", `{"pаth":"decoy","path":"real"}`), `unknown argument "pаth"`)
	require.ErrorContains(t, call("t", `{"path":"a","Hidden":"x"}`), `unknown argument "Hidden"`)
	require.ErrorContains(t, call("t", `{"path":"a","inner":{"N":1,"M":2}}`), `unknown argument "inner.M"`)
	require.ErrorContains(t, call("t", `{"path":"a","nope":null}`), `unknown argument "nope"`)
	require.ErrorContains(t, call("self", `{"path":"a","other":1}`), `unknown argument "other"`)

	require.NoError(t, call("t", `{"path":"a","mode":"m","-":"d","inner":{"N":1},"extra":{"any":1},"custom":{"free":1}}`))
	require.Equal(t, "m", got.Mode)
	require.Equal(t, "d", got.Dash)
	require.NoError(t, call("m", `{"any":"key","pаth":"x"}`))
}

func TestSubAgentArgsAreValidatedStrictly(t *testing.T) {
	helper, err := New("helper", ChatModelGPT5, WithAPIKey("k"))
	require.NoError(t, err)
	parent, err := New("p", ChatModelGPT5, WithAPIKey("k"), WithSubAgent(helper, "d"))
	require.NoError(t, err)
	tool := parent.tools[len(parent.tools)-1]
	require.Equal(t, "agent_helper", tool.name)
	call := func(args string) error {
		_, _, err := tool.invoke(t.Context(), json.RawMessage(args))
		return err
	}

	require.ErrorContains(t, call(`{"task":"safe","task":"evil"}`), `argument "task" is given more than once`)
	require.ErrorContains(t, call(`{"TASK":"evil"}`), `"TASK" must be spelled "task"`)
	require.ErrorContains(t, call(`{"task":"a","note":"b"}`), `unknown argument "note"`)
	require.ErrorContains(t, call(`{"task":1}`), "invalid arguments")
	require.ErrorContains(t, call(`{"task":"  "}`), "needs a non-empty task")
	require.ErrorContains(t, call(`{}`), "invalid arguments")
}
