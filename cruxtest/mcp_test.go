package cruxtest_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"crux.foo"
	"crux.foo/cruxtest"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/require"
)

type noteArgs struct {
	Text string `json:"text"`
}

// notesServer serves an MCP server with a read-only list_notes tool and an
// add_note tool, and counts the notes added.
func notesServer(t *testing.T) (string, *atomic.Int32) {
	var added atomic.Int32
	server := mcp.NewServer(&mcp.Implementation{Name: "notes"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "list_notes", Description: "List notes", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true}},
		func(ctx context.Context, req *mcp.CallToolRequest, in struct{}) (*mcp.CallToolResult, any, error) {
			return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "buy milk"}}}, nil, nil
		})
	mcp.AddTool(server, &mcp.Tool{Name: "add_note", Description: "Add a note"},
		func(ctx context.Context, req *mcp.CallToolRequest, in noteArgs) (*mcp.CallToolResult, any, error) {
			added.Add(1)
			return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "added " + in.Text}}}, nil, nil
		})
	ts := httptest.NewServer(mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return server }, nil))
	t.Cleanup(ts.Close)
	return ts.URL, &added
}

func TestMCPToolsInARun(t *testing.T) {
	url, added := notesServer(t)
	reg := crux.NewToolsRegistry()
	srv, err := crux.ConfigureMCPWithRegistry(t.Context(), reg, "notes", crux.MCPRemote(url),
		crux.WithMCPOAuth(crux.OAuthConfig{TokenStore: crux.NewFileTokenStore(t.TempDir())}))
	require.NoError(t, err)
	t.Cleanup(func() { srv.Close() })

	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("notes_list_notes", map[string]any{})
	mock.Expect().ReturnToolCall("notes_add_note", map[string]any{"text": "call mom"})
	agent := crux.Must(crux.New("assistant", crux.OpenAIGPT5_4,
		append(mock.AgentOptions(), crux.WithToolsetsRegistry(reg, "notes"))...))

	s := crux.MustSession(crux.NewSession(t.Context(), agent))
	_, err = s.Run(t.Context(), "add a note")
	require.ErrorIs(t, err, crux.ErrApprovalNeeded, "add_note is not read-only")
	require.Zero(t, added.Load())
	require.Contains(t, mock.Requests()[1].BodyString(), "buy milk")

	pending := s.PendingApprovals()
	require.Len(t, pending, 1)
	require.Equal(t, "notes_add_note", pending[0].Name)
	require.NoError(t, s.Approve(t.Context(), pending[0].ID))

	mock.Expect().ReturnText("done")
	out, err := s.Resume(t.Context())
	require.NoError(t, err)
	require.Equal(t, "done", out)
	require.EqualValues(t, 1, added.Load())
	require.Contains(t, mock.Requests()[2].BodyString(), "added call mom")
}
