package cruxtest_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/apzuk3/crux"
	"github.com/apzuk3/crux/cruxtest"
	"github.com/stretchr/testify/require"
)

type mathToolset struct{}

type mathArgs struct {
	A int `json:"a"`
	B int `json:"b"`
}

func (mathToolset) Register(reg crux.ToolsRegistry) error {
	crux.RegisterToolWithRegistry(reg, "add", "Add two numbers", func(ctx context.Context, in mathArgs) (int, *crux.StateDelta, error) {
		return in.A + in.B, nil, nil
	})
	crux.RegisterToolWithRegistry(reg, "multiply", "Multiply two numbers", func(ctx context.Context, in mathArgs) (int, *crux.StateDelta, error) {
		return in.A * in.B, nil, nil
	})
	return nil
}

func TestAddToolsetWithRegistry(t *testing.T) {
	reg := crux.NewToolsRegistry()
	require.NoError(t, crux.AddToolsetWithRegistry(reg, mathToolset{}))

	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("add", map[string]any{"a": 2, "b": 3})
	mock.Expect().ReturnToolCall("multiply", map[string]any{"a": 5, "b": 4})
	mock.Expect().ReturnText("20")

	agent, err := crux.New("calc", crux.ChatModelGPT5_6Sol,
		append(mock.AgentOptions(), crux.WithToolsRegistry([]string{"add", "multiply"}, reg))...)
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), agent)
	require.NoError(t, err)

	out, err := sess.Run(t.Context(), "(2+3)*4")
	require.NoError(t, err)
	require.Equal(t, "20", out)

	reqs := mock.Requests()
	require.Len(t, reqs, 3)
	require.Contains(t, reqs[1].BodyString(), `"5"`)
	require.Contains(t, reqs[2].BodyString(), `"20"`)
}

type failingToolset struct{}

var errBadConfig = errors.New("bad config")

func (failingToolset) Register(crux.ToolsRegistry) error { return errBadConfig }

func TestAddToolsetError(t *testing.T) {
	err := crux.AddToolsetWithRegistry(crux.NewToolsRegistry(), failingToolset{})
	require.ErrorIs(t, err, errBadConfig)
	require.True(t, strings.Contains(err.Error(), "failingToolset"), err.Error())
}

func TestFilesystemWriteNeedsApproval(t *testing.T) {
	root := t.TempDir()
	reg := crux.NewToolsRegistry()
	require.NoError(t, crux.AddToolsetWithRegistry(reg, crux.Filesystem(root)))

	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("write_file", map[string]any{"path": "notes/todo.txt", "content": "ship it"})
	mock.Expect().ReturnToolCall("read_file", map[string]any{"path": "notes/todo.txt"})
	mock.Expect().ReturnText("done")

	agent, err := crux.New("writer", crux.ChatModelGPT5_6Sol,
		append(mock.AgentOptions(), crux.WithToolsRegistry([]string{"write_file", "read_file"}, reg))...)
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), agent)
	require.NoError(t, err)

	_, err = sess.Run(t.Context(), "Write a todo")
	require.ErrorIs(t, err, crux.ErrApprovalNeeded)
	require.NoFileExists(t, filepath.Join(root, "notes", "todo.txt"))

	pending := sess.PendingApprovals()
	require.Len(t, pending, 1)
	require.NoError(t, sess.Approve(t.Context(), pending[0].ID))

	out, err := sess.Resume(t.Context())
	require.NoError(t, err)
	require.Equal(t, "done", out)

	data, err := os.ReadFile(filepath.Join(root, "notes", "todo.txt"))
	require.NoError(t, err)
	require.Equal(t, "ship it", string(data))
	require.Contains(t, mock.Requests()[2].BodyString(), "ship it")
}
