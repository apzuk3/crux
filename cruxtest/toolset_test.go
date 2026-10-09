package cruxtest_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"crux.foo"
	"crux.foo/cruxtest"
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
	}, crux.WithToolset("math"))
	crux.RegisterToolWithRegistry(reg, "multiply", "Multiply two numbers", func(ctx context.Context, in mathArgs) (int, *crux.StateDelta, error) {
		return in.A * in.B, nil, nil
	}, crux.WithToolset("math"))
	return nil
}

func TestAddToolsetWithRegistry(t *testing.T) {
	reg := crux.NewToolsRegistry()
	require.NoError(t, crux.AddToolsetWithRegistry(reg, mathToolset{}))

	mock := cruxtest.NewMock()
	mock.Expect().ReturnToolCall("add", map[string]any{"a": 2, "b": 3})
	mock.Expect().ReturnToolCall("multiply", map[string]any{"a": 5, "b": 4})
	mock.Expect().ReturnText("20")

	agent, err := crux.New("calc", crux.OpenAIGPT5_6Sol,
		append(mock.AgentOptions(), crux.WithToolsetsRegistry(reg, "math"))...)
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

	agent, err := crux.New("writer", crux.OpenAIGPT5_6Sol,
		append(mock.AgentOptions(), crux.WithToolsetsRegistry(reg, "filesystem"))...)
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

func TestWithToolsetsCombinesWithTools(t *testing.T) {
	reg := crux.NewToolsRegistry()
	require.NoError(t, crux.AddToolsetWithRegistry(reg, mathToolset{}))
	crux.RegisterToolWithRegistry(reg, "echo", "Echo text", func(ctx context.Context, in struct {
		Text string `json:"text"`
	}) (string, *crux.StateDelta, error) {
		return in.Text, nil, nil
	})

	mock := cruxtest.NewMock()
	mock.Expect().ReturnText("ok")

	agent, err := crux.New("calc", crux.OpenAIGPT5_6Sol, append(mock.AgentOptions(),
		crux.WithToolsRegistry([]string{"echo", "add"}, reg),
		crux.WithToolsetsRegistry(reg, "math"),
	)...)
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), agent)
	require.NoError(t, err)
	_, err = sess.Run(t.Context(), "hi")
	require.NoError(t, err)

	body := mock.Requests()[0].BodyString()
	for _, name := range []string{`"echo"`, `"add"`, `"multiply"`} {
		require.Contains(t, body, name)
	}
	require.Equal(t, 1, strings.Count(body, `"name":"add"`), body)

	_, err = crux.New("calc", crux.OpenAIGPT5_6Sol, append(mock.AgentOptions(), crux.WithToolsetsRegistry(reg, "nope"))...)
	require.ErrorIs(t, err, crux.ErrToolNotFound)
}

func TestFilesystemIndividualTools(t *testing.T) {
	root := t.TempDir()
	reg := crux.NewToolsRegistry()
	require.NoError(t, crux.AddToolsetWithRegistry(reg, crux.Filesystem(root)))

	mock := cruxtest.NewMock()
	mock.Expect().ReturnText("ok")

	agent, err := crux.New("reader", crux.OpenAIGPT5_6Sol, append(mock.AgentOptions(),
		crux.WithToolsRegistry([]string{crux.FsReadFile, crux.FsGlob}, reg),
	)...)
	require.NoError(t, err)

	sess, err := crux.NewSession(t.Context(), agent)
	require.NoError(t, err)
	_, err = sess.Run(t.Context(), "read something")
	require.NoError(t, err)

	body := mock.Requests()[0].BodyString()
	require.Contains(t, body, fmt.Sprintf(`"name":%q`, crux.FsReadFile))
	require.Contains(t, body, fmt.Sprintf(`"name":%q`, crux.FsGlob))
	require.NotContains(t, body, fmt.Sprintf(`"name":%q`, crux.FsWriteFile))
	require.NotContains(t, body, fmt.Sprintf(`"name":%q`, crux.FsEditFile))
}

func TestFilesystemAgentsFile(t *testing.T) {
	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, "AGENTS.md"), []byte("Run go test before you finish."), 0o644))

	reg := crux.NewToolsRegistry()
	require.NoError(t, crux.AddToolsetWithRegistry(reg, crux.Filesystem(root)))
	mock := cruxtest.NewMock()
	mock.Expect().ReturnText("one")
	mock.Expect().ReturnText("two")
	agent, err := crux.New("coder", crux.OpenAIGPT5_6Sol, append(mock.AgentOptions(),
		crux.WithInstructions("You are a coder."), crux.WithToolsetsRegistry(reg, "filesystem"))...)
	require.NoError(t, err)
	sess, err := crux.NewSession(t.Context(), agent)
	require.NoError(t, err)

	_, err = sess.Run(t.Context(), "hi")
	require.NoError(t, err)
	body := mock.Requests()[0].BodyString()
	require.Contains(t, body, "You are a coder.")
	require.Equal(t, 1, strings.Count(body, "Run go test before you finish."), "added once, though every tool carries it")

	// The file is read before each request.
	require.NoError(t, os.WriteFile(filepath.Join(root, "AGENTS.md"), []byte("Use tabs."), 0o644))
	_, err = sess.Run(t.Context(), "again")
	require.NoError(t, err)
	require.Contains(t, mock.Requests()[1].BodyString(), "Use tabs.")

	// Off with the option, and absent when there is no file.
	for _, ts := range []crux.Toolset{
		crux.Filesystem(root, crux.WithFilesystemAgentsFile(false)),
		crux.Filesystem(t.TempDir()),
	} {
		reg := crux.NewToolsRegistry()
		require.NoError(t, crux.AddToolsetWithRegistry(reg, ts))
		mock := cruxtest.NewMock()
		mock.Expect().ReturnText("ok")
		agent, err := crux.New("coder", crux.OpenAIGPT5_6Sol, append(mock.AgentOptions(),
			crux.WithToolsRegistry([]string{"read_file"}, reg))...)
		require.NoError(t, err)
		sess, err := crux.NewSession(t.Context(), agent)
		require.NoError(t, err)
		_, err = sess.Run(t.Context(), "hi")
		require.NoError(t, err)
		require.NotContains(t, mock.Requests()[0].BodyString(), "agents_md")
	}
}
