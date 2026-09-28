package crux

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func newFilesystemRegistry(t *testing.T, files map[string]string) (ToolsRegistry, string) {
	t.Helper()
	root := t.TempDir()
	for name, content := range files {
		p := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	registry := NewToolsRegistry()
	if err := AddToolsetWithRegistry(registry, Filesystem(root)); err != nil {
		t.Fatal(err)
	}
	return registry, root
}

func callFSTool(t *testing.T, registry ToolsRegistry, name string, args any) (string, error) {
	t.Helper()
	tool, ok := registry.tools[name]
	if !ok {
		t.Fatalf("tool %q not registered", name)
	}
	raw, err := json.Marshal(args)
	if err != nil {
		t.Fatal(err)
	}
	out, _, err := tool.invoke(t.Context(), raw)
	return out, err
}

func mustFSTool(t *testing.T, registry ToolsRegistry, name string, args any) string {
	t.Helper()
	out, err := callFSTool(t, registry, name, args)
	if err != nil {
		t.Fatalf("%s(%v): unexpected error: %v", name, args, err)
	}
	return out
}

func wantFSError(t *testing.T, registry ToolsRegistry, name string, args any, contains string) {
	t.Helper()
	_, err := callFSTool(t, registry, name, args)
	if err == nil {
		t.Fatalf("%s(%v): expected error containing %q", name, args, contains)
	}
	if !strings.Contains(err.Error(), contains) {
		t.Fatalf("%s(%v): error %q does not contain %q", name, args, err, contains)
	}
}

func TestFilesystemRegister(t *testing.T) {
	registry, _ := newFilesystemRegistry(t, nil)

	approval := map[string]bool{
		"read_file":            false,
		"read_multiple_files":  false,
		"list_directory":       false,
		"directory_tree":       false,
		"glob":                 false,
		"search_files_content": false,
		"write_file":           true,
		"edit_file":            true,
		"create_directory":     true,
		"remove_directory":     true,
	}
	if len(registry.tools) != len(approval) {
		t.Fatalf("registered %d tools, want %d", len(registry.tools), len(approval))
	}
	for name, want := range approval {
		tool, ok := registry.tools[name]
		if !ok {
			t.Fatalf("tool %q not registered", name)
		}
		if tool.approvalNeeded != want {
			t.Errorf("%s approvalNeeded = %v, want %v", name, tool.approvalNeeded, want)
		}
		if tool.toolset != "filesystem" {
			t.Errorf("%s toolset = %q, want filesystem", name, tool.toolset)
		}
	}

	missing := filepath.Join(t.TempDir(), "missing")
	if err := AddToolsetWithRegistry(NewToolsRegistry(), Filesystem(missing)); err == nil {
		t.Fatal("expected error for missing root")
	}
	file := filepath.Join(t.TempDir(), "file.txt")
	if err := os.WriteFile(file, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := AddToolsetWithRegistry(NewToolsRegistry(), Filesystem(file)); err == nil || !strings.Contains(err.Error(), "not a directory") {
		t.Fatalf("expected not a directory error, got %v", err)
	}
}

func TestFilesystemReadFile(t *testing.T) {
	registry, _ := newFilesystemRegistry(t, map[string]string{
		"a.txt":     "one\ntwo\nthree\nfour\n",
		"empty.txt": "",
		"bin.dat":   "\x00\x01\x02binary",
		"dir/b.txt": "b",
	})
	intp := func(v int) *int { return &v }

	if got := mustFSTool(t, registry, "read_file", readFileInput{Path: "a.txt"}); got != "one\ntwo\nthree\nfour\n" {
		t.Fatalf("full read = %q", got)
	}
	if got := mustFSTool(t, registry, "read_file", readFileInput{Path: "a.txt", Line: intp(2), Limit: intp(2)}); got != "two\nthree\n" {
		t.Fatalf("range read = %q", got)
	}
	if got := mustFSTool(t, registry, "read_file", readFileInput{Path: "a.txt", Line: intp(4)}); got != "four\n" {
		t.Fatalf("tail read = %q", got)
	}
	if got := mustFSTool(t, registry, "read_file", readFileInput{Path: "a.txt", Line: intp(10)}); !strings.Contains(got, "fewer than 10 lines") {
		t.Fatalf("out of range read = %q", got)
	}
	if got := mustFSTool(t, registry, "read_file", readFileInput{Path: "empty.txt"}); !strings.Contains(got, "empty") {
		t.Fatalf("empty read = %q", got)
	}
	if got := mustFSTool(t, registry, "read_file", readFileInput{Path: "./dir/../dir/b.txt"}); got != "b" {
		t.Fatalf("cleaned path read = %q", got)
	}
	wantFSError(t, registry, "read_file", readFileInput{Path: "missing.txt"}, "not found")
	wantFSError(t, registry, "read_file", readFileInput{Path: "dir"}, "is a directory")
	wantFSError(t, registry, "read_file", readFileInput{Path: "bin.dat"}, "binary")
	wantFSError(t, registry, "read_file", readFileInput{Path: "a.txt", Line: intp(0)}, "line must be >= 1")
	wantFSError(t, registry, "read_file", readFileInput{Path: "a.txt", Limit: intp(0)}, "limit must be >= 1")
}

func TestFilesystemReadFileTruncates(t *testing.T) {
	line := strings.Repeat("x", 1023) + "\n"
	registry, _ := newFilesystemRegistry(t, map[string]string{
		"big.txt": strings.Repeat(line, 2048),
	})
	got := mustFSTool(t, registry, "read_file", readFileInput{Path: "big.txt"})
	if !strings.Contains(got, "[Output truncated") || len(got) > fsMaxReadBytes+200 {
		t.Fatalf("expected truncated output, got %d bytes", len(got))
	}
}

func TestFilesystemReadMultipleFiles(t *testing.T) {
	registry, _ := newFilesystemRegistry(t, map[string]string{"a.txt": "A", "b.txt": "B"})
	got := mustFSTool(t, registry, "read_multiple_files", readMultipleFilesInput{Paths: []string{"a.txt", "missing.txt", "b.txt"}})
	for _, want := range []string{"=== a.txt ===\nA", "=== missing.txt ===\nError: missing.txt: not found", "=== b.txt ===\nB"} {
		if !strings.Contains(got, want) {
			t.Fatalf("output %q does not contain %q", got, want)
		}
	}
}

func TestFilesystemListDirectory(t *testing.T) {
	registry, root := newFilesystemRegistry(t, map[string]string{"a.txt": "", "sub/b.txt": ""})
	if err := os.Mkdir(filepath.Join(root, "empty"), 0o755); err != nil {
		t.Fatal(err)
	}

	if got := mustFSTool(t, registry, "list_directory", listDirectoryInput{Path: "."}); got != "FILE a.txt\nDIR  empty\nDIR  sub\n" {
		t.Fatalf("list = %q", got)
	}
	if got := mustFSTool(t, registry, "list_directory", listDirectoryInput{Path: "sub"}); got != "FILE b.txt\n" {
		t.Fatalf("list sub = %q", got)
	}
	if got := mustFSTool(t, registry, "list_directory", listDirectoryInput{Path: "empty"}); !strings.Contains(got, "empty") {
		t.Fatalf("list empty = %q", got)
	}
	wantFSError(t, registry, "list_directory", listDirectoryInput{Path: "nope"}, "not found")
}

func TestFilesystemDirectoryTree(t *testing.T) {
	registry, _ := newFilesystemRegistry(t, map[string]string{
		"a.txt":         "",
		"src/main.go":   "",
		"src/pkg/x.go":  "",
		".git/HEAD":     "",
		".git/refs/foo": "",
	})
	got := mustFSTool(t, registry, "directory_tree", directoryTreeInput{Path: ""})
	want := "./\n  a.txt\n  src/\n    main.go\n    pkg/\n      x.go\n"
	if got != want {
		t.Fatalf("tree = %q, want %q", got, want)
	}

	depth := 1
	got = mustFSTool(t, registry, "directory_tree", directoryTreeInput{Path: "", MaxDepth: &depth})
	if want := "./\n  a.txt\n  src/\n"; got != want {
		t.Fatalf("tree depth 1 = %q, want %q", got, want)
	}

	zero := 0
	wantFSError(t, registry, "directory_tree", directoryTreeInput{MaxDepth: &zero}, "max_depth must be >= 1")
	wantFSError(t, registry, "directory_tree", directoryTreeInput{Path: "a.txt"}, "not a directory")
	wantFSError(t, registry, "directory_tree", directoryTreeInput{Path: "missing"}, "not found")

	got = mustFSTool(t, registry, "directory_tree", directoryTreeInput{Path: "src"})
	if want := "src/\n  main.go\n  pkg/\n    x.go\n"; got != want {
		t.Fatalf("tree src = %q, want %q", got, want)
	}
}

func TestFilesystemGlob(t *testing.T) {
	registry, _ := newFilesystemRegistry(t, map[string]string{
		"main.go":         "",
		"README.md":       "",
		"pkg/a.go":        "",
		"pkg/deep/b.go":   "",
		"pkg/deep/c.txt":  "",
		".git/objects.go": "",
	})

	for _, tt := range []struct {
		pattern, path, want string
	}{
		{pattern: "**/*.go", want: "main.go\npkg/a.go\npkg/deep/b.go"},
		{pattern: "*.go", want: "main.go"},
		{pattern: "pkg/*", want: "pkg/a.go"},
		{pattern: "pkg/**", want: "pkg/a.go\npkg/deep/b.go\npkg/deep/c.txt"},
		{pattern: "*.go", path: "pkg", want: "pkg/a.go"},
		{pattern: "**/*.rs", want: "No files found"},
	} {
		if got := mustFSTool(t, registry, "glob", globInput{Pattern: tt.pattern, Path: tt.path}); got != tt.want {
			t.Errorf("glob(%q, %q) = %q, want %q", tt.pattern, tt.path, got, tt.want)
		}
	}
	wantFSError(t, registry, "glob", globInput{Pattern: "[x"}, "invalid glob")
	wantFSError(t, registry, "glob", globInput{Pattern: ""}, "must not be empty")
}

func TestFilesystemSearchFilesContent(t *testing.T) {
	registry, _ := newFilesystemRegistry(t, map[string]string{
		"main.go":             "package main\n\nfunc main() {}\n",
		"util.go":             "package main\r\n\r\nfunc helper() {}\r\n",
		"util_test.go":        "func TestHelper() {}\n",
		"notes.txt":           "remember func\n",
		"node_modules/x.js":   "function func() {}\n",
		"bin.dat":             "\x00func",
		".git/config":         "func",
		"long.txt":            strings.Repeat("a", 500) + "NEEDLE" + strings.Repeat("b", 500) + "\n",
		"nested/deep/file.go": "func deep() {}\n",
	})

	got := mustFSTool(t, registry, "search_files_content", searchFilesContentInput{Query: "func", Include: "**/*.go", ExcludePatterns: []string{"**/*_test.go"}})
	want := "main.go:3:1: func main() {}\nnested/deep/file.go:1:1: func deep() {}\nutil.go:3:1: func helper() {}"
	if got != want {
		t.Fatalf("search = %q, want %q", got, want)
	}

	got = mustFSTool(t, registry, "search_files_content", searchFilesContentInput{Query: `func \w+\(\)`, IsRegex: true, ExcludePatterns: []string{"node_modules", "nested/*"}})
	want = "main.go:3:1: func main() {}\nutil.go:3:1: func helper() {}\nutil_test.go:1:1: func TestHelper() {}"
	if got != want {
		t.Fatalf("regex search = %q, want %q", got, want)
	}

	got = mustFSTool(t, registry, "search_files_content", searchFilesContentInput{Query: "NEEDLE"})
	if !strings.HasPrefix(got, "long.txt:1:501: ") || len(got) > fsMaxPreview+50 {
		t.Fatalf("long line search = %q", got)
	}

	if got := mustFSTool(t, registry, "search_files_content", searchFilesContentInput{Query: "absent"}); got != "No results found" {
		t.Fatalf("no match = %q", got)
	}
	wantFSError(t, registry, "search_files_content", searchFilesContentInput{Query: "(", IsRegex: true}, "invalid regular expression")
	wantFSError(t, registry, "search_files_content", searchFilesContentInput{Query: ""}, "must not be empty")
}

func TestFilesystemWriteFile(t *testing.T) {
	registry, root := newFilesystemRegistry(t, map[string]string{"old.txt": "old"})

	mustFSTool(t, registry, "write_file", writeFileInput{Path: "new/nested/file.txt", Content: "hello"})
	mustFSTool(t, registry, "write_file", writeFileInput{Path: "old.txt", Content: "new"})

	for name, want := range map[string]string{"new/nested/file.txt": "hello", "old.txt": "new"} {
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(name)))
		if err != nil {
			t.Fatal(err)
		}
		if string(data) != want {
			t.Fatalf("%s = %q, want %q", name, data, want)
		}
	}
}

func TestFilesystemEditFile(t *testing.T) {
	registry, root := newFilesystemRegistry(t, map[string]string{
		"a.go":   "alpha\nbeta\ngamma\nbeta\n",
		"crlf.c": "one\r\ntwo\r\nthree\r\n",
	})
	read := func(name string) string {
		data, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}

	mustFSTool(t, registry, "edit_file", editFileInput{Path: "a.go", Edits: []fileEdit{
		{OldText: "alpha", NewText: "ALPHA"},
		{OldText: "gamma\nbeta", NewText: "GAMMA\nBETA"},
	}})
	if got := read("a.go"); got != "ALPHA\nbeta\nGAMMA\nBETA\n" {
		t.Fatalf("edited = %q", got)
	}

	wantFSError(t, registry, "edit_file", editFileInput{Path: "a.go", Edits: []fileEdit{{OldText: "ALPHA", NewText: "x"}, {OldText: "missing", NewText: "y"}}}, "edit 2: old_text not found")
	if got := read("a.go"); got != "ALPHA\nbeta\nGAMMA\nBETA\n" {
		t.Fatalf("failed edit changed the file: %q", got)
	}
	wantFSError(t, registry, "edit_file", editFileInput{Path: "a.go", Edits: []fileEdit{{OldText: "\n", NewText: ""}}}, "appears 4 times")
	wantFSError(t, registry, "edit_file", editFileInput{Path: "a.go", Edits: []fileEdit{{OldText: "", NewText: "x"}}}, "must not be empty")
	wantFSError(t, registry, "edit_file", editFileInput{Path: "a.go"}, "edits must not be empty")
	wantFSError(t, registry, "edit_file", editFileInput{Path: "missing.go", Edits: []fileEdit{{OldText: "a", NewText: "b"}}}, "not found")

	mustFSTool(t, registry, "edit_file", editFileInput{Path: "crlf.c", Edits: []fileEdit{{OldText: "one\ntwo", NewText: "1\n2"}}})
	if got := read("crlf.c"); got != "1\r\n2\r\nthree\r\n" {
		t.Fatalf("crlf edit = %q", got)
	}
}

func TestFilesystemDirectories(t *testing.T) {
	registry, root := newFilesystemRegistry(t, map[string]string{"full/file.txt": "x", "file.txt": "x"})

	mustFSTool(t, registry, "create_directory", directoriesInput{Paths: []string{"a/b/c", "d"}})
	for _, dir := range []string{"a/b/c", "d"} {
		if info, err := os.Stat(filepath.Join(root, filepath.FromSlash(dir))); err != nil || !info.IsDir() {
			t.Fatalf("%s not created: %v", dir, err)
		}
	}

	mustFSTool(t, registry, "remove_directory", directoriesInput{Paths: []string{"a/b/c", "d"}})
	for _, dir := range []string{"a/b/c", "d"} {
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(dir))); !os.IsNotExist(err) {
			t.Fatalf("%s not removed: %v", dir, err)
		}
	}

	wantFSError(t, registry, "remove_directory", directoriesInput{Paths: []string{"full"}}, "only empty directories")
	wantFSError(t, registry, "remove_directory", directoriesInput{Paths: []string{"file.txt"}}, "not a directory")
	wantFSError(t, registry, "remove_directory", directoriesInput{Paths: []string{"."}}, "root directory")
	wantFSError(t, registry, "remove_directory", directoriesInput{Paths: []string{"missing"}}, "not found")
}

func TestFilesystemStaysInsideRoot(t *testing.T) {
	outside := t.TempDir()
	secret := filepath.Join(outside, "secret.txt")
	if err := os.WriteFile(secret, []byte("secret"), 0o644); err != nil {
		t.Fatal(err)
	}
	registry, root := newFilesystemRegistry(t, map[string]string{"inside.txt": "inside"})

	for _, name := range []string{"../secret.txt", "a/../../secret.txt", secret, filepath.ToSlash(secret)} {
		wantFSError(t, registry, "read_file", readFileInput{Path: name}, "outside the root")
		wantFSError(t, registry, "write_file", writeFileInput{Path: name, Content: "x"}, "outside the root")
	}
	wantFSError(t, registry, "list_directory", listDirectoryInput{Path: ".."}, "outside the root")
	wantFSError(t, registry, "search_files_content", searchFilesContentInput{Path: "..", Query: "secret"}, "outside the root")

	if got := mustFSTool(t, registry, "read_file", readFileInput{Path: filepath.Join(root, "inside.txt")}); got != "inside" {
		t.Fatalf("absolute path inside root = %q", got)
	}

	if err := os.Symlink(secret, filepath.Join(root, "link.txt")); err != nil {
		if runtime.GOOS == "windows" {
			t.Skipf("cannot create symlinks: %v", err)
		}
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(root, "linkdir")); err != nil {
		t.Fatal(err)
	}
	if _, err := callFSTool(t, registry, "read_file", readFileInput{Path: "link.txt"}); err == nil {
		t.Fatal("read through symlink escaped the root")
	}
	if _, err := callFSTool(t, registry, "write_file", writeFileInput{Path: "linkdir/new.txt", Content: "x"}); err == nil {
		t.Fatal("write through symlinked directory escaped the root")
	}
	if _, err := os.Stat(filepath.Join(outside, "new.txt")); !os.IsNotExist(err) {
		t.Fatal("file was written outside the root")
	}
	if got := mustFSTool(t, registry, "search_files_content", searchFilesContentInput{Query: "secret"}); got != "No results found" {
		t.Fatalf("search followed symlink out of root: %q", got)
	}
}

func TestMatchGlob(t *testing.T) {
	for _, tt := range []struct {
		pattern, name string
		want          bool
	}{
		{"*.go", "main.go", true},
		{"*.go", "pkg/main.go", false},
		{"**/*.go", "main.go", true},
		{"**/*.go", "a/b/c.go", true},
		{"a/**/c.go", "a/c.go", true},
		{"a/**/c.go", "a/x/y/c.go", true},
		{"a/**", "a/x/y", true},
		{"a/**", "b/x", false},
		{"**", "anything/at/all", true},
		{"?.txt", "a.txt", true},
		{"[ab].txt", "c.txt", false},
	} {
		if got := matchGlob(tt.pattern, tt.name); got != tt.want {
			t.Errorf("matchGlob(%q, %q) = %v, want %v", tt.pattern, tt.name, got, tt.want)
		}
	}
}
