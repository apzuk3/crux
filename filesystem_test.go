package crux

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"
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
	wantFSError(t, registry, "edit_file", editFileInput{Path: "a.go", Edits: []fileEdit{}}, "edits must not be empty")
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

// withTimeout fails the test when fn does not return within d.
func withTimeout(t *testing.T, d time.Duration, fn func()) {
	t.Helper()
	done := make(chan struct{})
	go func() {
		defer close(done)
		fn()
	}()
	select {
	case <-done:
	case <-time.After(d):
		t.Fatalf("did not finish within %v", d)
	}
}

func allocatedDuring(fn func()) uint64 {
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	fn()
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc
}

func TestCleanGlob(t *testing.T) {
	for _, tt := range []struct{ in, want string }{
		{"**/**/*.go", "**/*.go"},
		{"a/**/**/**/b", "a/**/b"},
		{"**", "**"},
		{"a/*/b", "a/*/b"},
	} {
		got, err := cleanGlob(tt.in)
		if err != nil || got != tt.want {
			t.Errorf("cleanGlob(%q) = %q, %v; want %q", tt.in, got, err, tt.want)
		}
	}
	if _, err := cleanGlob(strings.Repeat("a/", fsMaxGlobSegments) + "b"); err == nil || !strings.Contains(err.Error(), "segments") {
		t.Errorf("expected segment limit error, got %v", err)
	}
	if _, err := cleanGlob(strings.Repeat("a", fsMaxGlobLength+1)); err == nil || !strings.Contains(err.Error(), "longer than") {
		t.Errorf("expected length limit error, got %v", err)
	}
	if got, err := cleanGlob(strings.Repeat("**/", 40) + "x"); err != nil || got != "**/x" {
		t.Errorf("repeated ** = %q, %v", got, err)
	}
}

func TestMatchGlobPathological(t *testing.T) {
	pattern := strings.Repeat("**/d/", 15) + "x"
	withTimeout(t, 5*time.Second, func() {
		if matchGlob(pattern, strings.Repeat("d/", 200)+"y") {
			t.Error("unexpected match")
		}
		if !matchGlob(pattern, strings.Repeat("d/", 200)+"x") {
			t.Error("expected match")
		}
	})
}

func TestFilesystemGlobPathological(t *testing.T) {
	registry, root := newFilesystemRegistry(t, nil)
	deep := filepath.Join(root, filepath.FromSlash(strings.Repeat("d/", 25)))
	if err := os.MkdirAll(deep, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(deep, "x.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	withTimeout(t, 10*time.Second, func() {
		for _, pattern := range []string{strings.Repeat("**/", 11) + "nomatch", strings.Repeat("**/d/", 6) + "nomatch"} {
			if got := mustFSTool(t, registry, "glob", globInput{Pattern: pattern}); got != "No files found" {
				t.Errorf("glob(%q) = %q", pattern, got)
			}
			got := mustFSTool(t, registry, "search_files_content", searchFilesContentInput{Query: "x", Include: pattern})
			if got != "No results found" {
				t.Errorf("search include %q = %q", pattern, got)
			}
		}
	})
}

func TestFilesystemWalkHonorsContext(t *testing.T) {
	registry, _ := newFilesystemRegistry(t, map[string]string{"a/b.txt": "x", "c.txt": "x"})
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	for name, args := range map[string]any{
		"glob":                 globInput{Pattern: "**"},
		"directory_tree":       directoryTreeInput{},
		"search_files_content": searchFilesContentInput{Query: "x"},
	} {
		raw, _ := json.Marshal(args)
		if _, _, err := registry.tools[name].invoke(ctx, raw); err == nil || !strings.Contains(err.Error(), "context canceled") {
			t.Errorf("%s with canceled context: err = %v", name, err)
		}
	}
}

func TestFilesystemReadFileLongLine(t *testing.T) {
	line := strings.Repeat("a", 8*fsMaxReadBytes)
	registry, _ := newFilesystemRegistry(t, map[string]string{
		"one.txt":   line,
		"two.txt":   line + "\nsecond\n",
		"multi.txt": strings.Repeat("é", fsMaxReadBytes),
	})

	var got string
	alloc := allocatedDuring(func() { got = mustFSTool(t, registry, "read_file", readFileInput{Path: "one.txt"}) })
	if !strings.HasPrefix(got, strings.Repeat("a", fsMaxReadBytes)+"\n[Output truncated") || !strings.Contains(got, "within line 1") {
		t.Fatalf("long line read = %q...", got[:min(len(got), 100)])
	}
	if alloc > 4*fsMaxReadBytes {
		t.Errorf("read_file of a %d byte line allocated %d bytes", len(line), alloc)
	}

	second := 2
	alloc = allocatedDuring(func() { got = mustFSTool(t, registry, "read_file", readFileInput{Path: "two.txt", Line: &second}) })
	if got != "second\n" {
		t.Fatalf("read after long line = %q", got)
	}
	if alloc > fsMaxReadBytes {
		t.Errorf("skipping a %d byte line allocated %d bytes", len(line), alloc)
	}

	got = mustFSTool(t, registry, "read_file", readFileInput{Path: "multi.txt"})
	content, _, _ := strings.Cut(got, "\n[Output truncated")
	if !utf8.ValidString(content) || len(content) > fsMaxReadBytes || len(content) < fsMaxReadBytes-4 {
		t.Fatalf("multi-byte truncation: %d bytes, valid=%v", len(content), utf8.ValidString(content))
	}
}

func TestFilesystemReadMultipleFilesLimits(t *testing.T) {
	big := strings.Repeat(strings.Repeat("x", 99)+"\n", 9000) // ~900 KB
	registry, _ := newFilesystemRegistry(t, map[string]string{
		"a.txt": "A", "b.txt": big, "c.txt": big, "d.txt": big, "e.txt": big,
	})

	got := mustFSTool(t, registry, "read_multiple_files", readMultipleFilesInput{Paths: []string{"a.txt", "./a.txt", "dir/../a.txt", "a.txt"}})
	if n := strings.Count(got, "=== "); n != 1 {
		t.Fatalf("duplicate paths read %d times: %q", n, got)
	}

	got = mustFSTool(t, registry, "read_multiple_files", readMultipleFilesInput{Paths: []string{"b.txt", "c.txt", "d.txt", "e.txt", "a.txt"}})
	if len(got) > fsMaxMultiReadBytes+1024 {
		t.Fatalf("output is %d bytes, cap %d", len(got), fsMaxMultiReadBytes)
	}
	if !strings.Contains(got, "Not read: ") || !strings.Contains(got, "a.txt") {
		t.Fatalf("missing truncation note: %q", got[max(0, len(got)-300):])
	}

	paths := make([]string, fsMaxMultiReadFiles+1)
	for i := range paths {
		paths[i] = fmt.Sprintf("f%d.txt", i)
	}
	wantFSError(t, registry, "read_multiple_files", readMultipleFilesInput{Paths: paths}, fmt.Sprintf("at most %d paths", fsMaxMultiReadFiles))
	wantFSError(t, registry, "read_multiple_files", readMultipleFilesInput{Paths: []string{}}, "must not be empty")
}

func TestFilesystemSearchOutputBounded(t *testing.T) {
	registry, _ := newFilesystemRegistry(t, map[string]string{
		"long.txt": strings.Repeat("a", 2<<20),
		"many.txt": strings.Repeat("match "+strings.Repeat("z", 150)+"\n", 5000),
	})
	got := mustFSTool(t, registry, "search_files_content", searchFilesContentInput{Query: ".*", IsRegex: true, Include: "long.txt"})
	if len(got) > fsMaxPreview+50 {
		t.Fatalf("preview of a long match is %d bytes", len(got))
	}
	got = mustFSTool(t, registry, "search_files_content", searchFilesContentInput{Query: "match"})
	body, _, ok := strings.Cut(got, "\n[Output truncated")
	if !ok || len(body) > fsMaxSearchOutput {
		t.Fatalf("search output body is %d bytes (cap %d), truncation note %v", len(body), fsMaxSearchOutput, ok)
	}
}

func TestPreviewBounded(t *testing.T) {
	line := strings.Repeat("é", 1000)
	for _, tt := range []struct{ from, to int }{{0, len(line)}, {500, 1500}, {1990, 2000}, {100, 102}} {
		got := preview(line, tt.from, tt.to)
		if len(got) > fsMaxPreview || !utf8.ValidString(got) {
			t.Errorf("preview(%d, %d) = %d bytes, valid=%v", tt.from, tt.to, len(got), utf8.ValidString(got))
		}
	}
}

func TestFilesystemNonRegularFiles(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("no FIFOs on Windows")
	}
	mkfifo, err := exec.LookPath("mkfifo")
	if err != nil {
		t.Skip("mkfifo not available")
	}
	registry, root := newFilesystemRegistry(t, map[string]string{"a.txt": "a"})
	if out, err := exec.Command(mkfifo, filepath.Join(root, "pipe")).CombinedOutput(); err != nil {
		t.Skipf("mkfifo: %v: %s", err, out)
	}
	withTimeout(t, 5*time.Second, func() {
		wantFSError(t, registry, "read_file", readFileInput{Path: "pipe"}, "not a regular file")
		wantFSError(t, registry, "edit_file", editFileInput{Path: "pipe", Edits: []fileEdit{{OldText: "a", NewText: "b"}}}, "not a regular file")
		got := mustFSTool(t, registry, "read_multiple_files", readMultipleFilesInput{Paths: []string{"pipe", "a.txt"}})
		if !strings.Contains(got, "pipe is not a regular file") || !strings.Contains(got, "=== a.txt ===\na") {
			t.Errorf("read_multiple_files = %q", got)
		}
		if got := mustFSTool(t, registry, "search_files_content", searchFilesContentInput{Query: "a"}); got != "a.txt:1:1: a" {
			t.Errorf("search = %q", got)
		}
	})
}

func TestFilesystemRootErrorHidesPath(t *testing.T) {
	registry, root := newFilesystemRegistry(t, nil)
	if err := os.RemoveAll(root); err != nil {
		t.Fatal(err)
	}
	for name, args := range map[string]any{
		"read_file":      readFileInput{Path: "a.txt"},
		"list_directory": listDirectoryInput{},
		"write_file":     writeFileInput{Path: "a.txt"},
	} {
		_, err := callFSTool(t, registry, name, args)
		if err == nil || !strings.Contains(err.Error(), "open filesystem root") {
			t.Fatalf("%s: err = %v", name, err)
		}
		if strings.Contains(err.Error(), root) || strings.Contains(err.Error(), filepath.ToSlash(root)) {
			t.Fatalf("%s: error leaks the root path: %v", name, err)
		}
	}
}

func TestIsBinary(t *testing.T) {
	boundary := []byte(strings.Repeat("a", fsBinarySniffBytes-1) + "é")[:fsBinarySniffBytes]
	for _, tt := range []struct {
		name string
		head []byte
		want bool
	}{
		{"empty", nil, false},
		{"text", []byte("hello\nworld\n"), false},
		{"control char", []byte("hello\x01world\n"), false},
		{"postscript", []byte("%!PS-Adobe-3.0\nhello\n"), false},
		{"utf8", []byte("héllo wörld ✓"), false},
		{"rune cut at window end", boundary, false},
		{"nul", []byte("abc\x00def"), true},
		{"invalid utf8", []byte("abc\xff\xfedef"), true},
		{"latin1", []byte("caf\xe9 au lait"), true},
	} {
		if got := isBinary(tt.head); got != tt.want {
			t.Errorf("isBinary(%s) = %v, want %v", tt.name, got, tt.want)
		}
	}

	registry, _ := newFilesystemRegistry(t, map[string]string{
		"late-nul.txt": strings.Repeat("x", 4000) + "\x00binary",
		"ctrl.txt":     "hello\x01world\n",
	})
	wantFSError(t, registry, "read_file", readFileInput{Path: "late-nul.txt"}, "binary")
	if got := mustFSTool(t, registry, "read_file", readFileInput{Path: "ctrl.txt"}); got != "hello\x01world\n" {
		t.Fatalf("ctrl read = %q", got)
	}
}

func TestFilesystemWalkBudget(t *testing.T) {
	old := fsMaxWalkEntries
	fsMaxWalkEntries = 50
	t.Cleanup(func() { fsMaxWalkEntries = old })

	files := map[string]string{}
	for i := range 40 {
		files[fmt.Sprintf("a/f%02d.txt", i)] = "needle"
		files[fmt.Sprintf("b/f%02d.txt", i)] = "needle"
	}
	for i := range 80 {
		files[fmt.Sprintf("flat/f%02d.txt", i)] = ""
	}
	registry, _ := newFilesystemRegistry(t, files)

	note := "Stopped after visiting 50 entries"
	if got := mustFSTool(t, registry, "directory_tree", directoryTreeInput{}); !strings.Contains(got, note) {
		t.Errorf("directory_tree = %q", got)
	}
	if got := mustFSTool(t, registry, "glob", globInput{Pattern: "**/*.txt"}); !strings.Contains(got, note) || strings.Count(got, "\n") > 50 {
		t.Errorf("glob = %q", got)
	}
	if got := mustFSTool(t, registry, "search_files_content", searchFilesContentInput{Query: "needle"}); !strings.Contains(got, note) {
		t.Errorf("search = %q", got)
	}
	if got := mustFSTool(t, registry, "list_directory", listDirectoryInput{Path: "flat"}); !strings.Contains(got, "more than 50 entries") || strings.Count(got, "FILE ") != 50 {
		t.Errorf("list_directory = %q", got)
	}
	if got := mustFSTool(t, registry, "glob", globInput{Pattern: "*.txt", Path: "a"}); strings.Contains(got, note) || strings.Count(got, "\n") != 39 {
		t.Errorf("glob within budget = %q", got)
	}
}

func TestFilesystemSearchByteBudget(t *testing.T) {
	old := fsMaxSearchBytes
	fsMaxSearchBytes = 100
	t.Cleanup(func() { fsMaxSearchBytes = old })

	registry, _ := newFilesystemRegistry(t, map[string]string{
		"a.txt": strings.Repeat("needle\n", 10),
		"b.txt": strings.Repeat("needle\n", 10),
	})
	got := mustFSTool(t, registry, "search_files_content", searchFilesContentInput{Query: "needle"})
	if !strings.HasPrefix(got, "a.txt:1:1: needle") || strings.Contains(got, "b.txt") || !strings.Contains(got, "Stopped after reading") {
		t.Fatalf("search = %q", got)
	}
}

func TestFilesystemAtomicWrites(t *testing.T) {
	registry, root := newFilesystemRegistry(t, map[string]string{"a.txt": "alpha\n", "b.txt": "b"})
	if err := os.Chmod(filepath.Join(root, "a.txt"), 0o600); err != nil {
		t.Fatal(err)
	}

	mustFSTool(t, registry, "edit_file", editFileInput{Path: "a.txt", Edits: []fileEdit{{OldText: "alpha", NewText: "beta"}}})
	mustFSTool(t, registry, "write_file", writeFileInput{Path: "b.txt", Content: "new"})
	mustFSTool(t, registry, "write_file", writeFileInput{Path: "c.txt", Content: "c"})

	info, err := os.Stat(filepath.Join(root, "a.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" && info.Mode().Perm() != 0o600 {
		t.Errorf("edit changed mode to %v", info.Mode().Perm())
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	if got := strings.Join(names, ","); got != "a.txt,b.txt,c.txt" {
		t.Errorf("directory holds %s; temporary files left behind?", got)
	}
	for name, want := range map[string]string{"a.txt": "beta\n", "b.txt": "new", "c.txt": "c"} {
		data, _ := os.ReadFile(filepath.Join(root, name))
		if string(data) != want {
			t.Errorf("%s = %q, want %q", name, data, want)
		}
	}

	wantFSError(t, registry, "write_file", writeFileInput{Path: ".", Content: "x"}, "is a directory")

	if err := os.Symlink("b.txt", filepath.Join(root, "link.txt")); err != nil {
		if runtime.GOOS == "windows" {
			t.Skipf("cannot create symlinks: %v", err)
		}
		t.Fatal(err)
	}
	// Writing through a link would change a file the approval did not name.
	wantFSError(t, registry, "write_file", writeFileInput{Path: "link.txt", Content: "via link"}, "symbolic link to b.txt")
	wantFSError(t, registry, "edit_file", editFileInput{Path: "link.txt", Edits: []fileEdit{{OldText: "new", NewText: "via link"}}}, "symbolic link to b.txt")
	if info, err := os.Lstat(filepath.Join(root, "link.txt")); err != nil || info.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("write replaced the symlink: %v", err)
	}
	if data, _ := os.ReadFile(filepath.Join(root, "b.txt")); string(data) != "new" {
		t.Fatalf("link target = %q", data)
	}
}

func TestFilesystemEditFileSizeLimit(t *testing.T) {
	registry, root := newFilesystemRegistry(t, nil)
	big := filepath.Join(root, "big.txt")
	if err := os.WriteFile(big, []byte(strings.Repeat("x", fsMaxEditBytes)+"y"), 0o644); err != nil {
		t.Fatal(err)
	}
	wantFSError(t, registry, "edit_file", editFileInput{Path: "big.txt", Edits: []fileEdit{{OldText: "y", NewText: "z"}}}, "larger than")
}

func TestFilesystemConcurrentEditsAreNotLost(t *testing.T) {
	registry, root := newFilesystemRegistry(t, map[string]string{"f.txt": "A\nB\n"})
	for range 20 {
		if err := os.WriteFile(filepath.Join(root, "f.txt"), []byte("A\nB\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		var wg sync.WaitGroup
		wg.Go(func() {
			mustFSTool(t, registry, "edit_file", editFileInput{Path: "f.txt", Edits: []fileEdit{{OldText: "A", NewText: "a"}}})
		})
		wg.Go(func() {
			mustFSTool(t, registry, "edit_file", editFileInput{Path: "f.txt", Edits: []fileEdit{{OldText: "B", NewText: "b"}}})
		})
		wg.Wait()
		if data, _ := os.ReadFile(filepath.Join(root, "f.txt")); string(data) != "a\nb\n" {
			t.Fatalf("f.txt = %q, want both edits", data)
		}
	}
}
