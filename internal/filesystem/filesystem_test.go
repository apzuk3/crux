package filesystem

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func intp(n int) *int { return &n }

// testRoot writes files (slash-separated paths) under a new directory and
// returns the tools for it and an open root.
func testRoot(t *testing.T, files map[string]string) (*Tools, *os.Root) {
	t.Helper()
	dir := t.TempDir()
	for name, content := range files {
		p := filepath.Join(dir, filepath.FromSlash(name))
		require.NoError(t, os.MkdirAll(filepath.Dir(p), 0o755))
		require.NoError(t, os.WriteFile(p, []byte(content), 0o644))
	}
	root, err := os.OpenRoot(dir)
	require.NoError(t, err)
	t.Cleanup(func() { root.Close() })
	return New(dir), root
}

func TestReadText(t *testing.T) {
	longLine := strings.Repeat("x", 100000)
	tools, root := testRoot(t, map[string]string{
		"lines.txt": "a\nb\nc\n",
		"empty.txt": "",
		"utf8.txt":  "héllo\nworld\n",
		"short.txt": "ab\ncd\n",
		"long.txt":  longLine + "\ny\n",
		"bin.dat":   "a\x00b",
		"sub/f.txt": "x",
	})
	tests := []struct {
		name     string
		file     string
		line     *int
		limit    *int
		maxBytes int
		want     string
		wantErr  string
	}{
		{name: "whole file", file: "lines.txt", want: "a\nb\nc\n"},
		{name: "from a line", file: "lines.txt", line: intp(2), want: "b\nc\n"},
		{name: "line and limit", file: "lines.txt", line: intp(2), limit: intp(1), want: "b\n"},
		{name: "limit", file: "lines.txt", limit: intp(2), want: "a\nb\n"},
		{name: "line past the end", file: "lines.txt", line: intp(10), want: "No content: lines.txt has fewer than 10 lines."},
		{name: "empty", file: "empty.txt", want: "empty.txt is empty."},
		{name: "empty from a line", file: "empty.txt", line: intp(2), want: "No content: empty.txt has fewer than 2 lines."},
		{name: "fits exactly", file: "lines.txt", maxBytes: 6, want: "a\nb\nc\n"},
		{
			name: "cut within a line keeps whole runes", file: "utf8.txt", maxBytes: 2,
			want: "h\n[Output truncated at 2 bytes, within line 1. The rest of that line is not shown; use line and limit to read later lines.]",
		},
		{
			name: "cut before a line", file: "short.txt", maxBytes: 3,
			want: "ab\n\n[Output truncated at 3 bytes, before line 2. Use line and limit to read the rest.]",
		},
		{
			name: "cut within a line longer than the buffer", file: "long.txt", maxBytes: 70000,
			want: longLine[:70000] + "\n[Output truncated at 70000 bytes, within line 1. The rest of that line is not shown; use line and limit to read later lines.]",
		},
		{name: "line after a line longer than the buffer", file: "long.txt", line: intp(2), want: "y\n"},
		{name: "binary", file: "bin.dat", wantErr: "bin.dat is a binary file"},
		{name: "directory", file: "sub", wantErr: "sub is a directory; use list_directory"},
		{name: "missing", file: "missing.txt", wantErr: "missing.txt: not found"},
		{name: "outside the root", file: "../x", wantErr: "outside the root directory"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			maxBytes := tt.maxBytes
			if maxBytes == 0 {
				maxBytes = fsMaxReadBytes
			}
			out, err := tools.readText(t.Context(), root, tt.file, tt.line, tt.limit, maxBytes)
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.want, out)
		})
	}

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err := tools.readText(ctx, root, "lines.txt", nil, nil, fsMaxReadBytes)
	require.ErrorIs(t, err, context.Canceled)
}

// walkCase is one TestWalkRoot case: a walk from start (the root when empty)
// whose callback returns errAt at path at, with the expected visited paths.
type walkCase struct {
	name          string
	start         string
	budget        int
	at            string
	errAt         error
	want          []string
	wantExhausted bool
	wantErr       error
}

// callback returns a walk callback that records the visited paths in got and
// returns the case's error at its path.
func (tt walkCase) callback(got *[]string) func(p string, d fs.DirEntry) error {
	return func(p string, d fs.DirEntry) error {
		*got = append(*got, p)
		if p == tt.at {
			return tt.errAt
		}
		return nil
	}
}

func runWalkCase(t *testing.T, root *os.Root, tt walkCase) {
	t.Helper()
	if tt.budget > 0 {
		defer func(n int) { fsMaxWalkEntries = n }(fsMaxWalkEntries)
		fsMaxWalkEntries = tt.budget
	}
	start := tt.start
	if start == "" {
		start = "."
	}
	var got []string
	exhausted, err := walkRoot(t.Context(), root, start, tt.callback(&got))
	if tt.wantErr != nil {
		require.ErrorIs(t, err, tt.wantErr)
	} else {
		require.NoError(t, err)
	}
	require.Equal(t, tt.want, got)
	require.Equal(t, tt.wantExhausted, exhausted)
}

func TestWalkRoot(t *testing.T) {
	_, root := testRoot(t, map[string]string{
		"a/x.txt": "",
		"b/y.txt": "",
		"c.txt":   "",
	})
	errBoom := errors.New("boom")
	tests := []walkCase{
		{name: "all", want: []string{"a", "a/x.txt", "b", "b/y.txt", "c.txt"}},
		{name: "skip a directory", at: "a", errAt: fs.SkipDir, want: []string{"a", "b", "b/y.txt", "c.txt"}},
		{name: "skip all", at: "b", errAt: fs.SkipAll, want: []string{"a", "a/x.txt", "b"}},
		{name: "callback error", at: "a/x.txt", errAt: errBoom, want: []string{"a", "a/x.txt"}, wantErr: errBoom},
		{name: "subdirectory start", start: "a", want: []string{"a/x.txt"}},
		{name: "missing start", start: "nope", wantErr: fs.ErrNotExist},
		{name: "entry budget", budget: 3, want: []string{"a", "a/x.txt", "b"}, wantExhausted: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) { runWalkCase(t, root, tt) })
	}

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err := walkRoot(ctx, root, ".", func(p string, d fs.DirEntry) error { return nil })
	require.ErrorIs(t, err, context.Canceled)
}

func TestSearchFilesContent(t *testing.T) {
	longLine := strings.Repeat("a", 2000)
	tools, _ := testRoot(t, map[string]string{
		"src/main.go":         "package main\n\nfunc main() {}\n",
		"src/main_test.go":    "package main\n\nfunc TestMain(t *testing.T) {}\n",
		"docs/readme.md":      "Main doc\r\nhello func\r\n",
		".git/config":         "func hidden\n",
		"node_modules/lib.js": "func lib() {}\n",
		"bin.dat":             "func\x00bin",
		"long.txt":            longLine + "needle\nneedle\n",
	})
	tests := []struct {
		name        string
		in          SearchFilesContentInput
		regexLine   int
		searchBytes int64
		want        string
		wantErr     string
	}{
		{
			name: "substring",
			in:   SearchFilesContentInput{Query: "func"},
			want: "docs/readme.md:2:7: hello func\nnode_modules/lib.js:1:1: func lib() {}\nsrc/main.go:3:1: func main() {}\nsrc/main_test.go:3:1: func TestMain(t *testing.T) {}",
		},
		{
			name: "regex",
			in:   SearchFilesContentInput{Query: "(?i)^main", IsRegex: true},
			want: "docs/readme.md:1:1: Main doc",
		},
		{
			name: "include",
			in:   SearchFilesContentInput{Query: "func", Include: "**/*.go"},
			want: "src/main.go:3:1: func main() {}\nsrc/main_test.go:3:1: func TestMain(t *testing.T) {}",
		},
		{
			name: "exclude",
			in:   SearchFilesContentInput{Query: "func", ExcludePatterns: []string{"node_modules", "**/*_test.go", "./docs/"}},
			want: "src/main.go:3:1: func main() {}",
		},
		{
			name: "exclude a directory's contents",
			in:   SearchFilesContentInput{Query: "package", ExcludePatterns: []string{"src/*"}},
			want: "No results found",
		},
		{
			name: "path and include relative to it",
			in:   SearchFilesContentInput{Path: "src", Query: "package", Include: "*.go"},
			want: "src/main.go:1:1: package main\nsrc/main_test.go:1:1: package main",
		},
		{
			name: "no results",
			in:   SearchFilesContentInput{Query: "zzz"},
			want: "No results found",
		},
		{
			name:      "regex searches only the start of a long line",
			in:        SearchFilesContentInput{Query: "needle", IsRegex: true},
			regexLine: 1024,
			want:      "long.txt:2:1: needle\n[Lines longer than 1 KB were searched only in their first 1 KB.]",
		},
		{
			name:      "substring searches the whole long line",
			in:        SearchFilesContentInput{Query: "needle", Include: "long.txt"},
			regexLine: 1024,
			want:      "long.txt:1:2001: " + longLine[:40] + "needle\nlong.txt:2:1: needle",
		},
		{
			name:        "read budget",
			in:          SearchFilesContentInput{Path: "src", Query: "package"},
			searchBytes: 40,
			want:        "src/main.go:1:1: package main\n[Stopped after reading 0 MB of files; results may be incomplete. Narrow the search with path, include or exclude_patterns.]",
		},
		{
			name:        "read budget without results",
			in:          SearchFilesContentInput{Query: "zzz"},
			searchBytes: 10,
			want:        "No results found\n[Stopped after reading 0 MB of files; results may be incomplete. Narrow the search with path, include or exclude_patterns.]",
		},
		{name: "empty query", in: SearchFilesContentInput{}, wantErr: "query must not be empty"},
		{name: "long query", in: SearchFilesContentInput{Query: strings.Repeat("q", fsMaxQueryLength+1)}, wantErr: "query is longer than"},
		{name: "invalid regex", in: SearchFilesContentInput{Query: "(", IsRegex: true}, wantErr: "invalid regular expression"},
		{name: "invalid include", in: SearchFilesContentInput{Query: "x", Include: "["}, wantErr: "invalid glob pattern"},
		{name: "invalid exclude", in: SearchFilesContentInput{Query: "x", ExcludePatterns: []string{"["}}, wantErr: "invalid glob pattern"},
		{name: "missing path", in: SearchFilesContentInput{Path: "missing", Query: "x"}, wantErr: "missing: not found"},
		{name: "file path", in: SearchFilesContentInput{Path: "bin.dat", Query: "x"}, wantErr: "bin.dat is not a directory"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.regexLine > 0 {
				defer func(n int) { fsMaxRegexLine = n }(fsMaxRegexLine)
				fsMaxRegexLine = tt.regexLine
			}
			if tt.searchBytes > 0 {
				defer func(n int64) { fsMaxSearchBytes = n }(fsMaxSearchBytes)
				fsMaxSearchBytes = tt.searchBytes
			}
			out, err := tools.SearchFilesContent(t.Context(), tt.in)
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.want, out)
		})
	}

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err := tools.SearchFilesContent(ctx, SearchFilesContentInput{Query: "func"})
	require.ErrorIs(t, err, context.Canceled)
}

func TestSearchFilesContentOutputCap(t *testing.T) {
	var content strings.Builder
	for i := range 4000 {
		fmt.Fprintf(&content, "match %d\n", i)
	}
	tools, _ := testRoot(t, map[string]string{"many.txt": content.String()})
	out, err := tools.SearchFilesContent(t.Context(), SearchFilesContentInput{Query: "match"})
	require.NoError(t, err)
	note := "\n[Output truncated. Narrow the search with a more specific query, path, include or exclude_patterns.]"
	require.True(t, strings.HasSuffix(out, note), out[max(len(out)-200, 0):])
	require.LessOrEqual(t, len(out)-len(note), fsMaxSearchOutput)
	require.True(t, strings.HasPrefix(out, "many.txt:1:1: match 0\nmany.txt:2:1: match 1\n"))
}
