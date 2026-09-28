package crux

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"unicode/utf8"
)

const (
	fsMaxReadBytes     = 1 << 20  // read_file output cap
	fsMaxEntries       = 1000     // list_directory, directory_tree and glob result cap
	fsMaxSearchOutput  = 64 << 10 // search_files_content output cap
	fsMaxSearchFile    = 10 << 20 // files larger than this are skipped by search
	fsBinarySniffBytes = 512
	fsMaxPreview       = 200
)

// Filesystem returns a toolset of file tools confined to root. Paths the model
// passes are relative to root (absolute paths are accepted when they are
// inside it), and nothing outside root can be reached, including through
// ".." or symlinks. Paths use forward slashes on every platform.
//
// Read-only tools: read_file, read_multiple_files, list_directory,
// directory_tree, glob, search_files_content.
// Tools that change files need approval: write_file, edit_file,
// create_directory, remove_directory.
//
// Register fails when root is not an existing directory.
func Filesystem(root string) Toolset {
	return &filesystemToolset{root: root}
}

type filesystemToolset struct {
	root string
}

func (f *filesystemToolset) Register(registry ToolsRegistry) error {
	root, err := filepath.Abs(f.root)
	if err != nil {
		return fmt.Errorf("filesystem root %q: %w", f.root, err)
	}
	info, err := os.Stat(root)
	if err != nil {
		return fmt.Errorf("filesystem root: %w", err)
	}
	if !info.IsDir() {
		return fmt.Errorf("filesystem root %q is not a directory", f.root)
	}

	t := &fsTools{root: root}
	approval := WithApprovalNeeded(true)

	registerFSTool(registry, "read_file", "Read a text file. The whole file is returned unless line (1-based start line) and limit (maximum number of lines) select a range.", t.readFile)
	registerFSTool(registry, "read_multiple_files", "Read several text files at once. Prefer this over sequential read_file calls.", t.readMultipleFiles)
	registerFSTool(registry, "list_directory", "List the files and directories directly inside a directory.", t.listDirectory)
	registerFSTool(registry, "directory_tree", "Show a recursive tree of files and directories.", t.directoryTree)
	registerFSTool(registry, "glob", "Find files whose path matches a glob pattern such as **/*.go or src/*.ts. ** matches any number of directories.", t.glob)
	registerFSTool(registry, "search_files_content", "Search file contents for text or a regular expression. Returns matches as path:line:column: text.", t.searchFilesContent)
	registerFSTool(registry, "write_file", "Create a file, or completely overwrite an existing one. Missing parent directories are created.", t.writeFile, approval)
	registerFSTool(registry, "edit_file", "Edit a text file by replacing exact text. Each old_text must appear exactly once in the file; include enough surrounding text to make it unique. Edits are applied in order and either all succeed or none are written.", t.editFile, approval)
	registerFSTool(registry, "create_directory", "Create one or more directories, including missing parents.", t.createDirectory, approval)
	registerFSTool(registry, "remove_directory", "Remove one or more empty directories.", t.removeDirectory, approval)

	return nil
}

func registerFSTool[In any](registry ToolsRegistry, name, description string, fn func(context.Context, In) (string, error), opts ...ToolOption) {
	RegisterToolWithRegistry(registry, name, description, func(ctx context.Context, in In) (string, *StateDelta, error) {
		out, err := fn(ctx, in)
		return out, nil, err
	}, opts...)
}

type fsTools struct {
	root string
}

// open opens the root for one tool call. os.Root keeps every operation inside
// the root directory, including when symlinks point elsewhere.
func (t *fsTools) open() (*os.Root, error) {
	root, err := os.OpenRoot(t.root)
	if err != nil {
		return nil, fmt.Errorf("open filesystem root: %w", err)
	}
	return root, nil
}

// resolve turns a model-supplied path into a clean path relative to the root,
// using the OS separator.
func (t *fsTools) resolve(name string) (string, error) {
	if name == "" {
		name = "."
	}
	p := filepath.FromSlash(name)
	if filepath.IsAbs(p) || strings.HasPrefix(p, string(filepath.Separator)) {
		abs, err := filepath.Abs(p)
		if err != nil {
			return "", fmt.Errorf("path %q: %w", name, err)
		}
		p, err = filepath.Rel(t.root, abs)
		if err != nil {
			return "", fmt.Errorf("path %q is outside the root directory", name)
		}
	}
	p = filepath.Clean(p)
	if p == ".." || strings.HasPrefix(p, ".."+string(filepath.Separator)) || filepath.VolumeName(p) != "" {
		return "", fmt.Errorf("path %q is outside the root directory", name)
	}
	return p, nil
}

// display formats a root-relative path for the model.
func display(rel string) string {
	return filepath.ToSlash(rel)
}

func fsError(name string, err error) error {
	if errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("%s: not found", name)
	}
	return fmt.Errorf("%s: %w", name, err)
}

type readFileInput struct {
	Path  string `json:"path" description:"File to read"`
	Line  *int   `json:"line,omitempty" description:"1-based line to start reading from"`
	Limit *int   `json:"limit,omitempty" description:"Maximum number of lines to read"`
}

func (t *fsTools) readFile(ctx context.Context, in readFileInput) (string, error) {
	if in.Line != nil && *in.Line < 1 {
		return "", fmt.Errorf("line must be >= 1, got %d", *in.Line)
	}
	if in.Limit != nil && *in.Limit < 1 {
		return "", fmt.Errorf("limit must be >= 1, got %d", *in.Limit)
	}
	root, err := t.open()
	if err != nil {
		return "", err
	}
	defer root.Close()

	return t.readText(root, in.Path, in.Line, in.Limit)
}

func (t *fsTools) readText(root *os.Root, name string, line, limit *int) (string, error) {
	rel, err := t.resolve(name)
	if err != nil {
		return "", err
	}
	info, err := root.Stat(rel)
	if err != nil {
		return "", fsError(name, err)
	}
	if info.IsDir() {
		return "", fmt.Errorf("%s is a directory; use list_directory", name)
	}
	f, err := root.Open(rel)
	if err != nil {
		return "", fsError(name, err)
	}
	defer f.Close()

	reader := bufio.NewReader(f)
	head, _ := reader.Peek(fsBinarySniffBytes)
	if isBinary(head) {
		return "", fmt.Errorf("%s is a binary file", name)
	}

	start, maxLines := 1, -1
	if line != nil {
		start = *line
	}
	if limit != nil {
		maxLines = *limit
	}

	var out strings.Builder
	truncated := false
	current := 1
	for {
		if maxLines >= 0 && current-start >= maxLines {
			break
		}
		chunk, readErr := reader.ReadString('\n')
		if chunk != "" && current >= start {
			if out.Len()+len(chunk) > fsMaxReadBytes {
				truncated = true
				break
			}
			out.WriteString(chunk)
		}
		if strings.HasSuffix(chunk, "\n") {
			current++
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return "", fsError(name, readErr)
		}
	}

	switch {
	case truncated:
		fmt.Fprintf(&out, "\n[Output truncated at %d bytes, before line %d. Use line and limit to read the rest.]", fsMaxReadBytes, current)
	case out.Len() == 0 && start > 1:
		return fmt.Sprintf("No content: %s has fewer than %d lines.", name, start), nil
	case out.Len() == 0:
		return fmt.Sprintf("%s is empty.", name), nil
	}
	return out.String(), nil
}

type readMultipleFilesInput struct {
	Paths []string `json:"paths" description:"Files to read"`
}

func (t *fsTools) readMultipleFiles(ctx context.Context, in readMultipleFilesInput) (string, error) {
	root, err := t.open()
	if err != nil {
		return "", err
	}
	defer root.Close()

	var out strings.Builder
	for _, name := range in.Paths {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		content, err := t.readText(root, name, nil, nil)
		if err != nil {
			content = "Error: " + err.Error()
		}
		fmt.Fprintf(&out, "=== %s ===\n%s\n\n", name, content)
	}
	return out.String(), nil
}

type listDirectoryInput struct {
	Path string `json:"path" description:"Directory to list"`
}

func (t *fsTools) listDirectory(ctx context.Context, in listDirectoryInput) (string, error) {
	rel, err := t.resolve(in.Path)
	if err != nil {
		return "", err
	}
	root, err := t.open()
	if err != nil {
		return "", err
	}
	defer root.Close()

	entries, err := fs.ReadDir(root.FS(), display(rel))
	if err != nil {
		return "", fsError(in.Path, err)
	}
	if len(entries) == 0 {
		return fmt.Sprintf("Directory is empty: %s", display(rel)), nil
	}

	var out strings.Builder
	for i, entry := range entries {
		if i == fsMaxEntries {
			fmt.Fprintf(&out, "[Output truncated: showing %d of %d entries.]\n", fsMaxEntries, len(entries))
			break
		}
		if entry.IsDir() {
			fmt.Fprintf(&out, "DIR  %s\n", entry.Name())
		} else {
			fmt.Fprintf(&out, "FILE %s\n", entry.Name())
		}
	}
	return out.String(), nil
}

type directoryTreeInput struct {
	Path     string `json:"path" description:"Directory to show"`
	MaxDepth *int   `json:"max_depth,omitempty" description:"How many directory levels to descend; unlimited when omitted"`
}

func (t *fsTools) directoryTree(ctx context.Context, in directoryTreeInput) (string, error) {
	if in.MaxDepth != nil && *in.MaxDepth < 1 {
		return "", fmt.Errorf("max_depth must be >= 1, got %d", *in.MaxDepth)
	}
	rel, err := t.resolve(in.Path)
	if err != nil {
		return "", err
	}
	root, err := t.open()
	if err != nil {
		return "", err
	}
	defer root.Close()

	if info, err := root.Stat(rel); err != nil {
		return "", fsError(in.Path, err)
	} else if !info.IsDir() {
		return "", fmt.Errorf("%s is not a directory", in.Path)
	}

	start := display(rel)
	var out strings.Builder
	fmt.Fprintf(&out, "%s/\n", start)
	count := 0
	err = fs.WalkDir(root.FS(), start, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			if p == start {
				return err
			}
			return nil
		}
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		if p == start {
			return nil
		}
		if d.IsDir() && d.Name() == ".git" {
			return fs.SkipDir
		}
		if count == fsMaxEntries {
			fmt.Fprintf(&out, "[Output truncated at %d entries. Use a deeper path or max_depth.]\n", fsMaxEntries)
			return fs.SkipAll
		}
		count++

		depth := strings.Count(relTo(start, p), "/") + 1
		name := d.Name()
		if d.IsDir() {
			name += "/"
		}
		fmt.Fprintf(&out, "%s%s\n", strings.Repeat("  ", depth), name)
		if d.IsDir() && in.MaxDepth != nil && depth >= *in.MaxDepth {
			return fs.SkipDir
		}
		return nil
	})
	if err != nil {
		return "", fsError(in.Path, err)
	}
	return out.String(), nil
}

type globInput struct {
	Pattern string `json:"pattern" description:"Glob pattern relative to path, for example **/*.go"`
	Path    string `json:"path,omitempty" description:"Directory to search from; defaults to the root"`
}

func (t *fsTools) glob(ctx context.Context, in globInput) (string, error) {
	pattern := strings.TrimPrefix(filepath.ToSlash(in.Pattern), "./")
	if err := validateGlob(pattern); err != nil {
		return "", err
	}
	rel, err := t.resolve(in.Path)
	if err != nil {
		return "", err
	}
	root, err := t.open()
	if err != nil {
		return "", err
	}
	defer root.Close()

	start := display(rel)
	var matches []string
	truncated := false
	err = fs.WalkDir(root.FS(), start, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			if p == start {
				return err
			}
			return nil
		}
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		if p == start {
			return nil
		}
		if d.IsDir() {
			if d.Name() == ".git" {
				return fs.SkipDir
			}
			return nil
		}
		if matchGlob(pattern, relTo(start, p)) {
			if len(matches) == fsMaxEntries {
				truncated = true
				return fs.SkipAll
			}
			matches = append(matches, p)
		}
		return nil
	})
	if err != nil {
		return "", fsError(in.Path, err)
	}
	if len(matches) == 0 {
		return "No files found", nil
	}
	out := strings.Join(matches, "\n")
	if truncated {
		out += fmt.Sprintf("\n[Output truncated at %d files. Use a narrower pattern.]", fsMaxEntries)
	}
	return out, nil
}

type searchFilesContentInput struct {
	Path            string   `json:"path,omitempty" description:"Directory to search from; defaults to the root"`
	Query           string   `json:"query" description:"Text or regular expression to search for"`
	IsRegex         bool     `json:"is_regex,omitempty" description:"Treat query as a regular expression (Go RE2 syntax)"`
	Include         string   `json:"include,omitempty" description:"Only search files whose path matches this glob, for example **/*.go"`
	ExcludePatterns []string `json:"exclude_patterns,omitempty" description:"Glob patterns for files or directories to skip, for example node_modules or **/*_test.go"`
}

func (t *fsTools) searchFilesContent(ctx context.Context, in searchFilesContentInput) (string, error) {
	if in.Query == "" {
		return "", errors.New("query must not be empty")
	}
	include := strings.TrimPrefix(filepath.ToSlash(in.Include), "./")
	if include != "" {
		if err := validateGlob(include); err != nil {
			return "", err
		}
	}
	excludes := make([]string, 0, len(in.ExcludePatterns))
	for _, pattern := range in.ExcludePatterns {
		pattern = strings.TrimSuffix(strings.TrimPrefix(filepath.ToSlash(pattern), "./"), "/")
		if err := validateGlob(pattern); err != nil {
			return "", err
		}
		excludes = append(excludes, pattern)
	}
	match := func(line string) (int, int, bool) {
		i := strings.Index(line, in.Query)
		return i, i + len(in.Query), i >= 0
	}
	if in.IsRegex {
		re, err := regexp.Compile(in.Query)
		if err != nil {
			return "", fmt.Errorf("invalid regular expression: %w", err)
		}
		match = func(line string) (int, int, bool) {
			loc := re.FindStringIndex(line)
			if loc == nil {
				return 0, 0, false
			}
			return loc[0], loc[1], true
		}
	}

	rel, err := t.resolve(in.Path)
	if err != nil {
		return "", err
	}
	root, err := t.open()
	if err != nil {
		return "", err
	}
	defer root.Close()

	fsys := root.FS()
	start := display(rel)
	var out strings.Builder
	matches := 0
	truncated := false
	err = fs.WalkDir(fsys, start, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			if p == start {
				return err
			}
			return nil
		}
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		if p == start {
			return nil
		}
		relPath := relTo(start, p)
		if d.IsDir() && d.Name() == ".git" || excluded(excludes, relPath, d.Name()) {
			if d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if d.IsDir() || include != "" && !matchGlob(include, relPath) {
			return nil
		}

		info, err := fs.Stat(fsys, p)
		if err != nil || !info.Mode().IsRegular() || info.Size() > fsMaxSearchFile {
			return nil
		}
		content, err := fs.ReadFile(fsys, p)
		if err != nil || isBinary(content[:min(len(content), fsBinarySniffBytes)]) {
			return nil
		}

		lineNum := 0
		for line := range strings.SplitSeq(string(content), "\n") {
			lineNum++
			line = strings.TrimSuffix(line, "\r")
			from, to, ok := match(line)
			if !ok {
				continue
			}
			if matches > 0 {
				out.WriteByte('\n')
			}
			fmt.Fprintf(&out, "%s:%d:%d: %s", p, lineNum, from+1, preview(line, from, to))
			matches++
			if out.Len() >= fsMaxSearchOutput {
				truncated = true
				return fs.SkipAll
			}
		}
		return nil
	})
	if err != nil {
		return "", fsError(in.Path, err)
	}
	if matches == 0 {
		return "No results found", nil
	}
	if truncated {
		out.WriteString("\n[Output truncated. Narrow the search with a more specific query, path, include or exclude_patterns.]")
	}
	return out.String(), nil
}

type writeFileInput struct {
	Path    string `json:"path" description:"File to write"`
	Content string `json:"content" description:"Full file content"`
}

func (t *fsTools) writeFile(ctx context.Context, in writeFileInput) (string, error) {
	rel, err := t.resolve(in.Path)
	if err != nil {
		return "", err
	}
	root, err := t.open()
	if err != nil {
		return "", err
	}
	defer root.Close()

	if dir := filepath.Dir(rel); dir != "." {
		if err := root.MkdirAll(dir, 0o755); err != nil {
			return "", fsError(in.Path, err)
		}
	}
	if err := root.WriteFile(rel, []byte(in.Content), 0o644); err != nil {
		return "", fsError(in.Path, err)
	}
	return fmt.Sprintf("Wrote %d bytes to %s", len(in.Content), display(rel)), nil
}

type fileEdit struct {
	OldText string `json:"old_text" description:"Exact text to replace; must appear exactly once in the file"`
	NewText string `json:"new_text" description:"Replacement text"`
}

type editFileInput struct {
	Path  string     `json:"path" description:"File to edit"`
	Edits []fileEdit `json:"edits" description:"Edits to apply in order"`
}

func (t *fsTools) editFile(ctx context.Context, in editFileInput) (string, error) {
	if len(in.Edits) == 0 {
		return "", errors.New("edits must not be empty")
	}
	rel, err := t.resolve(in.Path)
	if err != nil {
		return "", err
	}
	root, err := t.open()
	if err != nil {
		return "", err
	}
	defer root.Close()

	data, err := root.ReadFile(rel)
	if err != nil {
		return "", fsError(in.Path, err)
	}
	content := string(data)
	crlf := strings.Contains(content, "\r\n")

	for i, edit := range in.Edits {
		oldText, newText := edit.OldText, edit.NewText
		if crlf {
			oldText, newText = toCRLF(oldText), toCRLF(newText)
		}
		if oldText == "" {
			return "", fmt.Errorf("edit %d: old_text must not be empty", i+1)
		}
		switch n := strings.Count(content, oldText); n {
		case 0:
			return "", fmt.Errorf("edit %d: old_text not found in %s", i+1, in.Path)
		case 1:
			content = strings.Replace(content, oldText, newText, 1)
		default:
			return "", fmt.Errorf("edit %d: old_text appears %d times in %s; include more surrounding text to make it unique", i+1, n, in.Path)
		}
	}

	if err := root.WriteFile(rel, []byte(content), 0o644); err != nil {
		return "", fsError(in.Path, err)
	}
	return fmt.Sprintf("Applied %d edit(s) to %s", len(in.Edits), display(rel)), nil
}

type directoriesInput struct {
	Paths []string `json:"paths" description:"Directories"`
}

func (t *fsTools) createDirectory(ctx context.Context, in directoriesInput) (string, error) {
	root, err := t.open()
	if err != nil {
		return "", err
	}
	defer root.Close()

	var out []string
	for _, name := range in.Paths {
		rel, err := t.resolve(name)
		if err != nil {
			return "", err
		}
		if err := root.MkdirAll(rel, 0o755); err != nil {
			return "", fsError(name, err)
		}
		out = append(out, "Created directory "+display(rel))
	}
	return strings.Join(out, "\n"), nil
}

func (t *fsTools) removeDirectory(ctx context.Context, in directoriesInput) (string, error) {
	root, err := t.open()
	if err != nil {
		return "", err
	}
	defer root.Close()

	var out []string
	for _, name := range in.Paths {
		rel, err := t.resolve(name)
		if err != nil {
			return "", err
		}
		if rel == "." {
			return "", errors.New("cannot remove the root directory")
		}
		info, err := root.Lstat(rel)
		if err != nil {
			return "", fsError(name, err)
		}
		if !info.IsDir() {
			return "", fmt.Errorf("%s is not a directory", name)
		}
		if err := root.Remove(rel); err != nil {
			return "", fmt.Errorf("remove %s (only empty directories can be removed): %w", name, err)
		}
		out = append(out, "Removed directory "+display(rel))
	}
	return strings.Join(out, "\n"), nil
}

// relTo returns p relative to the walk start; both use forward slashes.
func relTo(start, p string) string {
	if start == "." {
		return p
	}
	return strings.TrimPrefix(p, start+"/")
}

func excluded(patterns []string, relPath, base string) bool {
	for _, pattern := range patterns {
		if matchGlob(pattern, relPath) || matchGlob(pattern, base) {
			return true
		}
		if dir, ok := strings.CutSuffix(pattern, "/*"); ok && matchGlob(dir, relPath) {
			return true
		}
	}
	return false
}

func validateGlob(pattern string) error {
	if pattern == "" {
		return errors.New("pattern must not be empty")
	}
	for _, segment := range strings.Split(pattern, "/") {
		if _, err := path.Match(segment, ""); err != nil {
			return fmt.Errorf("invalid glob pattern %q: %w", pattern, err)
		}
	}
	return nil
}

// matchGlob matches a slash-separated path against a glob pattern in which
// "**" matches any number of path segments.
func matchGlob(pattern, name string) bool {
	return matchSegments(strings.Split(pattern, "/"), strings.Split(name, "/"))
}

func matchSegments(pattern, name []string) bool {
	for len(pattern) > 0 {
		if pattern[0] == "**" {
			for i := range len(name) + 1 {
				if matchSegments(pattern[1:], name[i:]) {
					return true
				}
			}
			return false
		}
		if len(name) == 0 {
			return false
		}
		if ok, _ := path.Match(pattern[0], name[0]); !ok {
			return false
		}
		pattern, name = pattern[1:], name[1:]
	}
	return len(name) == 0
}

func isBinary(head []byte) bool {
	if len(head) == 0 {
		return false
	}
	if bytes.IndexByte(head, 0) >= 0 {
		return true
	}
	return !strings.HasPrefix(http.DetectContentType(head), "text/")
}

func preview(line string, from, to int) string {
	if len(line) <= fsMaxPreview {
		return line
	}
	start := max(from-40, 0)
	end := min(max(to+40, start+fsMaxPreview), len(line))
	for start > 0 && !utf8.RuneStart(line[start]) {
		start--
	}
	for end < len(line) && !utf8.RuneStart(line[end]) {
		end++
	}
	return line[start:end]
}

func toCRLF(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "\r\n", "\n"), "\n", "\r\n")
}
