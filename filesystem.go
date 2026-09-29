package crux

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"
	"unicode/utf8"
)

const (
	fsMaxReadBytes       = 1 << 20  // read_file output cap
	fsMaxMultiReadBytes  = 2 << 20  // read_multiple_files total output cap
	fsMaxMultiReadFiles  = 50       // read_multiple_files path cap
	fsMaxEntries         = 1000     // list_directory, directory_tree and glob result cap
	fsMaxSearchOutput    = 64 << 10 // search_files_content output cap
	fsMaxSearchFile      = 10 << 20 // files larger than this are skipped by search
	fsMaxEditBytes       = 10 << 20 // edit_file size cap
	fsBinarySniffBytes   = 8 << 10
	fsMaxPreview         = 200
	fsMaxGlobSegments    = 32
	fsMaxGlobLength      = 1024
	fsCtxCheckIterations = 256
)

// Work budgets per call; variables so tests can lower them.
var (
	fsMaxWalkEntries       = 20000     // entries one call may visit
	fsMaxSearchBytes int64 = 100 << 20 // bytes search_files_content may read
)

const filesystemToolsetName = "filesystem"

var (
	errIsDir      = errors.New("is a directory")
	errNotRegular = errors.New("not a regular file")
	errReadOnly   = errors.New("file is read-only")
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
// Read-only tools run without approval, so the model can read any file under
// root, including secrets such as .env files or private keys, and its content
// is sent to the provider. Scope root narrowly to the files the agent needs.
//
// Output and work per call are bounded: large files, long listings and
// searches are cut off with a note telling the model how to narrow the
// request.
//
// The tools belong to the "filesystem" toolset, so WithToolsets("filesystem")
// gives an agent all of them. Register fails when root is not an existing
// directory.
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
	registerFSTool(registry, "read_multiple_files", fmt.Sprintf("Read several text files at once (at most %d). Prefer this over sequential read_file calls.", fsMaxMultiReadFiles), t.readMultipleFiles)
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
	opts = append([]ToolOption{WithToolset(filesystemToolsetName)}, opts...)
	RegisterToolWithRegistry(registry, name, description, func(ctx context.Context, in In) (string, *StateDelta, error) {
		out, err := fn(ctx, in)
		return out, nil, err
	}, opts...)
}

type fsTools struct {
	root string
	// writeMu runs the tools that change files one at a time. Calls from one
	// model turn run concurrently, and two edits of the same file would
	// otherwise both read the old content and one would be lost.
	writeMu sync.Mutex
}

// open opens the root for one tool call. os.Root keeps every operation inside
// the root directory, including when symlinks point elsewhere. The error does
// not name the host path, because it is sent to the model.
func (t *fsTools) open() (*os.Root, error) {
	root, err := os.OpenRoot(t.root)
	if err != nil {
		var pathErr *fs.PathError
		if errors.As(err, &pathErr) {
			err = pathErr.Err
		}
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
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return fmt.Errorf("%s: not found", name)
	case errors.Is(err, errIsDir):
		return fmt.Errorf("%s is a directory", name)
	case errors.Is(err, errNotRegular):
		return fmt.Errorf("%s is not a regular file", name)
	}
	return fmt.Errorf("%s: %w", name, err)
}

// openRegular opens a regular file for reading. Directories, FIFOs, devices
// and sockets are rejected before opening, and again after opening in case
// the path was replaced in between.
func openRegular(root *os.Root, rel string) (*os.File, fs.FileInfo, error) {
	info, err := root.Stat(rel)
	if err != nil {
		return nil, nil, err
	}
	if info.IsDir() {
		return nil, nil, errIsDir
	}
	if !info.Mode().IsRegular() {
		return nil, nil, errNotRegular
	}
	f, err := root.Open(rel)
	if err != nil {
		return nil, nil, err
	}
	info, err = f.Stat()
	if err != nil {
		f.Close()
		return nil, nil, err
	}
	if !info.Mode().IsRegular() {
		f.Close()
		return nil, nil, errNotRegular
	}
	return f, info, nil
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

	return t.readText(ctx, root, in.Path, in.Line, in.Limit, fsMaxReadBytes)
}

// readText reads a text file, returning at most maxBytes of content. Memory
// use is bounded by maxBytes plus a small buffer, however long the lines are.
func (t *fsTools) readText(ctx context.Context, root *os.Root, name string, line, limit *int, maxBytes int) (string, error) {
	rel, err := t.resolve(name)
	if err != nil {
		return "", err
	}
	f, info, err := openRegular(root, rel)
	if errors.Is(err, errIsDir) {
		return "", fmt.Errorf("%s is a directory; use list_directory", name)
	}
	if err != nil {
		return "", fsError(name, err)
	}
	defer f.Close()

	reader := bufio.NewReaderSize(f, 64<<10)
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

	var out bytes.Buffer
	if start == 1 && maxLines < 0 {
		out.Grow(int(min(info.Size(), int64(maxBytes))) + 256)
	}
	truncated, midLine := false, false
	current := 1
	for i := 0; ; i++ {
		if maxLines >= 0 && current-start >= maxLines {
			break
		}
		if i%fsCtxCheckIterations == 0 {
			if err := ctx.Err(); err != nil {
				return "", err
			}
		}
		chunk, readErr := reader.ReadSlice('\n')
		if len(chunk) > 0 && current >= start {
			if room := maxBytes - out.Len(); len(chunk) > room {
				for room > 0 && !utf8.RuneStart(chunk[room]) {
					room--
				}
				out.Write(chunk[:room])
				truncated = true
				midLine = room > 0 || out.Len() > 0 && out.Bytes()[out.Len()-1] != '\n'
				break
			}
			out.Write(chunk)
		}
		if len(chunk) > 0 && chunk[len(chunk)-1] == '\n' {
			current++
		}
		if errors.Is(readErr, bufio.ErrBufferFull) {
			continue
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return "", fsError(name, readErr)
		}
	}

	switch {
	case truncated && midLine:
		fmt.Fprintf(&out, "\n[Output truncated at %d bytes, within line %d. The rest of that line is not shown; use line and limit to read later lines.]", maxBytes, current)
	case truncated:
		fmt.Fprintf(&out, "\n[Output truncated at %d bytes, before line %d. Use line and limit to read the rest.]", maxBytes, current)
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
	if len(in.Paths) == 0 {
		return "", errors.New("paths must not be empty")
	}
	if len(in.Paths) > fsMaxMultiReadFiles {
		return "", fmt.Errorf("at most %d paths can be read at once, got %d; split them into several calls", fsMaxMultiReadFiles, len(in.Paths))
	}
	root, err := t.open()
	if err != nil {
		return "", err
	}
	defer root.Close()

	var out strings.Builder
	var skipped []string
	seen := make(map[string]bool, len(in.Paths))
	for _, name := range in.Paths {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		if rel, err := t.resolve(name); err == nil {
			if seen[rel] {
				continue
			}
			seen[rel] = true
		}
		remaining := fsMaxMultiReadBytes - out.Len()
		if remaining < 1024 {
			skipped = append(skipped, name)
			continue
		}
		content, err := t.readText(ctx, root, name, nil, nil, min(fsMaxReadBytes, remaining))
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return "", ctxErr
			}
			content = "Error: " + err.Error()
		}
		fmt.Fprintf(&out, "=== %s ===\n%s\n\n", name, content)
	}
	if len(skipped) > 0 {
		fmt.Fprintf(&out, "[Output limit of %d bytes reached. Not read: %s. Read them in another call.]", fsMaxMultiReadBytes, strings.Join(skipped, ", "))
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

	entries, more, err := readDirLimited(root, rel, fsMaxWalkEntries)
	if err != nil {
		return "", fsError(in.Path, err)
	}
	if len(entries) == 0 {
		return fmt.Sprintf("Directory is empty: %s", display(rel)), nil
	}

	var out strings.Builder
	for i, entry := range entries {
		if i == fsMaxEntries {
			break
		}
		if entry.IsDir() {
			fmt.Fprintf(&out, "DIR  %s\n", entry.Name())
		} else {
			fmt.Fprintf(&out, "FILE %s\n", entry.Name())
		}
	}
	switch {
	case more:
		fmt.Fprintf(&out, "[Output truncated: showing %d of more than %d entries. Use glob to find specific files.]\n", min(len(entries), fsMaxEntries), fsMaxWalkEntries)
	case len(entries) > fsMaxEntries:
		fmt.Fprintf(&out, "[Output truncated: showing %d of %d entries.]\n", fsMaxEntries, len(entries))
	}
	return out.String(), nil
}

func checkDir(root *os.Root, rel, name string) error {
	info, err := root.Stat(rel)
	if err != nil {
		return fsError(name, err)
	}
	if !info.IsDir() {
		return fmt.Errorf("%s is not a directory", name)
	}
	return nil
}

// readDirLimited reads at most limit entries of a directory, sorted by name,
// and reports whether it has more.
func readDirLimited(root *os.Root, rel string, limit int) ([]fs.DirEntry, bool, error) {
	f, err := root.Open(rel)
	if err != nil {
		return nil, false, err
	}
	defer f.Close()
	var entries []fs.DirEntry
	if limit > 0 {
		entries, err = f.ReadDir(limit + 1)
	} else {
		entries, err = f.ReadDir(1)
	}
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, false, err
	}
	more := len(entries) > limit
	entries = entries[:min(len(entries), limit)]
	slices.SortFunc(entries, func(a, b fs.DirEntry) int { return strings.Compare(a.Name(), b.Name()) })
	return entries, more, nil
}

// walkRoot walks the tree under start (a slash-separated root-relative path)
// in lexical order without following symlinks, calling fn for every entry but
// start itself. fn may return fs.SkipDir for a directory or fs.SkipAll. At
// most fsMaxWalkEntries entries are visited; walkRoot reports whether it
// stopped because of that budget.
func walkRoot(ctx context.Context, root *os.Root, start string, fn func(p string, d fs.DirEntry) error) (bool, error) {
	visited := 0
	exhausted := false
	var walk func(dir string) error
	walk = func(dir string) error {
		entries, more, err := readDirLimited(root, filepath.FromSlash(dir), fsMaxWalkEntries-visited)
		if err != nil {
			if dir == start {
				return err
			}
			return nil
		}
		if more {
			exhausted = true
		}
		for _, entry := range entries {
			if err := ctx.Err(); err != nil {
				return err
			}
			visited++
			p := path.Join(dir, entry.Name())
			err := fn(p, entry)
			if errors.Is(err, fs.SkipDir) {
				continue
			}
			if err != nil {
				return err
			}
			if entry.IsDir() {
				if err := walk(p); err != nil {
					return err
				}
			}
			if exhausted {
				return fs.SkipAll
			}
		}
		if exhausted {
			return fs.SkipAll
		}
		return nil
	}
	err := walk(start)
	if errors.Is(err, fs.SkipAll) {
		err = nil
	}
	return exhausted, err
}

func walkBudgetNote() string {
	return fmt.Sprintf("[Stopped after visiting %d entries; results may be incomplete. Use a narrower path.]", fsMaxWalkEntries)
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

	if err := checkDir(root, rel, in.Path); err != nil {
		return "", err
	}

	start := display(rel)
	var out strings.Builder
	fmt.Fprintf(&out, "%s/\n", start)
	count := 0
	exhausted, err := walkRoot(ctx, root, start, func(p string, d fs.DirEntry) error {
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
	if exhausted {
		out.WriteString(walkBudgetNote() + "\n")
	}
	return out.String(), nil
}

type globInput struct {
	Pattern string `json:"pattern" description:"Glob pattern relative to path, for example **/*.go"`
	Path    string `json:"path,omitempty" description:"Directory to search from; defaults to the root"`
}

func (t *fsTools) glob(ctx context.Context, in globInput) (string, error) {
	pattern, err := cleanGlob(strings.TrimPrefix(filepath.ToSlash(in.Pattern), "./"))
	if err != nil {
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

	if err := checkDir(root, rel, in.Path); err != nil {
		return "", err
	}

	start := display(rel)
	var matches []string
	truncated := false
	exhausted, err := walkRoot(ctx, root, start, func(p string, d fs.DirEntry) error {
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
	out := strings.Join(matches, "\n")
	if len(matches) == 0 {
		out = "No files found"
	}
	switch {
	case truncated:
		out += fmt.Sprintf("\n[Output truncated at %d files. Use a narrower pattern.]", fsMaxEntries)
	case exhausted:
		out += "\n" + walkBudgetNote()
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
		var err error
		if include, err = cleanGlob(include); err != nil {
			return "", err
		}
	}
	excludes := make([]string, 0, len(in.ExcludePatterns))
	for _, pattern := range in.ExcludePatterns {
		pattern, err := cleanGlob(strings.TrimSuffix(strings.TrimPrefix(filepath.ToSlash(pattern), "./"), "/"))
		if err != nil {
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

	if err := checkDir(root, rel, in.Path); err != nil {
		return "", err
	}

	start := display(rel)
	var out strings.Builder
	matches := 0
	scanned := int64(0)
	var stopped string
	exhausted, err := walkRoot(ctx, root, start, func(p string, d fs.DirEntry) error {
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

		f, info, err := openRegular(root, filepath.FromSlash(p))
		if err != nil {
			return nil
		}
		defer f.Close()
		if info.Size() > fsMaxSearchFile {
			return nil
		}
		if scanned+info.Size() > fsMaxSearchBytes {
			stopped = fmt.Sprintf("[Stopped after reading %d MB of files; results may be incomplete. Narrow the search with path, include or exclude_patterns.]", fsMaxSearchBytes>>20)
			return fs.SkipAll
		}
		content, err := io.ReadAll(io.LimitReader(f, fsMaxSearchFile+1))
		scanned += int64(len(content))
		if err != nil || len(content) > fsMaxSearchFile || isBinary(content[:min(len(content), fsBinarySniffBytes)]) {
			return nil
		}

		lineNum := 0
		for line := range strings.SplitSeq(string(content), "\n") {
			lineNum++
			if lineNum%fsCtxCheckIterations == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			line = strings.TrimSuffix(line, "\r")
			from, to, ok := match(line)
			if !ok {
				continue
			}
			entry := fmt.Sprintf("%s:%d:%d: %s", p, lineNum, from+1, preview(line, from, to))
			if out.Len()+len(entry)+1 > fsMaxSearchOutput {
				stopped = "[Output truncated. Narrow the search with a more specific query, path, include or exclude_patterns.]"
				return fs.SkipAll
			}
			if matches > 0 {
				out.WriteByte('\n')
			}
			out.WriteString(entry)
			matches++
		}
		return nil
	})
	if err != nil {
		return "", fsError(in.Path, err)
	}
	if stopped == "" && exhausted {
		stopped = walkBudgetNote()
	}
	if matches == 0 {
		if stopped != "" {
			return "No results found\n" + stopped, nil
		}
		return "No results found", nil
	}
	if stopped != "" {
		out.WriteString("\n" + stopped)
	}
	return out.String(), nil
}

type writeFileInput struct {
	Path    string `json:"path" description:"File to write"`
	Content string `json:"content" description:"Full file content"`
}

func (t *fsTools) writeFile(ctx context.Context, in writeFileInput) (string, error) {
	t.writeMu.Lock()
	defer t.writeMu.Unlock()
	rel, err := t.resolve(in.Path)
	if err != nil {
		return "", err
	}
	root, err := t.open()
	if err != nil {
		return "", err
	}
	defer root.Close()

	if err := refuseSymlinkParents(root, rel); err != nil {
		return "", err
	}
	if dir := filepath.Dir(rel); dir != "." {
		if err := root.MkdirAll(dir, 0o755); err != nil {
			return "", fsError(in.Path, err)
		}
	}
	if err := replaceFile(root, rel, []byte(in.Content)); err != nil {
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
	t.writeMu.Lock()
	defer t.writeMu.Unlock()
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

	if err := refuseSymlinkParents(root, rel); err != nil {
		return "", err
	}
	data, err := readForEdit(root, rel)
	if err != nil {
		return "", fsError(in.Path, err)
	}
	content := string(data)
	crlf := strings.Contains(content, "\r\n")

	for i, edit := range in.Edits {
		oldText, newText := edit.OldText, edit.NewText
		if oldText == "" {
			return "", fmt.Errorf("edit %d: old_text must not be empty", i+1)
		}
		n := countOverlapping(content, oldText)
		if n == 0 && crlf && toCRLF(oldText) != oldText {
			// The model usually writes \n; match the file's CRLF lines.
			oldText, newText = toCRLF(oldText), toCRLF(newText)
			n = countOverlapping(content, oldText)
		}
		switch n {
		case 0:
			return "", fmt.Errorf("edit %d: old_text not found in %s", i+1, in.Path)
		case 1:
			content = strings.Replace(content, oldText, newText, 1)
		default:
			return "", fmt.Errorf("edit %d: old_text appears %d times in %s; include more surrounding text to make it unique", i+1, n, in.Path)
		}
	}

	if err := replaceFile(root, rel, []byte(content)); err != nil {
		return "", fsError(in.Path, err)
	}
	return fmt.Sprintf("Applied %d edit(s) to %s", len(in.Edits), display(rel)), nil
}

// countOverlapping counts the places sub starts in s, including overlapping
// ones, which strings.Count skips. Two overlapping matches make an edit
// ambiguous just as two separate ones do.
func countOverlapping(s, sub string) int {
	n := 0
	for {
		i := strings.Index(s, sub)
		if i < 0 {
			return n
		}
		n++
		s = s[i+1:]
	}
}

func readForEdit(root *os.Root, rel string) ([]byte, error) {
	f, info, err := openRegular(root, rel)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	tooLarge := fmt.Errorf("file is larger than %d MB; edit_file cannot change it", fsMaxEditBytes>>20)
	if info.Size() > fsMaxEditBytes {
		return nil, tooLarge
	}
	data, err := io.ReadAll(io.LimitReader(f, fsMaxEditBytes+1))
	if err != nil {
		return nil, err
	}
	if len(data) > fsMaxEditBytes {
		return nil, tooLarge
	}
	return data, nil
}

// replaceFile writes data to a temporary file next to rel and renames it over
// rel, so rel never holds partly written content. An existing file keeps its
// permissions. Symlinks are not written through, because the change would
// land on a file other than the one the approval named.
func replaceFile(root *os.Root, rel string, data []byte) error {
	perm, keepPerm := os.FileMode(0o644), false
	if info, err := root.Lstat(rel); err == nil {
		switch {
		case info.Mode()&fs.ModeSymlink != 0:
			return errSymlinkWrite(root, rel)
		case info.IsDir():
			return errIsDir
		case !info.Mode().IsRegular():
			return errNotRegular
		case info.Mode().Perm()&0o200 == 0:
			// Renaming over the file needs only directory permission, so
			// check the file's own before replacing it.
			return errReadOnly
		}
		perm, keepPerm = info.Mode().Perm(), true
	} else if !errors.Is(err, fs.ErrNotExist) {
		return err
	}

	tmp := filepath.Join(filepath.Dir(rel), "."+filepath.Base(rel)+"."+rand.Text()[:10]+".tmp")
	f, err := root.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_EXCL, perm)
	if err != nil {
		return err
	}
	_, err = f.Write(data)
	if err == nil && keepPerm {
		err = f.Chmod(perm)
	}
	if err == nil {
		err = f.Sync()
	}
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	if err == nil {
		err = root.Rename(tmp, rel)
	}
	if err != nil {
		root.Remove(tmp)
		return err
	}
	return nil
}

// refuseSymlinkParents fails when a directory on the way to rel is a symbolic
// link, since a change there would land somewhere the approval did not name.
// Missing directories are fine; they are created as real directories.
func refuseSymlinkParents(root *os.Root, rel string) error {
	dir := filepath.Dir(rel)
	if dir == "." {
		return nil
	}
	var prefix string
	for _, part := range strings.Split(filepath.ToSlash(dir), "/") {
		prefix = filepath.Join(prefix, part)
		info, err := root.Lstat(prefix)
		if errors.Is(err, fs.ErrNotExist) {
			return nil
		}
		if err != nil {
			return err
		}
		if info.Mode()&fs.ModeSymlink != 0 {
			return fmt.Errorf("%s is a symbolic link; use the path it points to instead", filepath.ToSlash(prefix))
		}
	}
	return nil
}

// errSymlinkWrite names the link's target when it is a relative path, so the
// model can write the target itself. An absolute target is not shown, because
// it would reveal host paths.
func errSymlinkWrite(root *os.Root, rel string) error {
	target, err := root.Readlink(rel)
	if err != nil || filepath.IsAbs(target) || filepath.VolumeName(target) != "" {
		return errors.New("is a symbolic link; write to the file it points to instead")
	}
	target = filepath.ToSlash(filepath.Join(filepath.Dir(rel), target))
	return fmt.Errorf("is a symbolic link to %s; write to that file instead", target)
}

type directoriesInput struct {
	Paths []string `json:"paths" description:"Directories"`
}

func (t *fsTools) createDirectory(ctx context.Context, in directoriesInput) (string, error) {
	t.writeMu.Lock()
	defer t.writeMu.Unlock()
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
		if err := refuseSymlinkParents(root, filepath.Join(rel, "x")); err != nil {
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
	t.writeMu.Lock()
	defer t.writeMu.Unlock()
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
		if err := refuseSymlinkParents(root, rel); err != nil {
			return "", err
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

// cleanGlob validates a slash-separated glob pattern, bounds its size and
// collapses repeated "**" segments, which match the same paths as one.
func cleanGlob(pattern string) (string, error) {
	if pattern == "" {
		return "", errors.New("pattern must not be empty")
	}
	if len(pattern) > fsMaxGlobLength {
		return "", fmt.Errorf("glob pattern is longer than %d characters", fsMaxGlobLength)
	}
	var segments []string
	for _, segment := range strings.Split(pattern, "/") {
		if segment == "**" && len(segments) > 0 && segments[len(segments)-1] == "**" {
			continue
		}
		if _, err := path.Match(segment, ""); err != nil {
			return "", fmt.Errorf("invalid glob pattern %q: %w", pattern, err)
		}
		segments = append(segments, segment)
	}
	if len(segments) > fsMaxGlobSegments {
		return "", fmt.Errorf("glob pattern %q has more than %d segments", pattern, fsMaxGlobSegments)
	}
	return strings.Join(segments, "/"), nil
}

// matchGlob matches a slash-separated path against a glob pattern in which
// "**" matches any number of path segments.
func matchGlob(pattern, name string) bool {
	return matchSegments(strings.Split(pattern, "/"), strings.Split(name, "/"))
}

// matchSegments runs in O(len(pattern) * len(name)) time. next[j] reports
// whether the pattern suffix after the current segment matches name[j:].
func matchSegments(pattern, name []string) bool {
	next := make([]bool, len(name)+1)
	cur := make([]bool, len(name)+1)
	next[len(name)] = true
	for i := len(pattern) - 1; i >= 0; i-- {
		if pattern[i] == "**" {
			found := false
			for j := len(name); j >= 0; j-- {
				found = found || next[j]
				cur[j] = found
			}
		} else {
			cur[len(name)] = false
			for j := len(name) - 1; j >= 0; j-- {
				ok := false
				if next[j+1] {
					ok, _ = path.Match(pattern[i], name[j])
				}
				cur[j] = ok
			}
		}
		next, cur = cur, next
	}
	return next[0]
}

// isBinary reports whether a file prefix looks binary: it contains a NUL byte
// or is not valid UTF-8. A rune cut off at the end of the prefix is ignored.
func isBinary(head []byte) bool {
	if bytes.IndexByte(head, 0) >= 0 {
		return true
	}
	for i := len(head) - 1; i >= 0 && i >= len(head)-utf8.UTFMax; i-- {
		if utf8.RuneStart(head[i]) {
			if !utf8.FullRune(head[i:]) {
				head = head[:i]
			}
			break
		}
	}
	return !utf8.Valid(head)
}

// preview returns at most fsMaxPreview bytes of line around a match.
func preview(line string, from, to int) string {
	if len(line) <= fsMaxPreview {
		return line
	}
	start := max(from-40, 0)
	end := min(start+fsMaxPreview, len(line))
	for start < end && !utf8.RuneStart(line[start]) {
		start++
	}
	for end > start && end < len(line) && !utf8.RuneStart(line[end]) {
		end--
	}
	return line[start:end]
}

func toCRLF(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "\r\n", "\n"), "\n", "\r\n")
}
