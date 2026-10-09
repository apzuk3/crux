package main

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path"
	"slices"
	"strings"
	"time"

	"crux.foo"
)

// An agent that finds what is taking up space in a directory and suggests
// what could go. It has two read-only tools, both confined to the directory
// with os.Root, and prints the session log after the answer.
//
//	go run ./examples/loop-tools-log ~/Downloads
//	go run ./examples/loop-tools-log ~/src "Which build caches can I delete?"

// maxEntries bounds each walk, so a huge tree can't stall a tool call.
const maxEntries = 200_000

var root *os.Root

type dirArgs struct {
	Path string `json:"path" description:"Directory to inspect, relative to the root; use . for the root itself"`
}

type DirSize struct {
	Name  string `json:"name"`
	Dir   bool   `json:"dir"`
	Bytes int64  `json:"bytes"`
	Files int    `json:"files"`
}

// dirSizes returns the total size of each entry directly inside a directory,
// largest first.
func dirSizes(ctx context.Context, args dirArgs) ([]DirSize, error) {
	entries, err := fs.ReadDir(root.FS(), clean(args.Path))
	if err != nil {
		return nil, err
	}
	sizes := make([]DirSize, 0, len(entries))
	for _, entry := range entries {
		size := DirSize{Name: entry.Name(), Dir: entry.IsDir()}
		err := walk(ctx, path.Join(clean(args.Path), entry.Name()), func(_ string, info fs.FileInfo) {
			size.Bytes += info.Size()
			size.Files++
		})
		if err != nil {
			return nil, err
		}
		sizes = append(sizes, size)
	}
	slices.SortFunc(sizes, func(a, b DirSize) int { return cmp.Compare(b.Bytes, a.Bytes) })
	return sizes, nil
}

type largestArgs struct {
	Path       string `json:"path" description:"Directory to search recursively, relative to the root; use . for the root itself"`
	Limit      int    `json:"limit,omitempty" description:"How many files to return, 1 to 100; defaults to 20"`
	MinAgeDays int    `json:"min_age_days,omitempty" description:"Only files not modified for at least this many days"`
}

type File struct {
	Path     string `json:"path"`
	Bytes    int64  `json:"bytes"`
	Modified string `json:"modified"` // YYYY-MM-DD
}

// largestFiles returns the biggest files under a directory, optionally only
// the ones that haven't changed in a while.
func largestFiles(ctx context.Context, args largestArgs) ([]File, error) {
	limit := args.Limit
	if limit == 0 {
		limit = 20
	}
	if limit < 1 || limit > 100 {
		return nil, fmt.Errorf("limit must be between 1 and 100, got %d", limit)
	}
	cutoff := time.Now().AddDate(0, 0, -args.MinAgeDays)
	var files []File
	err := walk(ctx, clean(args.Path), func(name string, info fs.FileInfo) {
		if info.ModTime().After(cutoff) {
			return
		}
		files = append(files, File{name, info.Size(), info.ModTime().Format(time.DateOnly)})
	})
	if err != nil {
		return nil, err
	}
	slices.SortFunc(files, func(a, b File) int { return cmp.Compare(b.Bytes, a.Bytes) })
	return files[:min(limit, len(files))], nil
}

// walk calls fn for every regular file under name (or name itself, if it is a
// file). Unreadable entries are skipped. Symlinks are not followed.
func walk(ctx context.Context, name string, fn func(name string, info fs.FileInfo)) error {
	seen := 0
	return fs.WalkDir(root.FS(), name, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			if p == name {
				return err // the model asked for a path that doesn't exist
			}
			return nil
		}
		if seen++; seen > maxEntries {
			return fmt.Errorf("more than %d entries under %q; inspect a subdirectory instead", maxEntries, name)
		}
		if seen%1000 == 0 && ctx.Err() != nil {
			return ctx.Err()
		}
		if !d.Type().IsRegular() {
			return nil
		}
		if info, err := d.Info(); err == nil {
			fn(p, info)
		}
		return nil
	})
}

// clean turns the model's path into one fs.FS accepts.
func clean(p string) string {
	p = path.Clean("/" + strings.ReplaceAll(p, `\`, "/"))
	if p == "/" {
		return "."
	}
	return p[1:]
}

func init() {
	crux.RegisterTool("dir_sizes", "Total size and file count of each entry directly inside a directory, largest first", dirSizes)
	crux.RegisterTool("largest_files", "The largest files anywhere under a directory, with their last modified date", largestFiles)
}

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: loop-tools-log <directory> [question]")
	}
	question := "What is taking up the most space here, and what looks safe to delete?"
	if len(os.Args) > 2 {
		question = strings.Join(os.Args[2:], " ")
	}
	var err error
	if root, err = os.OpenRoot(os.Args[1]); err != nil {
		log.Fatal(err)
	}
	defer root.Close()

	agent := crux.Must(crux.New("disk-usage", crux.ClaudeHaiku4_5,
		crux.WithInstructions(`You help people free up disk space in one directory.
Paths are relative to that directory. Start with dir_sizes on ".", then drill into
the biggest entries. Use largest_files with min_age_days to find files nobody has
touched in a while. Recommend what to delete and why, with sizes in MB or GB.
Flag anything that looks like source code, documents or photos as "check first".
You can't delete anything yourself; say so if asked.`),
		crux.WithTools([]string{"dir_sizes", "largest_files"}),
		crux.WithMaxTurns(12),
	))

	ctx := context.Background()
	session := crux.MustSession(crux.NewSession(ctx, agent))
	answer, err := session.Run(ctx, question)
	if errors.Is(err, crux.ErrMaxTurns) {
		log.Print("the agent ran out of turns before it finished")
	} else if err != nil {
		log.Fatal(err)
	}
	fmt.Println(answer)

	// Everything that happened is in the session log.
	fmt.Println("\n--- session log ---")
	for _, e := range session.Logs() {
		fmt.Printf("%3d  %s\n", e.Seq, describe(e))
	}
}

func describe(e crux.Entry) string {
	switch e.Kind {
	case crux.KindUser:
		return "user: " + e.Text()
	case crux.KindRunStarted:
		return "run started"
	case crux.KindRunFinished:
		return "run finished: " + string(e.Run.Outcome)
	case crux.KindTurnStarted:
		return "request to " + e.Turn.Model
	case crux.KindToolCall:
		return fmt.Sprintf("model asks for %s %s", e.ToolCall.Name, e.ToolCall.Args)
	case crux.KindToolStarted:
		return "running " + e.ToolCall.Name
	case crux.KindToolResult:
		if e.ToolResult.Error != "" {
			return "tool error, sent to the model: " + e.ToolResult.Error
		}
		return fmt.Sprintf("tool result, %d bytes, in %s", len(e.ToolResult.Output), e.Duration.Round(time.Millisecond))
	case crux.KindAssistant:
		return fmt.Sprintf("answer, %d characters", len(e.Text()))
	}
	return fmt.Sprintf("entry of kind %d", e.Kind)
}
