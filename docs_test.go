package crux

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

var markdownLink = regexp.MustCompile(`\]\(([^)\s]+)\)`)

// TestDocsLinks checks that the relative links in AGENTS.md and docs/ point
// to files that exist.
func TestDocsLinks(t *testing.T) {
	docs, err := filepath.Glob("docs/*.md")
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) == 0 {
		t.Fatal("no docs found")
	}
	for _, file := range append([]string{"AGENTS.md"}, docs...) {
		data, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range markdownLink.FindAllStringSubmatch(string(data), -1) {
			target, _, _ := strings.Cut(m[1], "#")
			if target == "" || strings.Contains(target, "://") || strings.HasPrefix(target, "mailto:") {
				continue
			}
			if _, err := os.Stat(filepath.Join(filepath.Dir(file), target)); err != nil {
				t.Errorf("%s links to %s, which does not exist", file, m[1])
			}
		}
	}
}
