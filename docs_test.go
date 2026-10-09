package crux

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"crux.foo/internal/skills"
)

// TestUserDocsSkill checks that skills/crux, the guide for users' coding
// agents, is a valid skill named crux.
func TestUserDocsSkill(t *testing.T) {
	s := skills.New("skills")
	if err := s.Check(); err != nil {
		t.Fatal(err)
	}
	list, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 1 || list[0].Name != "crux" {
		t.Fatalf("skills = %+v, want one named crux", list)
	}
}

var markdownLink = regexp.MustCompile(`\]\(([^)\s]+)\)`)

// TestUserDocsLinks checks that the relative links in AGENTS.md and the skill
// point to files that exist.
func TestUserDocsLinks(t *testing.T) {
	files := []string{"AGENTS.md"}
	pages, err := filepath.Glob("skills/crux/*.md")
	if err != nil {
		t.Fatal(err)
	}
	refs, err := filepath.Glob("skills/crux/references/*.md")
	if err != nil {
		t.Fatal(err)
	}
	files = append(append(files, pages...), refs...)
	for _, file := range files {
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
