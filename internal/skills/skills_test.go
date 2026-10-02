package skills

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func writeSkill(t *testing.T, root, folder, content string) {
	t.Helper()
	require.NoError(t, os.MkdirAll(filepath.Join(root, folder), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(root, folder, SkillFile), []byte(content), 0o644))
}

func TestParse(t *testing.T) {
	fm, body, err := Parse([]byte("\ufeff---\r\nname: pdf\r\ndescription: Fill forms.\r\nlicense: MIT\r\n---\r\n# Steps\r\n"))
	require.NoError(t, err)
	require.Equal(t, Frontmatter{Name: "pdf", Description: "Fill forms."}, fm)
	require.Equal(t, "# Steps\n", body)

	for content, want := range map[string]string{
		"name: pdf\n":                                                           "must start with a --- line",
		"---\nname: pdf\ndescription: x\n":                                      "no closing ---",
		"---\n- a\n---\n":                                                       "must be a YAML mapping",
		"---\ndescription: x\n---\n":                                            "name is required",
		"---\nname: pdf\n---\n":                                                 "description is required",
		"---\nname: PDF\ndescription: x\n---\n":                                 "lowercase",
		"---\nname: -pdf\ndescription: x\n---\n":                                "lowercase",
		"---\nname: a--b\ndescription: x\n---\n":                                "lowercase",
		"---\nname: [a]\ndescription: x\n---\n":                                 "frontmatter",
		"---\nname: " + strings.Repeat("a", 65) + "\ndescription: x\n---\n":     "longer than 64",
		"---\nname: pdf\ndescription: " + strings.Repeat("d", 1025) + "\n---\n": "longer than 1024",
	} {
		_, _, err := Parse([]byte(content))
		require.ErrorContains(t, err, want, content)
	}
}

func TestListAndCheck(t *testing.T) {
	root := t.TempDir()
	writeSkill(t, root, "b-skill", "---\nname: b-skill\ndescription: B.\n---\nb")
	writeSkill(t, root, "a-skill", "---\nname: a-skill\ndescription: A <&> B.\n---\na")
	writeSkill(t, root, "broken", "---\nname: other\ndescription: x\n---\n")
	require.NoError(t, os.MkdirAll(filepath.Join(root, "not-a-skill"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(root, "README.md"), []byte("hi"), 0o644))

	s := New(root)
	list, err := s.List()
	require.NoError(t, err)
	require.Equal(t, []Frontmatter{{"a-skill", "A <&> B."}, {"b-skill", "B."}}, list)
	require.ErrorContains(t, s.Check(), `skill "broken": name "other" does not match its folder "broken"`)

	text, err := s.Instructions()
	require.NoError(t, err)
	require.Contains(t, text, "<skill>\n<name>a-skill</name>\n<description>A &lt;&amp;&gt; B.</description>\n</skill>")
	require.NotContains(t, text, "broken")

	empty, err := New(t.TempDir()).Instructions()
	require.NoError(t, err)
	require.Contains(t, empty, "There are no skills yet.")
}

func TestLoadAndResource(t *testing.T) {
	root := t.TempDir()
	writeSkill(t, root, "pdf", "---\nname: pdf\ndescription: PDFs.\n---\n\nUse the template.\n")
	require.NoError(t, os.MkdirAll(filepath.Join(root, "pdf", "assets"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(root, "pdf", "assets", "t.txt"), []byte("template"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(root, "pdf", "assets", "logo.png"), []byte{0x89, 'P', 0, 0xff}, 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(root, "secret.txt"), []byte("secret"), 0o644))
	s := New(root)
	ctx := t.Context()

	out, err := s.Load(ctx, LoadInput{Name: "pdf"})
	require.NoError(t, err)
	require.Equal(t, "# Skill: pdf\n\nUse the template.\n\n## Files in this skill (read them with load_skill_resource)\n- assets/logo.png\n- assets/t.txt\n", out)

	_, err = s.Load(ctx, LoadInput{Name: "../pdf"})
	require.ErrorContains(t, err, "no skill named")
	_, err = s.Load(ctx, LoadInput{Name: "missing"})
	require.ErrorContains(t, err, "no skill named")

	out, err = s.Resource(ctx, ResourceInput{SkillName: "pdf", ResourcePath: "assets/t.txt"})
	require.NoError(t, err)
	require.Equal(t, "template", out)

	for _, p := range []string{"", "../secret.txt", "/etc/passwd", `assets\t.txt`, "assets/../../secret.txt"} {
		_, err := s.Resource(ctx, ResourceInput{SkillName: "pdf", ResourcePath: p})
		require.Error(t, err, p)
	}
	_, err = s.Resource(ctx, ResourceInput{SkillName: "pdf", ResourcePath: "assets/logo.png"})
	require.ErrorContains(t, err, "binary")
	_, err = s.Resource(ctx, ResourceInput{SkillName: "pdf", ResourcePath: "assets/none.txt"})
	require.ErrorContains(t, err, "has no file")

	outside := filepath.Join(t.TempDir(), "outside.txt")
	require.NoError(t, os.WriteFile(outside, []byte("outside"), 0o644))
	if err := os.Symlink(outside, filepath.Join(root, "pdf", "link.txt")); err == nil {
		_, err = s.Resource(ctx, ResourceInput{SkillName: "pdf", ResourcePath: "link.txt"})
		require.Error(t, err, "symlinks out of the root are refused by os.Root")
	}

	big := strings.Repeat("x", maxFileBytes+1)
	require.NoError(t, os.WriteFile(filepath.Join(root, "pdf", "big.txt"), []byte(big), 0o644))
	_, err = s.Resource(ctx, ResourceInput{SkillName: "pdf", ResourcePath: "big.txt"})
	require.ErrorContains(t, err, "larger than")
}

func TestSave(t *testing.T) {
	root := t.TempDir()
	s := New(root)
	ctx := t.Context()

	out, err := s.Save(ctx, SaveInput{Name: "notes", Description: "Take notes.", Instructions: "Be short.",
		Files: []File{{Path: "references/style.md", Content: "plain"}}})
	require.NoError(t, err)
	require.Equal(t, `Created skill "notes" and wrote 1 other file(s).`, out)

	loaded, err := s.Load(ctx, LoadInput{Name: "notes"})
	require.NoError(t, err)
	require.Contains(t, loaded, "Be short.")
	require.Contains(t, loaded, "- references/style.md")

	// Updating keeps other frontmatter fields and files that are not listed.
	writeSkill(t, root, "notes", "---\nlicense: MIT\nname: notes\ndescription: Old.\nmetadata:\n  version: \"1\"\n---\nold")
	out, err = s.Save(ctx, SaveInput{Name: "notes", Description: "Take notes: \"quoted\"", Instructions: "New."})
	require.NoError(t, err)
	require.Equal(t, `Updated skill "notes".`, out)
	data, err := os.ReadFile(filepath.Join(root, "notes", SkillFile))
	require.NoError(t, err)
	require.Equal(t, "---\nlicense: MIT\nname: notes\ndescription: 'Take notes: \"quoted\"'\nmetadata:\n    version: \"1\"\n---\n\nNew.\n", string(data))
	fm, _, err := Parse(data)
	require.NoError(t, err)
	require.Equal(t, `Take notes: "quoted"`, fm.Description)
	require.FileExists(t, filepath.Join(root, "notes", "references", "style.md"))

	for in, want := range map[*SaveInput]string{
		{Name: "Bad", Description: "x", Instructions: "x"}:                                                 "lowercase",
		{Name: "ok", Description: "", Instructions: "x"}:                                                   "description is required",
		{Name: "ok", Description: "x", Instructions: " "}:                                                  "instructions are required",
		{Name: "ok", Description: "x", Instructions: "x", Files: []File{{Path: "../escape.md"}}}:           "inside the skill folder",
		{Name: "ok", Description: "x", Instructions: "x", Files: []File{{Path: "skill.md"}}}:               "leave it out of files",
		{Name: "ok", Description: "x", Instructions: "x", Files: []File{{Path: "a.md"}, {Path: "./a.md"}}}: "listed twice",
	} {
		_, err := s.Save(ctx, *in)
		require.ErrorContains(t, err, want)
	}
	require.NoDirExists(t, filepath.Join(root, "ok"), "an invalid save writes nothing")
	require.NoFileExists(t, filepath.Join(root, "escape.md"))
}
