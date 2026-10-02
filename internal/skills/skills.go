// Package skills implements agent skills: folders with a SKILL.md (YAML
// frontmatter with a name and a description, then markdown instructions) and
// any other files the instructions refer to. Every skill is a direct
// subdirectory of one root directory, named like the skill. The crux package
// registers the tools as the "skills" toolset.
package skills

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"html"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/apzuk3/crux/internal/filesystem"
	"gopkg.in/yaml.v3"
)

const (
	// SkillFile is the file that holds a skill's frontmatter and instructions.
	SkillFile = "SKILL.md"

	maxNameLength        = 64
	maxDescriptionLength = 1024
	maxFileBytes         = 1 << 20 // SKILL.md and resource cap, read and write
	maxSaveFiles         = 50      // files one save_skill call may write
	maxListedFiles       = 200     // files listed by load_skill
)

var nameRE = regexp.MustCompile(`^[a-z0-9]+(-[a-z0-9]+)*$`)

// Frontmatter is the metadata of a skill that the model sees before loading it.
type Frontmatter struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
}

// Skills reads and writes the skills under one root directory.
type Skills struct {
	root string
	fs   *filesystem.Tools
	mu   sync.Mutex // serialises saves
}

// New returns the skills under root, which must be an absolute path to an
// existing directory.
func New(root string) *Skills {
	return &Skills{root: root, fs: filesystem.New(root)}
}

func (s *Skills) open() (*os.Root, error) {
	root, err := os.OpenRoot(s.root)
	if err != nil {
		var pathErr *fs.PathError
		if errors.As(err, &pathErr) {
			err = pathErr.Err
		}
		return nil, fmt.Errorf("open skills directory: %w", err)
	}
	return root, nil
}

// List returns the frontmatter of every valid skill, sorted by name.
// Subdirectories without a SKILL.md are not skills and are ignored; skills
// with an invalid SKILL.md are left out, so one being edited never stops a run.
func (s *Skills) List() ([]Frontmatter, error) {
	skills, _, err := s.scan()
	return skills, err
}

// Check reports every skill with an invalid or unreadable SKILL.md.
func (s *Skills) Check() error {
	_, invalid, err := s.scan()
	if err != nil {
		return err
	}
	return errors.Join(invalid...)
}

func (s *Skills) scan() (skills []Frontmatter, invalid []error, err error) {
	root, err := s.open()
	if err != nil {
		return nil, nil, err
	}
	defer root.Close()

	entries, err := fs.ReadDir(root.FS(), ".")
	if err != nil {
		return nil, nil, fmt.Errorf("read skills directory: %w", err)
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		data, err := readFile(root, filepath.Join(entry.Name(), SkillFile))
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}
		if err == nil {
			var fm Frontmatter
			if fm, _, err = Parse(data); err == nil {
				err = checkFolder(fm, entry.Name())
			}
			if err == nil {
				skills = append(skills, fm)
				continue
			}
		}
		invalid = append(invalid, fmt.Errorf("skill %q: %w", entry.Name(), err))
	}
	return skills, invalid, nil
}

func checkFolder(fm Frontmatter, folder string) error {
	if fm.Name != folder {
		return fmt.Errorf("name %q does not match its folder %q", fm.Name, folder)
	}
	return nil
}

// Instructions tells the model how to use skills and lists the valid ones.
func (s *Skills) Instructions() (string, error) {
	skills, err := s.List()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(instructions)
	b.WriteString("\n")
	if len(skills) == 0 {
		b.WriteString("There are no skills yet.")
		return b.String(), nil
	}
	b.WriteString("<available_skills>\n")
	for _, fm := range skills {
		fmt.Fprintf(&b, "<skill>\n<name>%s</name>\n<description>%s</description>\n</skill>\n",
			html.EscapeString(fm.Name), html.EscapeString(fm.Description))
	}
	b.WriteString("</available_skills>")
	return b.String(), nil
}

const instructions = `You can use skills: folders of instructions and resources for specialised tasks. Each skill has a SKILL.md with its instructions and may have more files, such as references/, assets/ or scripts/.

1. If a skill is relevant to the request, call load_skill with its name to read its instructions before doing anything else.
2. Then follow those instructions exactly, completing every step in order before you reply.
3. Read a skill's other files with load_skill_resource. Do not use other tools to access them.
4. When you work out a reusable way to do a task, or a skill's instructions turn out wrong or incomplete, create or update a skill with save_skill. Saving replaces the skill's SKILL.md, so load an existing skill first and pass its full instructions.
`

// Parse splits a SKILL.md into its frontmatter and its instructions, and
// validates the frontmatter.
func Parse(data []byte) (Frontmatter, string, error) {
	node, body, err := split(data)
	if err != nil {
		return Frontmatter{}, "", err
	}
	var fm Frontmatter
	if err := node.Decode(&fm); err != nil {
		return Frontmatter{}, "", fmt.Errorf("frontmatter: %w", err)
	}
	if err := Validate(fm); err != nil {
		return Frontmatter{}, "", err
	}
	return fm, body, nil
}

// split separates the YAML frontmatter, between "---" lines at the top of the
// file, from the markdown after it.
func split(data []byte) (*yaml.Node, string, error) {
	text := strings.ReplaceAll(string(data), "\r\n", "\n")
	text = strings.TrimPrefix(text, "\ufeff")
	rest, ok := strings.CutPrefix(text, "---\n")
	if !ok {
		return nil, "", errors.New("SKILL.md must start with a --- line opening its YAML frontmatter")
	}
	var front, body string
	if strings.HasPrefix(rest, "---\n") || rest == "---" {
		front, body = "", strings.TrimPrefix(strings.TrimPrefix(rest, "---"), "\n")
	} else if i := strings.Index(rest, "\n---\n"); i >= 0 {
		front, body = rest[:i], rest[i+len("\n---\n"):]
	} else if strings.HasSuffix(rest, "\n---") {
		front = strings.TrimSuffix(rest, "\n---")
	} else {
		return nil, "", errors.New("SKILL.md frontmatter has no closing --- line")
	}
	var doc yaml.Node
	if err := yaml.Unmarshal([]byte(front), &doc); err != nil {
		return nil, "", fmt.Errorf("frontmatter: %w", err)
	}
	if len(doc.Content) == 0 {
		return &yaml.Node{Kind: yaml.MappingNode}, body, nil
	}
	if doc.Content[0].Kind != yaml.MappingNode {
		return nil, "", errors.New("frontmatter must be a YAML mapping")
	}
	return doc.Content[0], body, nil
}

// Validate checks a skill's name and description.
func Validate(fm Frontmatter) error {
	switch {
	case fm.Name == "":
		return errors.New("name is required")
	case len(fm.Name) > maxNameLength:
		return fmt.Errorf("name %q is longer than %d characters", fm.Name, maxNameLength)
	case !nameRE.MatchString(fm.Name):
		return fmt.Errorf("name %q must be lowercase letters, digits and single hyphens, not starting or ending with a hyphen", fm.Name)
	case strings.TrimSpace(fm.Description) == "":
		return errors.New("description is required")
	case utf8.RuneCountInString(fm.Description) > maxDescriptionLength:
		return fmt.Errorf("description is longer than %d characters", maxDescriptionLength)
	}
	return nil
}

// LoadInput holds the arguments of Load.
type LoadInput struct {
	Name string `json:"name" description:"Name of the skill to load"`
}

// Load returns a skill's instructions and the names of its other files.
func (s *Skills) Load(ctx context.Context, in LoadInput) (string, error) {
	if !nameRE.MatchString(in.Name) {
		return "", fmt.Errorf("no skill named %q", in.Name)
	}
	root, err := s.open()
	if err != nil {
		return "", err
	}
	defer root.Close()

	data, err := readFile(root, filepath.Join(in.Name, SkillFile))
	if errors.Is(err, fs.ErrNotExist) {
		return "", fmt.Errorf("no skill named %q", in.Name)
	}
	if err != nil {
		return "", fmt.Errorf("skill %q: %w", in.Name, err)
	}
	fm, body, err := Parse(data)
	if err == nil {
		err = checkFolder(fm, in.Name)
	}
	if err != nil {
		return "", fmt.Errorf("skill %q is invalid: %w", in.Name, err)
	}

	files, more, err := listFiles(ctx, root, in.Name)
	if err != nil {
		return "", err
	}
	var out strings.Builder
	fmt.Fprintf(&out, "# Skill: %s\n\n%s", fm.Name, strings.TrimSpace(body))
	if len(files) > 0 {
		out.WriteString("\n\n## Files in this skill (read them with load_skill_resource)\n")
		for _, f := range files {
			fmt.Fprintf(&out, "- %s\n", f)
		}
		if more {
			fmt.Fprintf(&out, "[Only the first %d files are listed.]\n", maxListedFiles)
		}
	}
	return out.String(), nil
}

// listFiles returns the skill's files other than SKILL.md, as slash paths
// relative to the skill folder.
func listFiles(ctx context.Context, root *os.Root, name string) ([]string, bool, error) {
	var files []string
	more := false
	err := fs.WalkDir(root.FS(), name, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if !d.Type().IsRegular() {
			return nil
		}
		rel := strings.TrimPrefix(p, name+"/")
		if rel == SkillFile {
			return nil
		}
		if len(files) == maxListedFiles {
			more = true
			return fs.SkipAll
		}
		files = append(files, rel)
		return nil
	})
	if err != nil {
		return nil, false, fmt.Errorf("list files of skill %q: %w", name, err)
	}
	return files, more, nil
}

// ResourceInput holds the arguments of Resource.
type ResourceInput struct {
	SkillName    string `json:"skill_name" description:"Name of the skill"`
	ResourcePath string `json:"resource_path" description:"Path of the file inside the skill folder, such as references/api.md or assets/template.txt"`
}

// Resource returns the text of one of a skill's files.
func (s *Skills) Resource(_ context.Context, in ResourceInput) (string, error) {
	if !nameRE.MatchString(in.SkillName) {
		return "", fmt.Errorf("no skill named %q", in.SkillName)
	}
	rel, err := resourcePath(in.ResourcePath)
	if err != nil {
		return "", err
	}
	root, err := s.open()
	if err != nil {
		return "", err
	}
	defer root.Close()

	if _, err := root.Stat(filepath.Join(in.SkillName, SkillFile)); err != nil {
		return "", fmt.Errorf("no skill named %q", in.SkillName)
	}
	data, err := readFile(root, filepath.Join(in.SkillName, rel))
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return "", fmt.Errorf("skill %q has no file %s", in.SkillName, in.ResourcePath)
	case err != nil:
		return "", fmt.Errorf("%s: %w", in.ResourcePath, err)
	case !utf8.Valid(data) || bytes.IndexByte(data, 0) >= 0:
		return "", fmt.Errorf("%s is a binary file", in.ResourcePath)
	case len(data) == 0:
		return fmt.Sprintf("%s is empty.", in.ResourcePath), nil
	}
	return string(data), nil
}

// resourcePath checks a model-supplied path inside a skill folder and
// returns it with OS separators.
func resourcePath(p string) (string, error) {
	if p == "" {
		return "", errors.New("resource_path is required")
	}
	if strings.Contains(p, `\`) || !filepath.IsLocal(filepath.FromSlash(p)) {
		return "", fmt.Errorf("resource_path %q must be a relative path inside the skill folder, using forward slashes", p)
	}
	return filepath.Clean(filepath.FromSlash(p)), nil
}

// readFile reads a regular file of at most maxFileBytes.
func readFile(root *os.Root, rel string) ([]byte, error) {
	info, err := root.Stat(rel)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, errors.New("not a regular file")
	}
	f, err := root.Open(rel)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, maxFileBytes+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxFileBytes {
		return nil, fmt.Errorf("file is larger than %d bytes", maxFileBytes)
	}
	return data, nil
}

// File is a file to write inside a skill folder.
type File struct {
	Path    string `json:"path" description:"Path inside the skill folder, such as references/api.md"`
	Content string `json:"content" description:"Full file content"`
}

// SaveInput holds the arguments of Save.
type SaveInput struct {
	Name         string `json:"name" description:"Skill name: lowercase letters, digits and hyphens, at most 64 characters"`
	Description  string `json:"description" description:"What the skill does and when to use it, at most 1024 characters. This is all the model sees before loading the skill."`
	Instructions string `json:"instructions" description:"Markdown instructions: the full body of SKILL.md, without frontmatter"`
	Files        []File `json:"files,omitempty" description:"Other files to create or overwrite in the skill folder; files not listed are kept"`
}

// Save creates a skill or replaces its SKILL.md, and writes the given files.
// Other frontmatter fields of an existing skill, such as license, are kept.
// Everything is validated before anything is written.
func (s *Skills) Save(ctx context.Context, in SaveInput) (string, error) {
	fm := Frontmatter{Name: in.Name, Description: strings.TrimSpace(in.Description)}
	if err := Validate(fm); err != nil {
		return "", err
	}
	if strings.TrimSpace(in.Instructions) == "" {
		return "", errors.New("instructions are required")
	}
	if len(in.Files) > maxSaveFiles {
		return "", fmt.Errorf("at most %d files can be saved at once, got %d", maxSaveFiles, len(in.Files))
	}
	seen := make(map[string]bool, len(in.Files))
	for _, f := range in.Files {
		rel, err := resourcePath(f.Path)
		if err != nil {
			return "", err
		}
		if strings.EqualFold(filepath.ToSlash(rel), SkillFile) {
			return "", fmt.Errorf("%s is written from name, description and instructions; leave it out of files", SkillFile)
		}
		if seen[rel] {
			return "", fmt.Errorf("file %s is listed twice", f.Path)
		}
		seen[rel] = true
		if len(f.Content) > maxFileBytes {
			return "", fmt.Errorf("file %s is larger than %d bytes", f.Path, maxFileBytes)
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	root, err := s.open()
	if err != nil {
		return "", err
	}
	existing, err := readFile(root, filepath.Join(in.Name, SkillFile))
	root.Close()
	existed := err == nil
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return "", fmt.Errorf("skill %q: %w", in.Name, err)
	}
	front, err := frontmatter(existing, fm)
	if err != nil {
		return "", err
	}
	content := "---\n" + front + "---\n\n" + strings.TrimSpace(in.Instructions) + "\n"
	if len(content) > maxFileBytes {
		return "", fmt.Errorf("%s would be larger than %d bytes", SkillFile, maxFileBytes)
	}

	skillFile := in.Name + "/" + SkillFile
	// Other files first, so the skill only appears in the catalog once they exist.
	for _, f := range in.Files {
		if _, err := s.fs.WriteFile(ctx, filesystem.WriteFileInput{Path: in.Name + "/" + f.Path, Content: f.Content}); err != nil {
			return "", err
		}
	}
	if _, err := s.fs.WriteFile(ctx, filesystem.WriteFileInput{Path: skillFile, Content: content}); err != nil {
		return "", err
	}

	verb := "Created"
	if existed {
		verb = "Updated"
	}
	msg := fmt.Sprintf("%s skill %q", verb, in.Name)
	if len(in.Files) > 0 {
		msg += fmt.Sprintf(" and wrote %d other file(s)", len(in.Files))
	}
	return msg + ".", nil
}

// frontmatter renders the YAML frontmatter for a saved skill: the existing
// SKILL.md's fields with name and description replaced, or just those two.
func frontmatter(existing []byte, fm Frontmatter) (string, error) {
	node := &yaml.Node{Kind: yaml.MappingNode}
	if existing != nil {
		if n, _, err := split(existing); err == nil {
			node = n
		}
	}
	setKey(node, "name", fm.Name)
	setKey(node, "description", fm.Description)
	out, err := yaml.Marshal(node)
	if err != nil {
		return "", fmt.Errorf("frontmatter: %w", err)
	}
	return string(out), nil
}

// setKey sets a string value in a YAML mapping, keeping the key's position
// when it exists; name and description go first otherwise.
func setKey(m *yaml.Node, key, value string) {
	for i := 0; i+1 < len(m.Content); i += 2 {
		if m.Content[i].Value == key {
			m.Content[i+1] = &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: value}
			return
		}
	}
	pos := 0
	if key == "description" && len(m.Content) >= 2 && m.Content[0].Value == "name" {
		pos = 2
	}
	pair := []*yaml.Node{
		{Kind: yaml.ScalarNode, Tag: "!!str", Value: key},
		{Kind: yaml.ScalarNode, Tag: "!!str", Value: value},
	}
	m.Content = slices.Insert(m.Content, pos, pair...)
}
