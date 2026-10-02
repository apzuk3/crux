package crux

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/apzuk3/crux/internal/skills"
)

// Skill tool names.
const (
	SkillLoad         = "load_skill"
	SkillLoadResource = "load_skill_resource"
	SkillSave         = "save_skill"
)

// ToolsetSkills is the name of the toolset registered by AddSkills.
const ToolsetSkills = "skills"

// AddSkills registers the skills in dir with the default registry. Each skill
// is a subdirectory named like the skill, holding a SKILL.md (YAML
// frontmatter with a name and a description, then markdown instructions) and
// any other files the instructions refer to. Agents use them with WithSkills.
//
// An agent with skills is told on every request which skills exist (their
// names and descriptions) and gets three tools: load_skill reads a skill's
// instructions, load_skill_resource reads one of its other files, and
// save_skill creates or updates a skill and needs approval. A saved skill is
// listed from the next request on, in every session.
//
// AddSkills fails when dir is not an existing directory or holds an invalid
// skill. It can be called once per process; dir may hold no skills yet.
func AddSkills(dir string) error {
	return AddToolset(&skillsToolset{dir: dir})
}

// WithSkills gives the agent the skills registered with AddSkills. Like
// WithToolsets, it adds to the agent's tools.
func WithSkills() AgentOption {
	return func(a *Agent) error {
		if err := WithToolsets(ToolsetSkills)(a); err != nil {
			if errors.Is(err, ErrToolNotFound) {
				return fmt.Errorf("%w (register skills with AddSkills before New)", err)
			}
			return err
		}
		return nil
	}
}

type skillsToolset struct {
	dir string
}

func (s *skillsToolset) Register(registry ToolsRegistry) error {
	dir, err := filepath.Abs(s.dir)
	if err != nil {
		return fmt.Errorf("skills directory %q: %w", s.dir, err)
	}
	info, err := os.Stat(dir)
	if err != nil {
		return fmt.Errorf("skills directory: %w", err)
	}
	if !info.IsDir() {
		return fmt.Errorf("skills directory %q is not a directory", s.dir)
	}
	sk := skills.New(dir)
	if err := sk.Check(); err != nil {
		return err
	}

	inSet := WithToolset(ToolsetSkills)
	registerTextTool(registry, SkillLoad, "Load a skill's instructions by name, with the list of its other files. Load a skill before doing a task it covers.", sk.Load, inSet, withInstructions(sk.Instructions))
	registerTextTool(registry, SkillLoadResource, "Read one of a skill's files other than SKILL.md, such as references/api.md.", sk.Resource, inSet)
	registerTextTool(registry, SkillSave, "Create a skill, or update one by replacing its SKILL.md. Other files of the skill are kept unless listed in files, which overwrites them.", sk.Save, inSet, WithApprovalNeeded(true))
	return nil
}
