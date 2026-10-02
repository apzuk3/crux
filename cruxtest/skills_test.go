package cruxtest_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/apzuk3/crux"
	"github.com/apzuk3/crux/cruxtest"
	"github.com/stretchr/testify/require"
)

// TestSkills covers every AddSkills case in one test, because AddSkills
// registers with the default registry and can only succeed once per process.
func TestSkills(t *testing.T) {
	_, err := crux.New("no-skills", crux.OpenAIGPT5_6Sol, crux.WithSkills())
	require.ErrorIs(t, err, crux.ErrToolNotFound)
	require.ErrorContains(t, err, "AddSkills")

	bad := t.TempDir()
	require.NoError(t, os.MkdirAll(filepath.Join(bad, "pdf"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(bad, "pdf", "SKILL.md"), []byte("---\nname: other\ndescription: x\n---\nbody"), 0o644))
	require.ErrorContains(t, crux.AddSkills(bad), `does not match its folder "pdf"`)
	require.Error(t, crux.AddSkills(filepath.Join(bad, "missing")))

	dir := t.TempDir()
	require.NoError(t, os.MkdirAll(filepath.Join(dir, "pdf", "references"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "pdf", "SKILL.md"), []byte("---\nname: pdf\ndescription: Fill in PDF forms.\n---\n\nRead references/forms.md first.\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "pdf", "references", "forms.md"), []byte("Use field names."), 0o644))
	require.NoError(t, crux.AddSkills(dir))
	require.Error(t, crux.AddSkills(dir), "the skill tools are already registered")

	for _, provider := range []crux.Provider{crux.ProviderOpenAI, crux.ProviderAnthropic, crux.ProviderGoogle} {
		t.Run(string(provider), func(t *testing.T) {
			mock := cruxtest.NewMock(cruxtest.WithProvider(provider))
			mock.Expect().ReturnToolCall(crux.SkillLoad, map[string]any{"name": "pdf"})
			mock.Expect().ReturnToolCall(crux.SkillLoadResource, map[string]any{"skill_name": "pdf", "resource_path": "references/forms.md"})
			mock.Expect().ReturnText("done")

			agent, err := crux.New("forms", "test-model", append(mock.AgentOptions(),
				crux.WithProvider(provider), crux.WithInstructions("Be brief."), crux.WithSkills())...)
			require.NoError(t, err)
			s, err := crux.NewSession(t.Context(), agent)
			require.NoError(t, err)
			out, err := s.Run(t.Context(), "Fill this form")
			require.NoError(t, err)
			require.Equal(t, "done", out)

			reqs := mock.Requests()
			first := unescapeHTML(reqs[0].BodyString())
			require.Contains(t, first, "Be brief.")
			require.Contains(t, first, "<name>pdf</name>")
			require.Contains(t, first, "Fill in PDF forms.")
			for _, name := range []string{crux.SkillLoad, crux.SkillLoadResource, crux.SkillSave} {
				require.Contains(t, first, `"`+name+`"`)
			}
			require.Contains(t, reqs[1].BodyString(), "Read references/forms.md first.")
			require.Contains(t, reqs[1].BodyString(), "- references/forms.md")
			require.Contains(t, reqs[2].BodyString(), "Use field names.")
			mock.AssertAllConsumed(t)
		})
	}

	t.Run("save", func(t *testing.T) {
		mock := cruxtest.NewMock()
		mock.Expect().ReturnToolCall(crux.SkillSave, map[string]any{
			"name":         "release-notes",
			"description":  "Write release notes from a list of merged changes.",
			"instructions": "Group changes by area.",
			"files":        []map[string]any{{"path": "assets/template.md", "content": "## Changes"}},
		})
		mock.Expect().ReturnText("saved")
		mock.Expect().ReturnText("ok")

		agent, err := crux.New("writer", crux.OpenAIGPT5_6Sol, append(mock.AgentOptions(), crux.WithSkills())...)
		require.NoError(t, err)
		s, err := crux.NewSession(t.Context(), agent)
		require.NoError(t, err)

		_, err = s.Run(t.Context(), "Remember how to write release notes")
		require.ErrorIs(t, err, crux.ErrApprovalNeeded)
		pending := s.PendingApprovals()
		require.Len(t, pending, 1)
		require.NoError(t, s.Approve(t.Context(), pending[0].ID))
		_, err = s.Resume(t.Context())
		require.NoError(t, err)

		data, err := os.ReadFile(filepath.Join(dir, "release-notes", "SKILL.md"))
		require.NoError(t, err)
		require.Equal(t, "---\nname: release-notes\ndescription: Write release notes from a list of merged changes.\n---\n\nGroup changes by area.\n", string(data))
		require.FileExists(t, filepath.Join(dir, "release-notes", "assets", "template.md"))
		require.Contains(t, mock.Requests()[1].BodyString(), `Created skill \"release-notes\"`)

		// The new skill is listed from the next request on, in any session.
		other, err := crux.NewSession(t.Context(), agent)
		require.NoError(t, err)
		_, err = other.Run(t.Context(), "hi")
		require.NoError(t, err)
		require.Contains(t, unescapeHTML(mock.Requests()[2].BodyString()), "<name>release-notes</name>")
		mock.AssertAllConsumed(t)
	})
}

// unescapeHTML undoes the \u003c and \u003e escapes encoding/json adds.
func unescapeHTML(body string) string {
	return strings.NewReplacer(`\u003c`, "<", `\u003e`, ">").Replace(body)
}
