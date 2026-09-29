package crux

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSearchRejectsOversizedQueries(t *testing.T) {
	registry, _ := newFilesystemRegistry(t, map[string]string{"a.txt": "needle\n"})

	slow := strings.Repeat(`([\w\W]{1000})`, 20) + "X"
	wantFSError(t, registry, "search_files_content", searchFilesContentInput{Query: slow, IsRegex: true}, "too complex")
	wantFSError(t, registry, "search_files_content", searchFilesContentInput{Query: `[\w\W]{1000}`, IsRegex: true}, "too complex")
	wantFSError(t, registry, "search_files_content", searchFilesContentInput{Query: strings.Repeat("a", fsMaxQueryLength+1)}, "longer than")

	got := mustFSTool(t, registry, "search_files_content", searchFilesContentInput{Query: `ne+dle\b`, IsRegex: true})
	require.Equal(t, "a.txt:1:1: needle", got)
}

func TestRegexSearchReadsOnlyTheStartOfLongLines(t *testing.T) {
	old := fsMaxRegexLine
	fsMaxRegexLine = 1024
	t.Cleanup(func() { fsMaxRegexLine = old })

	registry, _ := newFilesystemRegistry(t, map[string]string{
		"a.txt": strings.Repeat("x", 10) + "needle" + strings.Repeat("x", 2000) + "\n" +
			strings.Repeat("x", 1500) + "needle\n",
	})
	got := mustFSTool(t, registry, "search_files_content", searchFilesContentInput{Query: "needle", IsRegex: true})
	require.True(t, strings.HasPrefix(got, "a.txt:1:11: "), got)
	require.NotContains(t, got, "a.txt:2:")
	require.Contains(t, got, "[Lines longer than 1 KB were searched only in their first 1 KB.]")

	// Plain text search is linear, so it still reads whole lines.
	got = mustFSTool(t, registry, "search_files_content", searchFilesContentInput{Query: "needle"})
	require.Contains(t, got, "a.txt:2:1501: ")
	require.NotContains(t, got, "Lines longer than")
}
