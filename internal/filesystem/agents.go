package filesystem

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"unicode/utf8"
)

const (
	// AgentsFile is the file of instructions for agents working in a directory.
	AgentsFile = "AGENTS.md"

	maxAgentsFileBytes = 32 << 10
)

// AgentsInstructions returns instructions quoting the AGENTS.md at the root,
// or "" when there is none. The file is cut at 32 KB.
func (t *Tools) AgentsInstructions() (string, error) {
	root, err := t.open()
	if err != nil {
		return "", err
	}
	defer root.Close()

	f, _, err := openRegular(root, AgentsFile)
	if errors.Is(err, fs.ErrNotExist) {
		return "", nil
	}
	if err != nil {
		return "", fsError(AgentsFile, err)
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, maxAgentsFileBytes+1))
	if err != nil {
		return "", fsError(AgentsFile, err)
	}
	truncated := len(data) > maxAgentsFileBytes
	if truncated {
		data = data[:maxAgentsFileBytes]
		for len(data) > 0 && !utf8.Valid(data) {
			data = data[:len(data)-1]
		}
	}
	data = bytes.TrimSpace(data)
	if len(data) == 0 {
		return "", nil
	}

	text := fmt.Sprintf("The directory you work in has an %s with instructions for working in it. Follow them.\n\n<agents_md>\n%s\n</agents_md>", AgentsFile, data)
	if truncated {
		text += fmt.Sprintf("\n[%s is longer than %d bytes; read the rest with read_file.]", AgentsFile, maxAgentsFileBytes)
	}
	return text, nil
}
