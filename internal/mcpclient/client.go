// Package mcpclient holds the parts of crux's MCP client that do not depend
// on crux types: OAuth for remote servers, the token store, result rendering
// and transport helpers.
package mcpclient

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"sync"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// RenderResult turns a tool result into the text the model sees. Text and
// text resources are passed on; other content is described. A result with no
// text gives its structured content as JSON. A tool error becomes an error.
func RenderResult(result *mcp.CallToolResult) (string, error) {
	var parts []string
	text := false
	for _, content := range result.Content {
		switch c := content.(type) {
		case *mcp.TextContent:
			parts = append(parts, c.Text)
			text = true
		case *mcp.ImageContent:
			parts = append(parts, fmt.Sprintf("[image %s, %d bytes]", c.MIMEType, len(c.Data)))
		case *mcp.AudioContent:
			parts = append(parts, fmt.Sprintf("[audio %s, %d bytes]", c.MIMEType, len(c.Data)))
		case *mcp.ResourceLink:
			parts = append(parts, fmt.Sprintf("[resource %s]", c.URI))
		case *mcp.EmbeddedResource:
			if c.Resource == nil {
				continue
			}
			if c.Resource.Text != "" {
				parts = append(parts, c.Resource.Text)
				text = true
			} else {
				parts = append(parts, fmt.Sprintf("[resource %s, %d bytes]", c.Resource.URI, len(c.Resource.Blob)))
			}
		}
	}
	if !text && result.StructuredContent != nil {
		raw, err := json.Marshal(result.StructuredContent)
		if err != nil {
			return "", fmt.Errorf("encode structured content: %w", err)
		}
		parts = append(parts, string(raw))
	}
	output := strings.Join(parts, "\n")
	if result.IsError {
		if output == "" {
			output = "tool failed"
		}
		return "", errors.New(output)
	}
	return output, nil
}

// HeaderTransport adds Header to every request it sends through Base.
type HeaderTransport struct {
	Base   http.RoundTripper
	Header http.Header
}

func (t HeaderTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	for key, values := range t.Header {
		req.Header[key] = slices.Clone(values)
	}
	return t.Base.RoundTrip(req)
}

// TailBuffer keeps the last bytes written to it, for a server's stderr.
type TailBuffer struct {
	mu  sync.Mutex
	max int
	buf []byte
}

// NewTailBuffer returns a buffer that keeps the last max bytes.
func NewTailBuffer(max int) *TailBuffer {
	return &TailBuffer{max: max}
}

func (b *TailBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.buf = append(b.buf, p...)
	if over := len(b.buf) - b.max; over > 0 {
		b.buf = b.buf[over:]
	}
	return len(p), nil
}

func (b *TailBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return strings.TrimSpace(string(b.buf))
}
