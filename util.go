package crux

import (
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"slices"
	"strings"

	"crux.foo/internal/schema"
)

// redactURLSecrets removes the credentials a base URL carries from err's
// message, because provider SDKs print the request URL in their errors.
// errors.Is and errors.As still see the original error.
func redactURLSecrets(err error, raw string) error {
	u, parseErr := url.Parse(raw)
	if parseErr != nil || (u.User == nil && u.RawQuery == "") {
		return err
	}
	msg := err.Error()
	redacted := redactQuery(redactUserinfo(msg, u, raw), u.Query())
	if redacted == msg {
		return err
	}
	return &redactedError{msg: redacted, err: err}
}

// redactUserinfo replaces the credentials of u in msg, as url.URL.String
// writes them and as the raw URL shows them.
func redactUserinfo(msg string, u *url.URL, raw string) string {
	if u.User == nil {
		return msg
	}
	for _, userinfo := range []string{u.User.String(), rawUserinfo(raw)} {
		if userinfo != "" {
			msg = strings.ReplaceAll(msg, userinfo+"@", "redacted@")
		}
	}
	return msg
}

// redactQuery replaces every non-empty query value in msg, escaped and raw.
func redactQuery(msg string, query url.Values) string {
	for key, values := range query {
		for _, value := range values {
			if value == "" {
				continue
			}
			for _, form := range []string{url.QueryEscape(value), value} {
				msg = strings.ReplaceAll(msg, url.QueryEscape(key)+"="+form, url.QueryEscape(key)+"=redacted")
			}
		}
	}
	return msg
}

// rawUserinfo returns the user info of a URL exactly as written.
func rawUserinfo(raw string) string {
	_, rest, ok := strings.Cut(raw, "://")
	if !ok {
		return ""
	}
	authority, _, _ := strings.Cut(rest, "/")
	userinfo, _, found := strings.Cut(authority, "@")
	if !found {
		return ""
	}
	return userinfo
}

type redactedError struct {
	msg string
	err error
}

func (e *redactedError) Error() string { return e.msg }
func (e *redactedError) Unwrap() error { return e.err }

// isTextTarget reports whether decodeInto takes the answer as it is for
// target, with no JSON decoding that could fail.
func isTextTarget(target any) bool {
	switch target.(type) {
	case *string, *any, *[]byte, *json.RawMessage:
		return true
	}
	return false
}

// decodeInto decodes a final answer into target, a non-nil pointer. Text
// targets take the answer as is; anything else is decoded as JSON, which may
// be wrapped in a fenced code block.
func decodeInto(text string, target any) error {
	switch dest := target.(type) {
	case *string:
		*dest = text
		return nil
	case *any:
		*dest = text
		return nil
	case *[]byte:
		*dest = []byte(text)
		return nil
	case *json.RawMessage:
		*dest = json.RawMessage(text)
		return nil
	}

	candidates := schema.JSONCandidates(text)
	for _, candidate := range candidates {
		if json.Unmarshal([]byte(candidate), target) == nil {
			return nil
		}
	}
	err := json.Unmarshal([]byte(candidates[0]), target)
	return fmt.Errorf("decode agent output as %T: %w", target, err)
}

func firstEnv(names []string) string {
	for _, name := range names {
		if value := os.Getenv(name); value != "" {
			return value
		}
	}
	return ""
}

func cloneState(state map[string]any) map[string]any {
	if state == nil {
		return nil
	}
	out := make(map[string]any, len(state))
	for key, value := range state {
		out[key] = cloneStateValue(value)
	}
	return out
}

func cloneStateValue(value any) any {
	switch value := value.(type) {
	case map[string]any:
		return cloneState(value)
	case []any:
		out := slices.Clone(value)
		for i, item := range value {
			out[i] = cloneStateValue(item)
		}
		return out
	case []byte:
		return slices.Clone(value)
	case json.RawMessage:
		return slices.Clone(value)
	default:
		return value
	}
}

func cloneSearchOptions(s *SearchOptions) *SearchOptions {
	if s == nil {
		return nil
	}
	search := *s
	search.UserLocation = cloneUserLocation(s.UserLocation)
	return &search
}

// cloneUserLocation copies the location with its coordinates, so the caller
// can change its own value afterwards.
func cloneUserLocation(l *UserLocation) *UserLocation {
	if l == nil {
		return nil
	}
	loc := *l
	if l.Latitude != nil {
		lat := *l.Latitude
		loc.Latitude = &lat
	}
	if l.Longitude != nil {
		long := *l.Longitude
		loc.Longitude = &long
	}
	return &loc
}

func cloneEntries(entries []Entry) []Entry {
	result := slices.Clone(entries)
	for i := range result {
		cloneEntry(&result[i])
	}
	return result
}

// cloneEntry replaces what e points to with copies, so the copy and the
// original share nothing.
func cloneEntry(e *Entry) {
	e.Content = slices.Clone(e.Content)
	e.Reasoning = clonePtr(e.Reasoning)
	e.ToolCall = cloneToolCall(e.ToolCall)
	e.ToolResult = clonePtr(e.ToolResult)
	e.Delta = cloneDelta(e.Delta)
	e.Approval = clonePtr(e.Approval)
	e.Usage = clonePtr(e.Usage)
	e.Run = clonePtr(e.Run)
	e.Turn = clonePtr(e.Turn)
	e.Response = clonePtr(e.Response)
	e.Opaque = cloneOpaque(e.Opaque)
}

// clonePtr returns a pointer to a copy of *p, or nil for nil.
func clonePtr[T any](p *T) *T {
	if p == nil {
		return nil
	}
	value := *p
	return &value
}

func cloneToolCall(c *ToolCall) *ToolCall {
	if c == nil {
		return nil
	}
	value := *c
	value.Args = slices.Clone(value.Args)
	return &value
}

func cloneDelta(d *StateDelta) *StateDelta {
	if d == nil {
		return nil
	}
	value := *d
	value.Set = cloneState(value.Set)
	value.Delete = slices.Clone(value.Delete)
	return &value
}

func cloneOpaque(opaque map[string][]byte) map[string][]byte {
	if opaque == nil {
		return nil
	}
	out := make(map[string][]byte, len(opaque))
	for key, value := range opaque {
		out[key] = slices.Clone(value)
	}
	return out
}
