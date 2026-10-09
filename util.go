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
	redacted := msg
	if u.User != nil {
		// As url.URL.String writes it, and as a raw string would show it.
		for _, userinfo := range []string{u.User.String(), rawUserinfo(raw)} {
			if userinfo != "" {
				redacted = strings.ReplaceAll(redacted, userinfo+"@", "redacted@")
			}
		}
	}
	for key, values := range u.Query() {
		for _, value := range values {
			if value == "" {
				continue
			}
			for _, form := range []string{url.QueryEscape(value), value} {
				redacted = strings.ReplaceAll(redacted, url.QueryEscape(key)+"="+form, url.QueryEscape(key)+"=redacted")
			}
		}
	}
	if redacted == msg {
		return err
	}
	return &redactedError{msg: redacted, err: err}
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
		e := &result[i]
		e.Content = slices.Clone(e.Content)
		if e.Reasoning != nil {
			value := *e.Reasoning
			e.Reasoning = &value
		}
		if e.ToolCall != nil {
			value := *e.ToolCall
			value.Args = slices.Clone(value.Args)
			e.ToolCall = &value
		}
		if e.ToolResult != nil {
			value := *e.ToolResult
			e.ToolResult = &value
		}
		if e.Delta != nil {
			value := *e.Delta
			value.Set = cloneState(value.Set)
			value.Delete = slices.Clone(value.Delete)
			e.Delta = &value
		}
		if e.Approval != nil {
			value := *e.Approval
			e.Approval = &value
		}
		if e.Usage != nil {
			value := *e.Usage
			e.Usage = &value
		}
		if e.Run != nil {
			value := *e.Run
			e.Run = &value
		}
		if e.Turn != nil {
			value := *e.Turn
			e.Turn = &value
		}
		if e.Response != nil {
			value := *e.Response
			e.Response = &value
		}
		if e.Opaque != nil {
			opaque := make(map[string][]byte, len(e.Opaque))
			for key, value := range e.Opaque {
				opaque[key] = slices.Clone(value)
			}
			e.Opaque = opaque
		}
	}
	return result
}
