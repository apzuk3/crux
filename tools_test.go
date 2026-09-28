package crux

import (
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestToolsRegistrySelected(t *testing.T) {
	registry := NewToolsRegistry()
	for _, name := range []string{"alpha", "beta", "gamma"} {
		registry.tools[name] = Tool{name: name}
	}

	for _, tt := range []struct {
		name      string
		allowed   []string
		want      []string
		expectErr error
	}{
		{name: "nil"},
		{name: "empty", allowed: []string{}},
		{name: "subset", allowed: []string{"beta"}, want: []string{"beta"}},
		{name: "allowed order", allowed: []string{"gamma", "alpha"}, want: []string{"gamma", "alpha"}},
		{name: "unknown", allowed: []string{"missing", "beta"}, expectErr: ErrToolNotFound},
		{name: "all unknown", allowed: []string{"missing"}, expectErr: ErrToolNotFound},
		{name: "duplicates", allowed: []string{"gamma", "alpha", "gamma", "alpha"}, want: []string{"gamma", "alpha"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tools, err := registry.selected(tt.allowed)
			if tt.expectErr != nil {
				if !errors.Is(err, tt.expectErr) {
					t.Fatalf("selected error = %v, want %v", err, tt.expectErr)
				}
				if !strings.Contains(err.Error(), "missing") {
					t.Fatalf("selected error %q does not contain missing tool name", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			var got []string
			for _, tool := range tools {
				got = append(got, tool.name)
			}
			if !slices.Equal(got, tt.want) {
				t.Fatalf("selected names = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestToolsRegistryInToolsets(t *testing.T) {
	registry := NewToolsRegistry()
	for _, tool := range []Tool{
		{name: "write", toolset: "fs"},
		{name: "read", toolset: "fs"},
		{name: "search", toolset: "web"},
		{name: "loose"},
	} {
		registry.tools[tool.name] = tool
	}

	for _, tt := range []struct {
		name     string
		toolsets []string
		want     []string
	}{
		{name: "one", toolsets: []string{"fs"}, want: []string{"read", "write"}},
		{name: "order", toolsets: []string{"web", "fs"}, want: []string{"search", "read", "write"}},
		{name: "duplicates", toolsets: []string{"fs", "web", "fs"}, want: []string{"read", "write", "search"}},
		{name: "none"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tools, err := registry.inToolsets(tt.toolsets)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			var got []string
			for _, tool := range tools {
				got = append(got, tool.name)
			}
			if !slices.Equal(got, tt.want) {
				t.Fatalf("names = %v, want %v", got, tt.want)
			}
		})
	}

	_, err := registry.inToolsets([]string{"fs", "missing"})
	if !errors.Is(err, ErrToolNotFound) || !strings.Contains(err.Error(), `toolset "missing"`) {
		t.Fatalf("error = %v, want ErrToolNotFound for toolset missing", err)
	}
}

type schemaNode struct {
	Name     string       `json:"name"`
	Children []schemaNode `json:"children,omitempty"`
	Parent   *schemaNode  `json:"parent,omitempty"`
}

type schemaBase struct {
	Tenant string `json:"tenant"`
	Shadow string `json:"shadow"`
}

type SchemaMeta struct {
	Trace string `json:"trace,omitempty"`
}

type schemaInput struct {
	schemaBase
	*SchemaMeta
	Shadow  int             `json:"shadow"`
	When    time.Time       `json:"when"`
	Blob    []byte          `json:"blob"`
	Raw     json.RawMessage `json:"raw"`
	Count   int             `json:"count,string"`
	Skipped string          `json:"-"`
}

func TestJSONSchemaRecursiveType(t *testing.T) {
	schema := jsonSchemaOf[schemaNode]()
	props := schema["properties"].(map[string]any)
	children := props["children"].(map[string]any)
	if children["type"] != "array" || children["items"].(map[string]any)["type"] != "object" {
		t.Fatalf("children schema = %v", children)
	}
	if parent := props["parent"].(map[string]any); parent["type"] != "object" {
		t.Fatalf("parent schema = %v", parent)
	}
	if _, err := json.Marshal(schema); err != nil {
		t.Fatalf("marshal schema: %v", err)
	}
}

func TestJSONSchemaFollowsEncodingJSON(t *testing.T) {
	schema := jsonSchemaOf[schemaInput]()
	props := schema["properties"].(map[string]any)

	want := map[string]map[string]any{
		"tenant": {"type": "string"},
		"trace":  {"type": "string"},
		"shadow": {"type": "integer"},
		"when":   {"type": "string", "format": "date-time"},
		"blob":   {"type": "string", "description": "Base64-encoded bytes"},
		"raw":    {},
		"count":  {"type": "string"},
	}
	if len(props) != len(want) {
		t.Fatalf("properties = %v, want keys of %v", props, want)
	}
	for name, w := range want {
		got, _ := json.Marshal(props[name])
		exp, _ := json.Marshal(w)
		if string(got) != string(exp) {
			t.Errorf("property %q = %s, want %s", name, got, exp)
		}
	}

	required := schema["required"].([]string)
	if slices.Contains(required, "trace") || !slices.Contains(required, "tenant") {
		t.Fatalf("required = %v; want tenant required and trace (via embedded pointer) optional", required)
	}

	// The schema must describe what the tool actually decodes.
	var in schemaInput
	args := `{"tenant":"acme","trace":"t1","shadow":2,"when":"2026-01-02T03:04:05Z","blob":"aGk=","raw":{"a":1},"count":"7"}`
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		t.Fatalf("decode args matching schema: %v", err)
	}
	if in.Tenant != "acme" || in.Trace != "t1" || in.Shadow != 2 || string(in.Blob) != "hi" || in.Count != 7 {
		t.Fatalf("decoded = %+v", in)
	}
}
