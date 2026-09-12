package crux

import (
	"slices"
	"testing"
)

func TestToolsRegistrySelected(t *testing.T) {
	registry := NewToolsRegistry()
	for _, name := range []string{"alpha", "beta", "gamma"} {
		registry.tools[name] = Tool{name: name}
	}

	for _, tt := range []struct {
		name    string
		allowed []string
		want    []string
	}{
		{name: "nil"},
		{name: "empty", allowed: []string{}},
		{name: "subset", allowed: []string{"beta"}, want: []string{"beta"}},
		{name: "allowed order", allowed: []string{"gamma", "alpha"}, want: []string{"gamma", "alpha"}},
		{name: "unknown", allowed: []string{"missing", "beta"}, want: []string{"beta"}},
		{name: "all unknown", allowed: []string{"missing"}},
		{name: "duplicates", allowed: []string{"gamma", "alpha", "gamma", "alpha"}, want: []string{"gamma", "alpha"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var got []string
			for _, tool := range registry.selected(tt.allowed) {
				got = append(got, tool.name)
			}
			if !slices.Equal(got, tt.want) {
				t.Fatalf("selected names = %v, want %v", got, tt.want)
			}
		})
	}
}
