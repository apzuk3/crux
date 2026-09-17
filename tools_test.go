package crux

import (
	"errors"
	"slices"
	"testing"
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
