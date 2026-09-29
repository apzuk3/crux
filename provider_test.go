package crux

import (
	"go/ast"
	"go/parser"
	"go/token"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Every exported model constant must be registered in its file's init, so New
// can infer the provider from it.
func TestModelConstantsAreRegistered(t *testing.T) {
	files, err := filepath.Glob("*_provider.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range files {
		parsed, err := parser.ParseFile(token.NewFileSet(), file, nil, 0)
		if err != nil {
			t.Fatal(err)
		}

		var constants []string
		registered := make(map[string]bool)
		for _, decl := range parsed.Decls {
			switch decl := decl.(type) {
			case *ast.GenDecl:
				if decl.Tok != token.CONST {
					continue
				}
				for _, spec := range decl.Specs {
					for _, name := range spec.(*ast.ValueSpec).Names {
						if name.IsExported() {
							constants = append(constants, name.Name)
						}
					}
				}
			case *ast.FuncDecl:
				if decl.Name.Name != "init" {
					continue
				}
				ast.Inspect(decl.Body, func(node ast.Node) bool {
					if kv, ok := node.(*ast.KeyValueExpr); ok {
						if key, ok := kv.Key.(*ast.Ident); ok && key.Name == "Name" {
							if value, ok := kv.Value.(*ast.Ident); ok {
								registered[value.Name] = true
							}
						}
					}
					return true
				})
			}
		}

		for _, name := range constants {
			if !registered[name] {
				t.Errorf("%s: model constant %s is not registered in init", file, name)
			}
		}
	}
}

func TestInferProviderForRegisteredModels(t *testing.T) {
	for model, want := range map[string]Provider{
		ClaudeOpus4_6:   ProviderAnthropic,
		ClaudeHaiku4_5:  ProviderAnthropic,
		ChatModelGPT5_4: ProviderOpenAI,
		Gemini3_8Flash:  ProviderGoogle,
	} {
		if got := inferProvider(model); got != want {
			t.Errorf("inferProvider(%q) = %q, want %q", model, got, want)
		}
	}
}

// Providers whose SDK would otherwise use a client without any timeout give up
// on a server that never answers instead of hanging the run forever.
func TestDefaultHTTPClientTimesOutWithoutResponseHeaders(t *testing.T) {
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-release
	}))
	defer server.Close()
	defer close(release)

	original := defaultHTTPClient
	client := newHTTPClient(50 * time.Millisecond)
	defaultHTTPClient = func() *http.Client { return client }
	t.Cleanup(func() { defaultHTTPClient = original })

	for _, provider := range []Provider{ProviderXAI, ProviderOllama, ProviderGoogle} {
		t.Run(string(provider), func(t *testing.T) {
			agent, err := New("a", "some-model", WithProvider(provider), WithBaseURL(server.URL), WithAPIKey("key"))
			if err != nil {
				t.Fatal(err)
			}
			session, err := NewSession(t.Context(), agent)
			if err != nil {
				t.Fatal(err)
			}

			done := make(chan error, 1)
			go func() {
				_, err := session.Run(t.Context(), "hi")
				done <- err
			}()
			select {
			case err := <-done:
				if err == nil || !strings.Contains(err.Error(), "timeout awaiting response headers") {
					t.Fatalf("expected a response header timeout, got %v", err)
				}
			case <-time.After(30 * time.Second):
				t.Fatal("request did not time out")
			}
		})
	}
}
