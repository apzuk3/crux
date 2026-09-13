package crux

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestWebSearchLocationRequests(t *testing.T) {
	zero, longitude := 0.0, -0.1278
	location := UserLocation{
		Country: "GB", City: "London", Region: "London", Timezone: "Europe/London",
		Latitude: &zero, Longitude: &longitude,
	}
	for _, provider := range []Provider{ProviderOpenAI, ProviderAnthropic, ProviderGoogle, ProviderXAI} {
		for _, scenario := range []string{"disabled", "no location", "location", "reset"} {
			t.Run(string(provider)+"/"+scenario, func(t *testing.T) {
				var request map[string]any
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
						t.Error(err)
					}
					w.Header().Set("Content-Type", "application/json")
					switch provider {
					case ProviderAnthropic:
						_, _ = w.Write([]byte(`{"type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}`))
					case ProviderGoogle:
						_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
					default:
						_, _ = w.Write([]byte(`{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`))
					}
				}))
				defer server.Close()
				opts := []AgentOption{WithProvider(provider), WithAPIKey("test"), WithBaseURL(server.URL), WithTools(NewToolsRegistry())}
				switch scenario {
				case "no location":
					opts = append(opts, WithWebSearch())
				case "location":
					opts = append(opts, WithWebSearch(WithUserLocation(location)))
				case "reset":
					opts = append(opts, WithWebSearch(WithUserLocation(location)), WithWebSearch())
				}
				if _, err := NewAgent("test-model", opts...).Run(context.Background(), "hello"); err != nil {
					t.Fatal(err)
				}
				tools, _ := request["tools"].([]any)
				if scenario == "disabled" {
					if len(tools) != 0 {
						t.Fatalf("search enabled by default: %v", tools)
					}
					return
				}
				if len(tools) != 1 {
					t.Fatalf("expected one search tool: %v", tools)
				}
				tool := tools[0].(map[string]any)
				if provider == ProviderGoogle {
					if scenario != "location" {
						if _, ok := request["toolConfig"]; ok {
							t.Fatalf("unexpected location config: %v", request["toolConfig"])
						}
						return
					}
					want := map[string]any{"retrievalConfig": map[string]any{"latLng": map[string]any{"latitude": zero, "longitude": longitude}}}
					if !reflect.DeepEqual(request["toolConfig"], want) {
						t.Fatalf("location config = %v, want %v", request["toolConfig"], want)
					}
					return
				}
				if scenario != "location" || provider == ProviderXAI {
					if _, ok := tool["user_location"]; ok {
						t.Fatalf("unexpected location: %v", tool)
					}
					return
				}
				want := map[string]any{"type": "approximate", "country": "GB", "city": "London", "region": "London", "timezone": "Europe/London"}
				if !reflect.DeepEqual(tool["user_location"], want) {
					t.Fatalf("location = %v, want %v", tool["user_location"], want)
				}
			})
		}
	}
}
