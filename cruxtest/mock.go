package cruxtest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"crux.foo"
)

// MockOption configures a Mock instance.
type MockOption func(*Mock)

// WithProvider forces the mock to encode wire responses for a specific provider
// instead of auto-detecting from the request URL/headers.
func WithProvider(p crux.Provider) MockOption {
	return func(m *Mock) {
		m.forcedProvider = p
	}
}

// Mock is an in-memory mock HTTP transport that intercepts API calls to LLM providers
// and returns expected responses configured as sequential turns.
type Mock struct {
	mu             sync.Mutex
	turns          []*Turn
	requests       []*CapturedRequest
	callCount      int
	forcedProvider crux.Provider
}

// NewMock creates a new Mock instance.
func NewMock(opts ...MockOption) *Mock {
	m := &Mock{
		turns:    make([]*Turn, 0),
		requests: make([]*CapturedRequest, 0),
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

// Expect enqueues a new expected Turn and returns it for configuration.
func (m *Mock) Expect() *Turn {
	m.mu.Lock()
	defer m.mu.Unlock()
	turn := &Turn{
		StatusCode: http.StatusOK,
	}
	m.turns = append(m.turns, turn)
	return turn
}

// Client returns an *http.Client backed by this mock transport.
func (m *Mock) Client() *http.Client {
	return &http.Client{Transport: m}
}

// AgentOptions returns standard crux.AgentOption helpers: WithHTTPClient and a mock WithAPIKey.
func (m *Mock) AgentOptions() []crux.AgentOption {
	return []crux.AgentOption{
		crux.WithHTTPClient(m.Client()),
		crux.WithAPIKey("cruxtest-mock-key"),
	}
}

// Calls returns the number of HTTP requests processed so far.
func (m *Mock) Calls() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.callCount
}

// Requests returns a copy of all captured requests.
func (m *Mock) Requests() []*CapturedRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	copied := make([]*CapturedRequest, len(m.requests))
	copy(copied, m.requests)
	return copied
}

// AssertAllConsumed verifies that all expected turns were consumed by requests.
func (m *Mock) AssertAllConsumed(t testing.TB) {
	t.Helper()
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.turns) > 0 {
		t.Fatalf("cruxtest: %d expected turn(s) were not consumed", len(m.turns))
	}
}

// AssertTurnCount verifies that exactly expectedCount requests were processed.
func (m *Mock) AssertTurnCount(t testing.TB, expectedCount int) {
	t.Helper()
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.callCount != expectedCount {
		t.Fatalf("cruxtest: expected %d call(s), got %d", expectedCount, m.callCount)
	}
}

// RoundTrip intercepts outgoing HTTP requests and responds with the next expected Turn.
func (m *Mock) RoundTrip(req *http.Request) (*http.Response, error) {
	m.mu.Lock()
	var bodyBytes []byte
	if req.Body != nil {
		var err error
		bodyBytes, err = io.ReadAll(req.Body)
		if err != nil {
			m.mu.Unlock()
			return nil, fmt.Errorf("cruxtest: read request body: %w", err)
		}
		req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
	}

	captured := &CapturedRequest{
		Method: req.Method,
		URL:    req.URL,
		Header: req.Header.Clone(),
		Body:   bodyBytes,
	}
	m.requests = append(m.requests, captured)
	m.callCount++

	provider := m.forcedProvider
	if provider == "" {
		provider = detectProvider(req)
	}
	if err := validateRequest(provider, bodyBytes); err != nil {
		// The expected turn stays queued, so AssertAllConsumed fails too.
		m.mu.Unlock()
		return invalidRequestResponse(provider, req, err), nil
	}

	if len(m.turns) == 0 {
		count := m.callCount
		m.mu.Unlock()
		return nil, fmt.Errorf("cruxtest: unexpected request #%d to %s %s: no more expected turns configured", count, req.Method, req.URL.String())
	}

	turn := m.turns[0]
	m.turns = m.turns[1:]
	callIndex := m.callCount
	m.mu.Unlock()

	var respBytes []byte
	var err error

	switch provider {
	case providerDecisions:
		respBytes, err = buildDecisionsResponse(turn)
	case crux.ProviderGoogle:
		respBytes, err = buildGeminiResponse(turn, callIndex)
	case crux.ProviderAnthropic:
		respBytes, err = buildAnthropicResponse(turn, callIndex)
	default: // OpenAI, OpenRouter, xAI, DeepSeek, Ollama
		respBytes, err = buildOpenAIResponse(turn, callIndex)
	}

	if err != nil {
		return nil, fmt.Errorf("cruxtest: build mock response: %w", err)
	}

	statusCode := turn.StatusCode
	if statusCode == 0 {
		statusCode = http.StatusOK
	}

	header := make(http.Header)
	header.Set("Content-Type", "application/json")
	var request struct {
		Stream bool `json:"stream"`
	}
	_ = json.Unmarshal(bodyBytes, &request)
	if (request.Stream || strings.Contains(req.URL.Path, ":streamGenerateContent")) && statusCode == http.StatusOK {
		header.Set("Content-Type", "text/event-stream")
		// Raw fixtures may supply exact SSE, including malformed/truncated streams.
		if len(turn.RawBody) == 0 {
			respBytes, err = buildStreamResponse(provider, respBytes)
			if err != nil {
				return nil, err
			}
		}
	}

	for key, value := range turn.Header {
		header.Set(key, value)
	}
	resp := &http.Response{
		StatusCode:    statusCode,
		Status:        fmt.Sprintf("%d %s", statusCode, http.StatusText(statusCode)),
		Header:        header,
		Body:          io.NopCloser(bytes.NewReader(respBytes)),
		ContentLength: int64(len(respBytes)),
		Request:       req,
	}

	return resp, nil
}

func detectProvider(req *http.Request) crux.Provider {
	path := req.URL.Path
	host := req.URL.Host

	if isDecisionsPath(path) {
		return providerDecisions
	}
	if strings.Contains(path, ":generateContent") || strings.Contains(path, ":streamGenerateContent") || strings.Contains(host, "googleapis.com") {
		return crux.ProviderGoogle
	}
	if strings.Contains(path, "/messages") || req.Header.Get("anthropic-version") != "" || strings.Contains(host, "anthropic.com") {
		return crux.ProviderAnthropic
	}
	return crux.ProviderOpenAI
}
