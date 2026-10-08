package provider

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// Decide asks a decision model typed questions about a state and returns one
// answer per question.
type Decide func(ctx context.Context, req *DecideRequest) (*DecideResponse, error)

// QuestionKind is the type of a question a decision model answers.
type QuestionKind string

const (
	QuestionNoul   QuestionKind = "noul"   // yes or no
	QuestionChoice QuestionKind = "choice" // one option of a set
	QuestionScore  QuestionKind = "score"  // a position on ordered levels
)

// Question is one typed question. Options are a choice's options or a score's
// levels, lowest first; True and False describe a noul's answers.
type Question struct {
	Name         string
	Kind         QuestionKind
	Instructions string
	Options      []Option
	True, False  string
}

// Option is a choice option or score level. A choice option's Description
// tells the model when to pick it.
type Option struct {
	Name        string
	Description string
}

// DecideRequest is everything a Decide needs.
type DecideRequest struct {
	Provider   string
	Model      string
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client // nil uses DefaultHTTPClient
	MaxRetries *int         // nil retries twice
	State      json.RawMessage
	Questions  []Question
}

// Answer is a decision model's answer to one question. Noul is the
// probability of yes; Score is the probability-weighted level index, which
// can fall between levels. Probabilities are keyed by option name, or by
// level name for scores.
type Answer struct {
	Kind          QuestionKind
	Noul          float64
	Choice        string
	Score         float64
	Confidence    float64
	Probabilities map[string]float64
}

// DecideResponse holds the answers by question name.
type DecideResponse struct {
	Model   string
	Answers map[string]Answer
	Usage   Usage
	Cost    float64 // in USD; 0 when the provider doesn't report it
}

// Decisions returns a Decide for the state-and-questions API that TypeSafe
// defined for Jev and OpenRouter also serves. path is resolved against the
// base URL, so "../alpha/decisions" on "https://openrouter.ai/api/v1" posts to
// "https://openrouter.ai/api/alpha/decisions".
func Decisions(path string) Decide {
	return func(ctx context.Context, req *DecideRequest) (*DecideResponse, error) {
		endpoint, err := resolveEndpoint(req.BaseURL, path)
		if err != nil {
			return nil, err
		}
		body, err := json.Marshal(decisionsRequest(req))
		if err != nil {
			return nil, fmt.Errorf("%s: encode request: %w", req.Provider, err)
		}
		client := req.HTTPClient
		if client == nil {
			client = DefaultHTTPClient()
		}
		maxRetries := 2
		if req.MaxRetries != nil {
			maxRetries = *req.MaxRetries
		}

		for attempt := 0; ; attempt++ {
			resp, err := postJSON(ctx, client, endpoint, req.APIKey, body)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", req.Provider, err)
			}
			raw, err := io.ReadAll(resp.Body)
			resp.Body.Close()
			if err != nil {
				return nil, fmt.Errorf("%s: read response: %w", req.Provider, err)
			}
			if resp.StatusCode == http.StatusOK {
				return decodeDecisions(req, raw)
			}
			if attempt < maxRetries && retryable(resp.StatusCode) {
				if err := sleepCtx(ctx, retryDelay(resp.Header.Get("Retry-After"), attempt)); err != nil {
					return nil, err
				}
				continue
			}
			return nil, fmt.Errorf("%s: %s: %s", req.Provider, resp.Status, bytes.TrimSpace(raw))
		}
	}
}

func resolveEndpoint(baseURL, path string) (string, error) {
	base, err := url.Parse(strings.TrimRight(baseURL, "/") + "/")
	if err != nil || base.Host == "" {
		return "", fmt.Errorf("invalid base URL %q", baseURL)
	}
	ref, err := url.Parse(path)
	if err != nil {
		return "", err
	}
	return base.ResolveReference(ref).String(), nil
}

func postJSON(ctx context.Context, client *http.Client, endpoint, apiKey string, body []byte) (*http.Response, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	if apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	}
	return client.Do(httpReq)
}

// retryable reports whether a status is worth another try: rate limits,
// overload (529) and server errors.
func retryable(status int) bool {
	return status == http.StatusTooManyRequests || status == http.StatusRequestTimeout || status >= 500
}

func retryDelay(retryAfter string, attempt int) time.Duration {
	if seconds, err := strconv.Atoi(retryAfter); err == nil && seconds >= 0 {
		return min(time.Duration(seconds)*time.Second, time.Minute)
	}
	return 500 * time.Millisecond << attempt
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

type wireQuestion struct {
	Type         QuestionKind `json:"type"`
	Instructions string       `json:"instructions,omitempty"`
	Criteria     any          `json:"criteria,omitempty"`
}

type wireNoulCriteria struct {
	True  string `json:"true,omitempty"`
	False string `json:"false,omitempty"`
}

func decisionsRequest(req *DecideRequest) map[string]any {
	questions := make(map[string]wireQuestion, len(req.Questions))
	for _, q := range req.Questions {
		wq := wireQuestion{Type: q.Kind, Instructions: q.Instructions}
		switch q.Kind {
		case QuestionNoul:
			if q.True != "" || q.False != "" {
				wq.Criteria = wireNoulCriteria{True: q.True, False: q.False}
			}
		case QuestionChoice:
			criteria := make(map[string]string, len(q.Options))
			for _, o := range q.Options {
				criteria[o.Name] = cmp.Or(o.Description, o.Name)
			}
			wq.Criteria = criteria
		case QuestionScore:
			levels := make([]string, len(q.Options))
			for i, o := range q.Options {
				levels[i] = o.Name
			}
			wq.Criteria = levels
		}
		questions[q.Name] = wq
	}
	return map[string]any{"model": req.Model, "state": req.State, "questions": questions}
}

type wireAnswer struct {
	Type          QuestionKind       `json:"type"`
	Noul          float64            `json:"noul"`
	Choice        string             `json:"choice"`
	Score         float64            `json:"score"`
	Confidence    float64            `json:"confidence"`
	Legend        map[string]string  `json:"legend"`
	Probabilities map[string]float64 `json:"probabilities"`
}

func decodeDecisions(req *DecideRequest, raw []byte) (*DecideResponse, error) {
	var body struct {
		Model   string                `json:"model"`
		Answers map[string]wireAnswer `json:"answers"`
		Usage   struct {
			InputTokens  int     `json:"input_tokens"`
			OutputTokens int     `json:"output_tokens"`
			Cost         float64 `json:"cost"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, fmt.Errorf("%s: decode response: %w", req.Provider, err)
	}
	out := &DecideResponse{
		Model:   body.Model,
		Answers: make(map[string]Answer, len(body.Answers)),
		Usage:   Usage{InputTokens: body.Usage.InputTokens, OutputTokens: body.Usage.OutputTokens},
		Cost:    body.Usage.Cost,
	}
	for _, q := range req.Questions {
		wa, ok := body.Answers[q.Name]
		if !ok {
			return nil, fmt.Errorf("%s: no answer to question %q", req.Provider, q.Name)
		}
		if wa.Type != q.Kind {
			return nil, fmt.Errorf("%s: question %q is a %s, got a %s answer", req.Provider, q.Name, q.Kind, wa.Type)
		}
		answer := Answer{Kind: wa.Type, Noul: wa.Noul, Choice: wa.Choice, Score: wa.Score, Confidence: wa.Confidence, Probabilities: wa.Probabilities}
		if q.Kind == QuestionScore {
			// Score probabilities are keyed by level index; name them.
			answer.Probabilities = make(map[string]float64, len(wa.Probabilities))
			for key, p := range wa.Probabilities {
				name := wa.Legend[key]
				if i, err := strconv.Atoi(key); name == "" && err == nil && i >= 0 && i < len(q.Options) {
					name = q.Options[i].Name
				}
				answer.Probabilities[cmp.Or(name, key)] = p
			}
		}
		out.Answers[q.Name] = answer
	}
	return out, nil
}
