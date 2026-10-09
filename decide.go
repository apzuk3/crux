package crux

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"sync"
	"unicode/utf8"

	"crux.foo/internal/decide"
	"crux.foo/internal/provider"
	"crux.foo/internal/schema"
	"github.com/invopop/jsonschema"
)

// Decider answers typed questions about a state: yes or no, one of a set of
// options, or a level on a scale. Decision models such as TypeSafe's Jev
// answer natively, with calibrated probabilities; any other model answers
// through structured output. It is stateless and safe for concurrent use.
type Decider struct {
	agent  *Agent
	native bool
	client *http.Client // nil uses the default client
}

// NewDecider returns a Decider for model. It takes the agent options that
// configure a connection (WithProvider, WithAPIKey, WithBaseURL,
// WithMaxRetries) and WithInstructions, which every question
// is asked under. Models without a decision API also take sampling options
// such as WithTemperature.
func NewDecider(model string, opts ...AgentOption) (*Decider, error) {
	agent, err := newAgent("decider", model, opts...)
	if err != nil {
		return nil, err
	}
	switch {
	case len(agent.tools) > 0:
		return nil, errors.New("a decider takes no tools or subagents")
	case agent.searchOptions != nil:
		return nil, errors.New("a decider takes no web search")
	case agent.outputSchema != nil:
		return nil, errors.New("a decider takes no output schema; the type passed to Decide is the schema")
	case agent.toolChoice != "":
		return nil, errors.New("a decider takes no tool choice")
	case agent.compaction != (CompactionOptions{}) || agent.contextWindow != 0:
		return nil, errors.New("a decider takes no compaction settings")
	}
	d := &Decider{agent: agent, native: providerSpecs[agent.provider].decidesNatively(agent.model)}
	if d.native && (agent.maxTokens != 0 || agent.temperature != nil || agent.reasoning != "" || agent.parallel != nil) {
		return nil, fmt.Errorf("%q is a decision model and takes no sampling settings", model)
	}
	return d, nil
}

// WithHTTPClient returns a copy of d that sends its requests through client;
// d is unchanged. Nil uses the default client. In tests, pass cruxtest's
// Mock.Client.
func (d *Decider) WithHTTPClient(client *http.Client) *Decider {
	copied := *d
	copied.client = client
	return &copied
}

// MustDecider panics if err is not nil.
func MustDecider(decider *Decider, err error) *Decider {
	if err != nil {
		panic(err)
	}
	return decider
}

func (d *Decider) Model() string      { return d.agent.model }
func (d *Decider) Provider() Provider { return d.agent.provider }

// Decision is a Decider's answer. Confidence and Probabilities are keyed by
// question, the field's json name; Probabilities then by answer: "true" and
// "false", an option, or a level. Both are nil when the model is not a
// decision model, because its answers carry no calibrated probabilities.
type Decision[T any] struct {
	Value         T
	Confidence    map[string]float64
	Probabilities map[string]map[string]float64
	Model         string // as the provider reports it
	Usage         Usage
	Cost          float64 // in USD; 0 when the provider doesn't report it
}

// Decide asks d the questions T describes about state and returns the
// answers as a T. T is a struct, and each exported field is one question:
//
//   - a bool asks yes or no; the tags `true:"…"` and `false:"…"` say what
//     each answer means. It is true when yes has a probability of at least 0.5;
//   - a string picks one of the options in its `choices:"billing=Payments
//     and invoices|bug=Something is broken|other"` tag, or of a string type
//     with a Choices() map[string]string method;
//   - an integer or float rates against the levels of its
//     `levels:"none|minor|major"` tag, lowest first. An integer gets the
//     nearest level's index; a float gets the probability-weighted position,
//     which can fall between levels.
//
// The `description` tag tells the model what the question is. Other field
// types, such as free text, are rejected before any request.
//
// state is like Run's inputs: strings and values sent as JSON. More than one
// is sent as an array. Decision models take no Attachments.
func Decide[T any](ctx context.Context, d *Decider, state ...any) (Decision[T], error) {
	var out Decision[T]
	fields, err := decisionFields(reflect.TypeFor[T]())
	if err != nil {
		return out, err
	}
	var resp *provider.DecideResponse
	if d.native {
		resp, err = d.decideNatively(ctx, fields, state)
	} else {
		resp, err = d.decideWithOutput(ctx, fields, state)
	}
	if err != nil {
		return out, err
	}
	if err := decide.Assign(reflect.ValueOf(&out.Value).Elem(), fields, resp.Answers); err != nil {
		return out, fmt.Errorf("%s: %w", d.agent.provider, err)
	}
	out.Model, out.Cost = resp.Model, resp.Cost
	out.Usage = Usage{InputTokens: resp.Usage.InputTokens, OutputTokens: resp.Usage.OutputTokens}
	if d.native {
		out.Confidence = make(map[string]float64, len(fields))
		out.Probabilities = make(map[string]map[string]float64, len(fields))
		for _, f := range fields {
			a := resp.Answers[f.Name]
			if f.Kind == provider.QuestionNoul {
				out.Confidence[f.Name] = max(a.Noul, 1-a.Noul)
				out.Probabilities[f.Name] = map[string]float64{"true": a.Noul, "false": 1 - a.Noul}
				continue
			}
			out.Confidence[f.Name] = a.Confidence
			out.Probabilities[f.Name] = a.Probabilities
		}
	}
	return out, nil
}

var decisionFieldCache sync.Map // reflect.Type -> []decide.Field

func decisionFields(t reflect.Type) ([]decide.Field, error) {
	if cached, ok := decisionFieldCache.Load(t); ok {
		return cached.([]decide.Field), nil
	}
	fields, err := decide.Fields(t)
	if err != nil {
		return nil, err
	}
	decisionFieldCache.Store(t, fields)
	return fields, nil
}

func (d *Decider) decideNatively(ctx context.Context, fields []decide.Field, inputs []any) (*provider.DecideResponse, error) {
	state, err := decisionState(inputs)
	if err != nil {
		return nil, err
	}
	a := d.agent
	resp, err := providerSpecs[a.provider].decide(ctx, &provider.DecideRequest{
		Provider:   string(a.provider),
		Model:      a.model,
		APIKey:     a.apiKey,
		BaseURL:    a.baseURL,
		HTTPClient: d.client,
		MaxRetries: a.maxRetries,
		State:      state,
		Questions:  decide.Questions(fields, a.instructions),
	})
	if err != nil {
		if provider.IsContextTooLong(err) {
			err = fmt.Errorf("%w: %w", ErrContextTooLong, err)
		} else {
			err = providerError(a.provider, err)
		}
		return nil, redactURLSecrets(err, a.baseURL)
	}
	return resp, nil
}

// decisionState encodes inputs as a decision model's state: one input as
// itself, more as an array.
func decisionState(inputs []any) (json.RawMessage, error) {
	var parts []json.RawMessage
	for _, input := range inputs {
		var value any
		switch v := input.(type) {
		case nil:
			continue
		case Attachment, *Attachment:
			return nil, errors.New("decision models take text and JSON, not attachments")
		case json.RawMessage:
			if !json.Valid(v) {
				return nil, errors.New("decision state is invalid JSON")
			}
			parts = append(parts, v)
			continue
		case []byte:
			if !utf8.Valid(v) {
				return nil, errors.New("decision state is binary")
			}
			value = string(v)
		case fmt.Stringer:
			value = v.String()
		default:
			value = v
		}
		raw, err := json.Marshal(value)
		if err != nil {
			return nil, fmt.Errorf("unsupported decision state type %T: %w", input, err)
		}
		parts = append(parts, raw)
	}
	switch len(parts) {
	case 0:
		return nil, errors.New("cannot decide with no state")
	case 1:
		return parts[0], nil
	}
	return json.Marshal(parts)
}

// decideWithOutput asks a generative model for the answers as structured
// output. Its answers carry no probabilities.
func (d *Decider) decideWithOutput(ctx context.Context, fields []decide.Field, inputs []any) (*provider.DecideResponse, error) {
	entry, err := NewUserEntry(inputs...)
	if err != nil {
		return nil, err
	}
	if len(entry.Content) == 0 {
		return nil, errors.New("cannot decide with no state")
	}
	raw, err := json.Marshal(decide.OutputSchema(fields, ""))
	if err != nil {
		return nil, err
	}
	output := new(jsonschema.Schema)
	if err := json.Unmarshal(raw, output); err != nil {
		return nil, fmt.Errorf("decision schema: %w", err)
	}
	agent := *d.agent
	agent.outputSchema = output
	if agent.instructions == "" {
		agent.instructions = "Answer every question about the user's input."
	}

	entries, err := agent.step(ctx, d.client, []Entry{entry}, nil, true)
	if err != nil {
		return nil, err
	}
	var text string
	var usage *Usage
	for _, e := range entries {
		if e.Kind == KindAssistant {
			text = e.Text()
		}
		if e.Usage != nil {
			usage = e.Usage
		}
	}
	validator, err := schema.CompileOutput(output)
	if err != nil {
		return nil, fmt.Errorf("decision schema: %w", err)
	}
	if err := validateOutput(validator, text); err != nil {
		return nil, err
	}
	var values map[string]any
	if err := decodeInto(text, &values); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrOutputValidation, err)
	}

	resp := &provider.DecideResponse{Model: agent.model, Answers: make(map[string]provider.Answer, len(fields))}
	if usage != nil {
		resp.Usage = provider.Usage{InputTokens: usage.InputTokens, OutputTokens: usage.OutputTokens}
	}
	for _, f := range fields {
		a := provider.Answer{Kind: f.Kind}
		switch v := values[f.Name].(type) {
		case bool:
			if v {
				a.Noul = 1
			}
		case string:
			a.Choice = v
		case float64:
			a.Score = v
		}
		resp.Answers[f.Name] = a
	}
	return resp, nil
}
