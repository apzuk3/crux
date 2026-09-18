package crux

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/invopop/jsonschema"
)

type Agent struct {
	name         string
	maxTurns     int32
	instructions string

	provider     Provider
	model        string
	baseURL      string
	apikey       string
	outputSchema *jsonschema.Schema

	searchOptions *SearchOptions // nil disables web search

	tools []Tool

	logs []Entry
}

// SearchOptions configures provider-executed web search.
type SearchOptions struct {
	UserLocation *UserLocation // nil means no location is supplied
}

type SearchOption func(*SearchOptions)

func NewAgent(name, model string, opts ...AgentOption) (*Agent, error) {
	agent := &Agent{
		name:     name,
		model:    model,
		maxTurns: 10,
		tools:    make([]Tool, 0),
	}
	for _, opt := range opts {
		if err := opt(agent); err != nil {
			return nil, err
		}
	}

	if agent.provider == "" {
		agent.provider = inferProvider(agent.model)
	}

	if agent.provider == "" {
		return nil, fmt.Errorf("cannot infer provider from model %q, please pass through crux.WithProvider", agent.model)
	}

	if agent.apikey == "" {
		agent.apikey = discoverAPIKey(agent.provider)
	}

	if agent.baseURL == "" {
		switch agent.provider {
		case ProviderOpenrouter:
			agent.baseURL = "https://openrouter.ai/api/v1"
		case ProviderXAI:
			agent.baseURL = "https://api.x.ai/v1"
		case ProviderDeepSeek:
			agent.baseURL = "https://api.deepseek.com"
		case ProviderOllama:
			agent.baseURL = "http://localhost:11434/v1"
		}
	}

	if agent.provider == ProviderOllama && agent.apikey == "" {
		agent.apikey = "ollama" // Local Ollama ignores authentication.
	}

	return agent, nil
}

func Must(agent *Agent, err error) *Agent {
	if err != nil {
		panic(err)
	}

	return agent
}

// Run continues the retained conversation and returns its final text response.
// User inputs, model entries, and tool results are retained even if a later step
// fails. Run must not execute concurrently with other operations on the agent.
func (a *Agent) Run(ctx context.Context, input any) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}

	if len(a.PendingApprovals()) > 0 {
		return "", ErrApprovalNeeded
	}

	if input == nil && len(a.logs) == 0 {
		return "", errors.New("cannot run agent with no input and empty history")
	}

	if input == nil {
		if text, ok := a.FinalOutput(); ok {
			return text, nil
		}
	}

	if input != nil && a.hasUnexecutedToolCalls() {
		return "", errors.New("cannot run agent with new user input while tool calls are pending execution; call Resume first")
	}

	if input != nil {
		// The user turn is part of the log, so every provider sees one shape and a
		// resumed session needs nothing but its history.
		entry, err := NewUserEntry(input)
		if err != nil {
			return "", err
		}
		if strings.TrimSpace(entry.Text()) == "" {
			return "", errors.New("user input produced empty text")
		}

		a.logs = append(a.logs, entry)
	}

	// ---> Notify start
	for range a.maxTurns {
		if err := ctx.Err(); err != nil {
			return "", err
		}

		a.logs = append(a.logs, a.executeUnexecutedToolCalls(ctx)...)
		if err := ctx.Err(); err != nil {
			return "", err
		}

		var (
			produced []Entry
			err      error
		)

		switch a.provider {
		case ProviderAnthropic:
			produced, err = a.anthropicStep(ctx, a.logs)
			if err != nil {
				return "", err
			}
		case ProviderOpenAI:
			produced, err = a.openAIstep(ctx, a.logs)
			if err != nil {
				return "", err
			}
		case ProviderOpenrouter:
			produced, err = a.openrouterStep(ctx, a.logs)
			if err != nil {
				return "", err
			}
		case ProviderGoogle:
			produced, err = a.geminiStep(ctx, a.logs)
			if err != nil {
				return "", err
			}
		case ProviderXAI:
			produced, err = a.xaiStep(ctx, a.logs)
			if err != nil {
				return "", err
			}
		case ProviderDeepSeek:
			produced, err = a.deepseekStep(ctx, a.logs)
			if err != nil {
				return "", err
			}
		case ProviderOllama:
			produced, err = a.ollamaStep(ctx, a.logs)
			if err != nil {
				return "", err
			}
		default:
			return "", fmt.Errorf("unsupported provider %q", a.provider)
		}

		// Retain all model entries in history before dispatching local tools.
		a.logs = append(a.logs, produced...)

		if len(a.PendingApprovals()) > 0 {
			return "", ErrApprovalNeeded
		}

		if refusal, refused := latestRefusal(produced); refused {
			if refusal != "" {
				return "", fmt.Errorf("model refused the request: %s", refusal)
			}
			return "", errors.New("model refused the request")
		}

		// If the latest turn produced the final answer without requesting further tools:
		if text, ok := a.FinalOutput(); ok {
			// ---> Notify end
			return text, nil
		}
	}

	return "", errors.New("max turns reached")
}

// FinalOutput returns the final assistant text if the latest turn completed
// without requesting further tools, along with a boolean indicating completion.
func (a *Agent) FinalOutput() (string, bool) {
	if len(a.logs) == 0 || len(a.PendingApprovals()) > 0 || a.hasUnexecutedToolCalls() {
		return "", false
	}

	start := len(a.logs)
	for i := len(a.logs) - 1; i >= 0; i-- {
		kind := a.logs[i].Kind
		if kind == KindUser || kind == KindToolResult || a.logs[i].HiddenFromModel() {
			start = i + 1
			break
		}
		if i == 0 {
			start = 0
		}
	}

	if start >= len(a.logs) {
		return "", false
	}

	latestTurn := a.logs[start:]
	if _, refused := latestRefusal(latestTurn); refused {
		return "", false
	}

	var hasAssistant bool
	for _, e := range latestTurn {
		if e.Kind == KindToolCall {
			return "", false
		}
		if e.Kind == KindAssistant {
			hasAssistant = true
		}
	}

	if !hasAssistant {
		return "", false
	}

	return finalText(latestTurn), true
}

func latestRefusal(entries []Entry) (string, bool) {
	for _, entry := range entries {
		if entry.Kind == KindAssistant {
			for _, part := range entry.Content {
				if part.Kind == ContentKindRefusal {
					return part.Text, true
				}
			}
		}
	}
	return "", false
}

func (a *Agent) Resume(ctx context.Context) (string, error) {
	return a.Run(ctx, nil)
}

func (a *Agent) toolRequiresApproval(name string) bool {
	index := slices.IndexFunc(a.tools, func(tool Tool) bool { return tool.name == name })
	if index < 0 {
		return false
	}
	return a.tools[index].approvalNeeded
}

func (a *Agent) hasUnexecutedToolCalls() bool {
	resolved := make(map[string]bool)
	for _, entry := range a.logs {
		if entry.Kind == KindToolResult && entry.ToolResult != nil && entry.ToolResult.CallID != "" {
			resolved[entry.ToolResult.CallID] = true
		}
	}
	for _, entry := range a.logs {
		if entry.Kind == KindToolCall && entry.ToolCall != nil && entry.ToolCall.ID != "" {
			if !resolved[entry.ToolCall.ID] {
				return true
			}
		}
	}
	return false
}

func (a *Agent) executeUnexecutedToolCalls(ctx context.Context) []Entry {
	if len(a.PendingApprovals()) > 0 {
		return nil
	}

	resolved := make(map[string]bool)
	decisions := make(map[string]*Approval)
	for _, entry := range a.logs {
		if entry.Kind == KindToolResult && entry.ToolResult != nil && entry.ToolResult.CallID != "" {
			resolved[entry.ToolResult.CallID] = true
		}
		if entry.Kind == KindApproval && entry.Approval != nil && entry.Approval.CallID != "" {
			decisions[entry.Approval.CallID] = entry.Approval
		}
	}

	var unexecuted []*ToolCall
	seen := make(map[string]bool)
	for _, entry := range a.logs {
		if entry.Kind == KindToolCall && entry.ToolCall != nil && entry.ToolCall.ID != "" {
			id := entry.ToolCall.ID
			if !resolved[id] && !seen[id] {
				seen[id] = true
				unexecuted = append(unexecuted, entry.ToolCall)
			}
		}
	}

	if len(unexecuted) == 0 {
		return nil
	}

	state := a.StateSnapshot()
	var entries []Entry
	// Execute in the exact order requested by the model.
	for _, call := range unexecuted {
		if err := ctx.Err(); err != nil {
			break
		}

		dec := decisions[call.ID]
		if dec != nil && !dec.Approved {
			reason := dec.Reason
			if reason == "" {
				reason = "tool execution declined by user"
			}
			result := ToolResult{
				CallID: call.ID,
				Error:  reason,
			}
			entries = append(entries, Entry{Kind: KindToolResult, ToolResult: &result})
			continue
		}

		results, delta := a.dispatch(ctx, call, state)
		entries = append(entries, results...)
		if delta != nil {
			if delta.Set != nil {
				maps.Copy(state, delta.Set)
			}
			for _, key := range delta.Delete {
				delete(state, key)
			}
		}
	}

	return entries
}

func (a *Agent) PendingApprovals() []*ToolCall {
	resolved := make(map[string]bool)
	decisions := make(map[string]bool)
	for _, entry := range a.logs {
		if entry.Kind == KindToolResult && entry.ToolResult != nil && entry.ToolResult.CallID != "" {
			resolved[entry.ToolResult.CallID] = true
		}
		if entry.Kind == KindApproval && entry.Approval != nil && entry.Approval.CallID != "" {
			decisions[entry.Approval.CallID] = true
		}
	}

	var pending []*ToolCall
	seen := make(map[string]bool)
	for _, entry := range a.logs {
		if entry.Kind == KindToolCall && entry.ToolCall != nil && entry.ToolCall.ID != "" {
			id := entry.ToolCall.ID
			if !resolved[id] && !decisions[id] && !seen[id] && a.toolRequiresApproval(entry.ToolCall.Name) {
				seen[id] = true
				pending = append(pending, entry.ToolCall)
			}
		}
	}
	return pending
}

func (a *Agent) Approve(ctx context.Context, callID string) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	if callID == "" {
		return errors.New("tool call ID cannot be empty")
	}

	pending := a.PendingApprovals()
	idx := slices.IndexFunc(pending, func(p *ToolCall) bool {
		return p.ID == callID
	})
	if idx < 0 {
		return fmt.Errorf("tool call %q is not pending approval", callID)
	}

	a.logs = append(a.logs, Entry{
		Kind: KindApproval,
		Approval: &Approval{
			CallID:   callID,
			Approved: true,
		},
	})
	return nil
}

func (a *Agent) Reject(ctx context.Context, callID string, reason string) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	if callID == "" {
		return errors.New("tool call ID cannot be empty")
	}

	pending := a.PendingApprovals()
	idx := slices.IndexFunc(pending, func(p *ToolCall) bool {
		return p.ID == callID
	})
	if idx < 0 {
		return fmt.Errorf("tool call %q is not pending approval", callID)
	}

	if reason == "" {
		reason = "tool execution declined by user"
	}

	a.logs = append(a.logs, Entry{
		Kind: KindApproval,
		Approval: &Approval{
			CallID:   callID,
			Approved: false,
			Reason:   reason,
		},
	})
	return nil
}

// Run executes the agent and decodes its final text response into Out. Anything
// that is not text is read back as JSON.
func Run[Out any](ctx context.Context, a *Agent, input any) (Out, error) {
	text, err := a.Run(ctx, input)
	if err != nil {
		var zero Out
		return zero, err
	}
	return decodeOutput[Out](text)
}

// Resume executes the agent to continue after tool approvals/rejections and decodes
// its final text response into Out.
func Resume[Out any](ctx context.Context, a *Agent) (Out, error) {
	text, err := a.Resume(ctx)
	if err != nil {
		var zero Out
		return zero, err
	}
	return decodeOutput[Out](text)
}

// dispatch runs a tool call locally. A failure is reported to the model rather
// than returned, because the call is still owed an answer.
//
// dispatch returns the tool result entry and any state delta that the tool introduced.
func (a *Agent) dispatch(ctx context.Context, call *ToolCall, snapshot map[string]any) ([]Entry, *StateDelta) {
	if call == nil {
		return nil, nil
	}

	result := ToolResult{CallID: call.ID}
	index := slices.IndexFunc(a.tools, func(tool Tool) bool { return tool.name == call.Name })
	if index < 0 {
		result.Error = fmt.Sprintf("tool %q is not allowed", call.Name)
		return []Entry{{Kind: KindToolResult, ToolResult: &result}}, nil
	}

	tool := a.tools[index]

	output, delta, err := tool.invoke(ContextWithState(ctx, snapshot), call.Args)
	if err != nil {
		result.Error = err.Error()
	} else {
		result.Output = output
	}

	var resp = []Entry{
		{Kind: KindToolResult, ToolResult: &result},
	}

	if delta != nil {
		if delta.By == "" {
			delta.By = call.Name
		}

		resp = append(resp, Entry{Kind: KindStateDelta, Delta: delta})
	}

	return resp, delta
}

func finalText(entries []Entry) string {
	var text strings.Builder
	for _, entry := range entries {
		if entry.Kind == KindAssistant {
			text.WriteString(entry.Text())
		}
	}
	return text.String()
}

func (a *Agent) OutputSchema() *jsonschema.Schema {
	return a.outputSchema
}

// decodeOutput renders the model's final text as Out. Anything that is not
// text is read back as JSON.
func decodeOutput[Out any](text string) (Out, error) {
	var output Out

	switch target := any(&output).(type) {
	case *string:
		*target = text
		return output, nil
	case *any:
		*target = text
		return output, nil
	}

	clean := strings.TrimSpace(text)
	if err := json.Unmarshal([]byte(clean), &output); err == nil {
		return output, nil
	}

	if start := strings.Index(clean, "```"); start != -1 {
		rest := clean[start+3:]
		if end := strings.LastIndex(rest, "```"); end != -1 {
			block := rest[:end]
			if nl := strings.Index(block, "\n"); nl != -1 {
				block = block[nl+1:]
			} else if strings.HasPrefix(strings.ToLower(block), "json") {
				block = block[4:]
			}
			block = strings.TrimSpace(block)
			if err := json.Unmarshal([]byte(block), &output); err == nil {
				return output, nil
			}
		}
	}

	if err := json.Unmarshal([]byte(clean), &output); err != nil {
		return output, fmt.Errorf("decode agent output as %T: %w", output, err)
	}
	return output, nil
}
