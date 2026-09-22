package crux

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/invopop/jsonschema"
)

type Agent struct {
	SessionID uuid.UUID

	// agent properties
	Name         string
	MaxTurns     int32
	Instructions string

	Provider      Provider
	Model         string
	BaseURL       string
	OutputSchema  *jsonschema.Schema
	SearchOptions *SearchOptions // nil disables web search
	Tools         []Tool

	// API key for the provider MUST be provided or discovered from the environment.
	apikey string

	sessionLogs []Entry
}

// SearchOptions configures provider-executed web search.
type SearchOptions struct {
	UserLocation *UserLocation // nil means no location is supplied
}

type SearchOption func(*SearchOptions)

func NewAgent(name, model string, opts ...AgentOption) (*Agent, error) {
	agent := &Agent{
		SessionID: uuid.New(),
		Name:      name,
		Model:     model,
		MaxTurns:  10,
		Tools:     make([]Tool, 0),
	}
	for _, opt := range opts {
		if err := opt(agent); err != nil {
			return nil, err
		}
	}

	if agent.Provider == "" {
		agent.Provider = inferProvider(agent.Model)
	}

	if agent.Provider == "" {
		return nil, fmt.Errorf("cannot infer provider from model %q, please pass through crux.WithProvider", agent.Model)
	}

	if agent.apikey == "" {
		agent.apikey = discoverAPIKey(agent.Provider)
	}

	if agent.BaseURL == "" {
		switch agent.Provider {
		case ProviderOpenrouter:
			agent.BaseURL = "https://openrouter.ai/api/v1"
		case ProviderXAI:
			agent.BaseURL = "https://api.x.ai/v1"
		case ProviderDeepSeek:
			agent.BaseURL = "https://api.deepseek.com"
		case ProviderOllama:
			agent.BaseURL = "http://localhost:11434/v1"
		}
	}

	if agent.Provider == ProviderOllama && agent.apikey == "" {
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

	if input == nil && len(a.sessionLogs) == 0 {
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

		a.appendLogs(entry)
	}

	// ---> Notify start
	for range a.MaxTurns {
		if err := ctx.Err(); err != nil {
			return "", err
		}

		if toolResults := a.executeUnexecutedToolCalls(ctx); len(toolResults) > 0 {
			a.appendLogs(toolResults...)
		}
		if err := ctx.Err(); err != nil {
			return "", err
		}

		start := time.Now()
		produced, err := a.step(ctx, a.sessionLogs)
		if err != nil {
			return "", err
		}
		if len(produced) > 0 {
			produced[len(produced)-1].Duration = time.Since(start)
		}

		// Retain all model entries in history before dispatching local tools.
		if len(produced) > 0 {
			a.appendLogs(produced...)
		}

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

func (a *Agent) step(ctx context.Context, log []Entry) ([]Entry, error) {
	switch a.Provider {
	case ProviderAnthropic:
		return a.anthropicStep(ctx, log)
	case ProviderOpenAI:
		return a.openAIstep(ctx, log)
	case ProviderOpenrouter:
		return a.openrouterStep(ctx, log)
	case ProviderGoogle:
		return a.geminiStep(ctx, log)
	case ProviderXAI:
		return a.xaiStep(ctx, log)
	case ProviderDeepSeek:
		return a.deepseekStep(ctx, log)
	case ProviderOllama:
		return a.ollamaStep(ctx, log)
	default:
		return nil, fmt.Errorf("unsupported provider %q", a.Provider)
	}
}

// FinalOutput returns the final assistant text if the latest turn completed
// without requesting further tools, along with a boolean indicating completion.
func (a *Agent) FinalOutput() (string, bool) {
	if len(a.sessionLogs) == 0 || len(a.PendingApprovals()) > 0 || a.hasUnexecutedToolCalls() {
		return "", false
	}

	start := len(a.sessionLogs)
	for i, v := range slices.Backward(a.sessionLogs) {
		kind := v.Kind
		if kind == KindUser || kind == KindToolResult || v.HiddenFromModel() {
			start = i + 1
			break
		}
		if i == 0 {
			start = 0
		}
	}

	if start >= len(a.sessionLogs) {
		return "", false
	}

	latestTurn := a.sessionLogs[start:]
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
	index := slices.IndexFunc(a.Tools, func(tool Tool) bool { return tool.name == name })
	if index < 0 {
		return false
	}
	return a.Tools[index].approvalNeeded
}

func (a *Agent) hasUnexecutedToolCalls() bool {
	resolved := make(map[string]bool)
	for _, entry := range a.sessionLogs {
		if entry.Kind == KindToolResult && entry.ToolResult != nil && entry.ToolResult.CallID != "" {
			resolved[entry.ToolResult.CallID] = true
		}
	}
	for _, entry := range a.sessionLogs {
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
	for _, entry := range a.sessionLogs {
		if entry.Kind == KindToolResult && entry.ToolResult != nil && entry.ToolResult.CallID != "" {
			resolved[entry.ToolResult.CallID] = true
		}
		if entry.Kind == KindApproval && entry.Approval != nil && entry.Approval.CallID != "" {
			decisions[entry.Approval.CallID] = entry.Approval
		}
	}

	var unexecuted []*ToolCall
	seen := make(map[string]bool)
	for _, entry := range a.sessionLogs {
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
			entries = append(entries, Entry{Kind: KindToolResult, ToolResult: &result, At: time.Now().UTC()})
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
	for _, entry := range a.sessionLogs {
		if entry.Kind == KindToolResult && entry.ToolResult != nil && entry.ToolResult.CallID != "" {
			resolved[entry.ToolResult.CallID] = true
		}
		if entry.Kind == KindApproval && entry.Approval != nil && entry.Approval.CallID != "" {
			decisions[entry.Approval.CallID] = true
		}
	}

	var pending []*ToolCall
	seen := make(map[string]bool)
	for _, entry := range a.sessionLogs {
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

	a.appendLogs(Entry{
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

	a.appendLogs(Entry{
		Kind: KindApproval,
		Approval: &Approval{
			CallID:   callID,
			Approved: false,
			Reason:   reason,
		},
	})
	return nil
}

// RunInto executes the agent and decodes its final response into target.
// target must be a non-nil pointer. Anything that is not text is decoded as JSON.
func (a *Agent) RunInto(ctx context.Context, input any, target any) error {
	text, err := a.Run(ctx, input)
	if err != nil {
		return err
	}
	return decodeInto(text, target)
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
	index := slices.IndexFunc(a.Tools, func(tool Tool) bool { return tool.name == call.Name })
	if index < 0 {
		result.Error = fmt.Sprintf("tool %q is not allowed", call.Name)
		return []Entry{{Kind: KindToolResult, ToolResult: &result}}, nil
	}

	tool := a.Tools[index]

	start := time.Now()
	output, delta, err := tool.invoke(ContextWithState(ctx, snapshot), call.Args)
	duration := time.Since(start)
	if err != nil {
		result.Error = err.Error()
	} else {
		result.Output = output
	}

	var resp = []Entry{
		{Kind: KindToolResult, ToolResult: &result, At: start.UTC(), Duration: duration},
	}

	if delta != nil {
		if delta.By == "" {
			delta.By = call.Name
		}

		resp = append(resp, Entry{Kind: KindStateDelta, Delta: delta, At: time.Now().UTC()})
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

func decodeInto(text string, target any) error {
	if target == nil {
		return errors.New("decode target cannot be nil")
	}

	rv := reflect.ValueOf(target)
	if rv.Kind() != reflect.Pointer || rv.IsNil() {
		return fmt.Errorf("decode target must be a non-nil pointer, got %T", target)
	}

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

	clean := strings.TrimSpace(text)
	if err := json.Unmarshal([]byte(clean), target); err == nil {
		return nil
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
			if err := json.Unmarshal([]byte(block), target); err == nil {
				return nil
			}
		}
	}

	if err := json.Unmarshal([]byte(clean), target); err != nil {
		return fmt.Errorf("decode agent output as %T: %w", target, err)
	}
	return nil
}

// Logs returns a clone of the agent's session log entries.
func (a *Agent) Logs() []Entry {
	return cloneEntries(a.sessionLogs)
}

func (a *Agent) appendLogs(entries ...Entry) {
	now := time.Now().UTC()
	for i := range entries {
		if entries[i].Seq == 0 {
			entries[i].Seq = uint64(len(a.sessionLogs) + 1)
		}
		if entries[i].At.IsZero() {
			entries[i].At = now
		}
		a.sessionLogs = append(a.sessionLogs, entries[i])
	}
}
