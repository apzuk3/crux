package crux

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"reflect"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/invopop/jsonschema"
)

// Agent is the immutable blueprint defining an agent's identity,
// capabilities, instructions, and provider connection.
// It is completely stateless and safe for concurrent use.
type Agent struct {
	id           uuid.UUID
	name         string
	model        string
	instructions string
	provider     Provider
	apiKey       string
	baseURL      string
	httpClient   *http.Client

	// Capabilities & Schemas
	tools         []Tool
	searchOptions *SearchOptions // nil disables web search
	outputSchema  *jsonschema.Schema

	// Execution Policies & Limits
	maxTurns    int
	maxRepairs  int
	maxTokens   int             // 0 uses the provider default
	temperature *float64        // nil uses the provider default
	reasoning   ReasoningEffort // "" uses the provider default
}

type Session struct {
	id         uuid.UUID
	parentID   uuid.UUID // session whose tool call created this one; uuid.Nil at the top level
	agent      *Agent
	logs       []Entry // cache of the entries the store has accepted
	httpClient *http.Client
	store      Store
	children   map[string]*Session // subagent sessions waiting for approval, by toolCallInfo.key
}

// sessionContextKey carries the session running a tool, so sessions created
// inside that tool (such as subagents) are recorded as its children.
type sessionContextKey struct{}

// toolCallContextKey carries the *toolCallInfo of the call a tool runs for.
type toolCallContextKey struct{}

// toolCallInfo identifies a running tool call and lets a subagent tool report
// that its session stopped for approval.
type toolCallInfo struct {
	key     string   // unique within the session: the call's entry Seq and ID
	waiting *Session // set by a subagent whose session waits for approval
}

// errToolWaiting is returned by a subagent tool whose session waits for
// approval. The call then gets no result and runs again once it is decided.
var errToolWaiting = errors.New("tool is waiting for approval")

// childSessionID derives the session ID of a subagent call, so a call that
// stopped for approval continues in the same session when it runs again.
func childSessionID(parent uuid.UUID, callKey string) uuid.UUID {
	return uuid.NewSHA1(parent, []byte(callKey))
}

// NewSession starts a conversation with agent. Sessions are kept in an
// in-memory store unless WithStore supplies another one.
//
// When the store already holds a session with the ID given by WithSessionID,
// its history is loaded so the conversation continues where it left off.
// History seeded with WithSessionLogs is written to the store, and cannot be
// combined with an ID that already has history.
func NewSession(ctx context.Context, agent *Agent, opts ...SessionOption) (*Session, error) {
	if ctx == nil {
		return nil, errors.New("context cannot be nil")
	}
	if agent == nil {
		return nil, errors.New("agent cannot be nil")
	}

	session := &Session{
		id:    uuid.New(),
		agent: agent,
		logs:  make([]Entry, 0),
	}

	for _, opt := range opts {
		if err := opt(session); err != nil {
			return nil, err
		}
	}

	if session.id == uuid.Nil {
		return nil, errors.New("session ID cannot be nil")
	}
	if parent, ok := ctx.Value(sessionContextKey{}).(*Session); ok && parent.id != session.id {
		session.parentID = parent.id
		if session.store == nil {
			session.store = parent.store
		}
	}
	if session.store == nil {
		session.store = NewMemoryStore()
	}

	stored, err := session.store.Get(ctx, session.id)
	if err != nil && !errors.Is(err, ErrSessionNotFound) {
		return nil, fmt.Errorf("load session %s: %w", session.id, err)
	}

	switch {
	case len(stored) > 0 && len(session.logs) > 0:
		return nil, fmt.Errorf("session %s already has history; WithSessionLogs cannot replace it", session.id)
	case len(stored) > 0:
		session.logs = stored
		if err := session.loadWaitingChildren(ctx); err != nil {
			return nil, err
		}
	case len(session.logs) > 0:
		now := time.Now().UTC()
		var prev uint64
		for i := range session.logs {
			if session.logs[i].Seq <= prev {
				session.logs[i].Seq = prev + 1
			}
			prev = session.logs[i].Seq
			if session.logs[i].At.IsZero() {
				session.logs[i].At = now
			}
		}
		if err := session.store.Append(ctx, session, session.logs...); err != nil {
			return nil, fmt.Errorf("persist session logs: %w", err)
		}
	}

	return session, nil
}

// loadWaitingChildren finds the subagent sessions that stopped for approval
// inside the session's open tool calls, so PendingApprovals lists their calls
// once the session is loaded again.
func (s *Session) loadWaitingChildren(ctx context.Context) error {
	for _, open := range s.openToolCalls() {
		index := slices.IndexFunc(s.agent.tools, func(tool Tool) bool { return tool.name == open.call.Name })
		if index < 0 || s.agent.tools[index].subAgent == nil {
			continue
		}
		child, err := NewSession(context.WithValue(ctx, sessionContextKey{}, s), s.agent.tools[index].subAgent,
			WithSessionID(childSessionID(s.id, open.key())), WithStore(s.store))
		if err != nil {
			return fmt.Errorf("load subagent session of call %q: %w", open.call.ID, err)
		}
		if len(child.PendingApprovals()) > 0 {
			if s.children == nil {
				s.children = make(map[string]*Session)
			}
			s.children[open.key()] = child
		}
	}
	return nil
}

func MustSession(session *Session, err error) *Session {
	if err != nil {
		panic(err)
	}

	return session
}

func (s *Session) ID() uuid.UUID {
	return s.id
}

func (s *Session) Agent() *Agent {
	return s.agent
}

func (s *Session) Logs() []Entry {
	return cloneEntries(s.logs)
}

// Usage returns the tokens used by every model request in the session.
// A fork starts at zero: history copied by Fork carries no usage.
func (s *Session) Usage() Usage {
	var total Usage
	for _, entry := range s.logs {
		if entry.Usage == nil {
			continue
		}
		total.InputTokens += entry.Usage.InputTokens
		total.OutputTokens += entry.Usage.OutputTokens
		total.CacheReadTokens += entry.Usage.CacheReadTokens
		total.CacheWriteTokens += entry.Usage.CacheWriteTokens
	}
	return total
}

func (s *Session) Store() Store {
	return s.store
}

// appendLogs persists entries and, once the store accepts them, adds them to
// the session. The store is the source of truth: after a failed write the
// session is unchanged, so the entries are produced again on the next run.
func (s *Session) appendLogs(ctx context.Context, entries ...Entry) error {
	now := time.Now().UTC()
	var last uint64
	if n := len(s.logs); n > 0 {
		last = s.logs[n-1].Seq
	}
	for i := range entries {
		entries[i].Seq = last + uint64(i) + 1
		if entries[i].At.IsZero() {
			entries[i].At = now
		}
	}

	if err := s.store.Append(ctx, s, entries...); err != nil {
		return fmt.Errorf("persist session logs: %w", err)
	}
	s.logs = append(s.logs, entries...)

	return nil
}

// SearchOptions configures provider-executed web search.
type SearchOptions struct {
	UserLocation *UserLocation // nil means no location is supplied
}

type SearchOption func(*SearchOptions)

func New(name, model string, opts ...AgentOption) (*Agent, error) {
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

	seen := make(map[string]bool, len(agent.tools))
	for _, tool := range agent.tools {
		if seen[tool.name] {
			return nil, fmt.Errorf("agent %q has two tools named %q", agent.name, tool.name)
		}
		seen[tool.name] = true
	}

	if agent.provider == "" {
		return nil, fmt.Errorf("cannot infer provider from model %q, please pass through crux.WithProvider", agent.model)
	}

	if agent.provider == ProviderAnthropic && agent.temperature != nil && agent.reasoning != "" && agent.reasoning != ReasoningOff {
		return nil, errors.New("anthropic does not accept a temperature while the model reasons; remove WithTemperature or use WithReasoning(ReasoningOff)")
	}

	if agent.apiKey == "" {
		agent.apiKey = discoverAPIKey(agent.provider)
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

	if agent.provider == ProviderOllama && agent.apiKey == "" {
		agent.apiKey = "ollama" // Local Ollama ignores authentication.
	}

	if agent.id == uuid.Nil {
		agent.id = agent.computeAgentID()
	}

	return agent, nil
}

func Must(agent *Agent, err error) *Agent {
	if err != nil {
		panic(err)
	}

	return agent
}

// agentNamespace is the base UUID namespace for computing deterministic agent IDs.
var agentNamespace = uuid.MustParse("e0f4f9a0-6f91-4c74-9844-3bfa3eb238b1")

func (a *Agent) CanonicalData() []byte {
	var toolNames []string
	if len(a.tools) > 0 {
		toolNames = make([]string, len(a.tools))
		for i, t := range a.tools {
			toolNames[i] = t.name
		}
	}

	data := map[string]any{
		"name":           a.name,
		"model":          a.model,
		"provider":       a.provider,
		"instructions":   a.instructions,
		"base_url":       redactURL(a.baseURL),
		"max_turns":      a.maxTurns,
		"max_repairs":    a.maxRepairs,
		"max_tokens":     a.maxTokens,
		"temperature":    a.temperature,
		"tools":          toolNames,
		"search_options": a.searchOptions,
		"output_schema":  a.outputSchema,
	}
	// Added only when set, so the IDs of existing agents do not change.
	if a.reasoning != "" {
		data["reasoning"] = a.reasoning
	}
	raw, _ := json.Marshal(data)
	return raw
}

// redactURL removes credentials a base URL may carry in its user info or
// query, because the canonical data is stored in plain text (GORMStore saves
// it in crux_agents).
func redactURL(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	if u.User != nil {
		u.User = url.User("redacted")
	}
	if u.RawQuery != "" {
		query := u.Query()
		for key := range query {
			query[key] = []string{"redacted"}
		}
		u.RawQuery = query.Encode()
	}
	return u.String()
}

// redactURLSecrets removes the credentials a base URL carries from err's
// message, because provider SDKs print the request URL in their errors.
// errors.Is and errors.As still see the original error.
func redactURLSecrets(err error, raw string) error {
	u, parseErr := url.Parse(raw)
	if parseErr != nil || (u.User == nil && u.RawQuery == "") {
		return err
	}
	msg := err.Error()
	redacted := msg
	if u.User != nil {
		// As url.URL.String writes it, and as a raw string would show it.
		for _, userinfo := range []string{u.User.String(), rawUserinfo(raw)} {
			if userinfo != "" {
				redacted = strings.ReplaceAll(redacted, userinfo+"@", "redacted@")
			}
		}
	}
	for key, values := range u.Query() {
		for _, value := range values {
			if value == "" {
				continue
			}
			for _, form := range []string{url.QueryEscape(value), value} {
				redacted = strings.ReplaceAll(redacted, url.QueryEscape(key)+"="+form, url.QueryEscape(key)+"=redacted")
			}
		}
	}
	if redacted == msg {
		return err
	}
	return &redactedError{msg: redacted, err: err}
}

// rawUserinfo returns the user info of a URL exactly as written.
func rawUserinfo(raw string) string {
	_, rest, ok := strings.Cut(raw, "://")
	if !ok {
		return ""
	}
	authority, _, _ := strings.Cut(rest, "/")
	userinfo, _, found := strings.Cut(authority, "@")
	if !found {
		return ""
	}
	return userinfo
}

type redactedError struct {
	msg string
	err error
}

func (e *redactedError) Error() string { return e.msg }
func (e *redactedError) Unwrap() error { return e.err }

func (a *Agent) computeAgentID() uuid.UUID {
	return uuid.NewSHA1(agentNamespace, a.CanonicalData())
}

func (a *Agent) ID() uuid.UUID {
	return a.id
}

// Provider returns the provider the agent sends requests to.
func (a *Agent) Provider() Provider {
	return a.provider
}

func (a *Agent) Name() string {
	return a.name
}

func (a *Agent) Model() string {
	return a.model
}

func (a *Agent) Instructions() string {
	return a.instructions
}

func (a *Agent) BaseURL() string {
	return a.baseURL
}

func (a *Agent) MaxTurns() int {
	return a.maxTurns
}

func (a *Agent) MaxRepairs() int {
	return a.maxRepairs
}

func (a *Agent) ToolNames() []string {
	names := make([]string, len(a.tools))
	for i, t := range a.tools {
		names[i] = t.name
	}
	return names
}

// Run continues the retained conversation and returns its final text response.
// User inputs, model entries, and tool results are retained even if a later step
// fails, so retry a failed Run with Resume: calling Run again with the same
// input would add it to the conversation twice. Run must not execute
// concurrently with other operations on the session.
func (s *Session) Run(ctx context.Context, input any) (string, error) {
	return s.run(ctx, input, nil)
}

func (s *Session) run(ctx context.Context, input any, emit chunkSink) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}

	if len(s.PendingApprovals()) > 0 {
		return "", ErrApprovalNeeded
	}

	if input == nil && len(s.logs) == 0 {
		return "", errors.New("cannot run agent with no input and empty history")
	}

	validator, err := compileValidator(s.agent.outputSchema)
	if err != nil {
		return "", fmt.Errorf("invalid output schema: %w", err)
	}

	repairsLeft := s.agent.maxRepairs
	if input == nil {
		if text, ok := s.FinalOutput(); ok {
			if validator == nil {
				return text, nil
			}
			valErr := validateOutput(validator, text)
			if valErr == nil {
				return text, nil
			}
			if repairsLeft == 0 {
				return "", valErr
			}
			// A stored answer that fails validation is repaired like a new one.
			repairsLeft--
			if err := s.requestRepair(ctx, valErr); err != nil {
				return "", err
			}
		}
	}

	if input != nil && s.hasUnexecutedToolCalls() {
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

		if err := s.appendLogs(ctx, entry); err != nil {
			return "", err
		}
	}

	// ---> Notify start
	// Repair requests do not count against maxTurns.
	for turn := 0; turn < s.agent.maxTurns+s.agent.maxRepairs-repairsLeft; turn++ {
		if err := ctx.Err(); err != nil {
			return "", err
		}

		if toolResults := s.executeUnexecutedToolCalls(ctx); len(toolResults) > 0 {
			// The tools have already run, so their results are kept even if ctx
			// was cancelled meanwhile; otherwise the next Resume would run them again.
			if err := s.appendLogs(context.WithoutCancel(ctx), toolResults...); err != nil {
				return "", err
			}
		}
		if err := ctx.Err(); err != nil {
			return "", err
		}
		if len(s.PendingApprovals()) > 0 {
			return "", ErrApprovalNeeded // a subagent stopped for approval
		}

		start := time.Now()
		var sink chunkSink
		if emit != nil {
			sink = func(chunk Chunk) error {
				chunk.Turn = turn + 1
				return emit(chunk)
			}
		}
		produced, err := s.step(ctx, s.logs, sink)
		if err != nil {
			return "", err
		}
		if len(produced) > 0 {
			produced[len(produced)-1].Duration = time.Since(start)
		}

		// Retain all model entries in history before dispatching local tools.
		if len(produced) > 0 {
			if err := s.appendLogs(ctx, produced...); err != nil {
				return "", err
			}
		}

		if len(s.PendingApprovals()) > 0 {
			return "", ErrApprovalNeeded
		}

		if refusal, refused := latestRefusal(produced); refused {
			if refusal != "" {
				return "", fmt.Errorf("%w: %s", ErrRefused, refusal)
			}
			return "", ErrRefused
		}

		// If the latest turn produced the final answer without requesting further tools:
		if text, ok := s.FinalOutput(); ok {
			if validator != nil {
				if valErr := validateOutput(validator, text); valErr != nil {
					if repairsLeft > 0 {
						repairsLeft--
						if err := s.requestRepair(ctx, valErr); err != nil {
							return "", err
						}
						continue
					}
					return "", valErr
				}
			}
			// ---> Notify end
			return text, nil
		}
	}

	return "", fmt.Errorf("%w (%d)", ErrMaxTurns, s.agent.maxTurns)
}

// requestRepair asks the model to correct an answer that failed validation.
func (s *Session) requestRepair(ctx context.Context, valErr error) error {
	entry, err := NewUserEntry(fmt.Sprintf("Return corrected JSON. Output validation failed: %v", valErr))
	if err != nil {
		return err
	}
	return s.appendLogs(ctx, entry)
}

func (s *Session) step(ctx context.Context, log []Entry, emit chunkSink) ([]Entry, error) {
	return s.agent.step(ctx, log, s.httpClient, emit)
}

func (a *Agent) effectiveHTTPClient(sessionClient *http.Client) *http.Client {
	if sessionClient != nil {
		return sessionClient
	}
	return a.httpClient
}

func (a *Agent) step(ctx context.Context, log []Entry, httpClient *http.Client, emit chunkSink) ([]Entry, error) {
	entries, err := a.providerStep(ctx, log, httpClient, emit)
	if err != nil {
		return nil, redactURLSecrets(err, a.baseURL)
	}
	return entries, nil
}

func (a *Agent) providerStep(ctx context.Context, log []Entry, httpClient *http.Client, emit chunkSink) ([]Entry, error) {
	client := a.effectiveHTTPClient(httpClient)
	switch a.provider {
	case ProviderAnthropic:
		return a.anthropicStep(ctx, log, client, emit)
	case ProviderOpenAI:
		return a.openAIstep(ctx, log, client, emit)
	case ProviderOpenrouter:
		return a.openrouterStep(ctx, log, client, emit)
	case ProviderGoogle:
		return a.geminiStep(ctx, log, client, emit)
	case ProviderXAI:
		return a.xaiStep(ctx, log, client, emit)
	case ProviderDeepSeek:
		return a.deepseekStep(ctx, log, client, emit)
	case ProviderOllama:
		return a.ollamaStep(ctx, log, client, emit)
	default:
		return nil, fmt.Errorf("unsupported provider %q", a.provider)
	}
}

// FinalOutput returns the final assistant text if the latest turn completed
// without requesting further tools, along with a boolean indicating completion.
func (s *Session) FinalOutput() (string, bool) {
	if len(s.logs) == 0 || len(s.PendingApprovals()) > 0 || s.hasUnexecutedToolCalls() {
		return "", false
	}

	start := len(s.logs)
	for i, v := range slices.Backward(s.logs) {
		kind := v.Kind
		if kind == KindUser || kind == KindToolResult || v.HiddenFromModel() {
			start = i + 1
			break
		}
		if i == 0 {
			start = 0
		}
	}

	if start >= len(s.logs) {
		return "", false
	}

	latestTurn := s.logs[start:]
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

// hasAnswerOrCall reports whether a model turn contains assistant content or a
// tool call. A turn with only reasoning or provider-tool events still needs an
// (empty) assistant entry, or it would never count as finished.
func hasAnswerOrCall(entries []Entry) bool {
	return slices.ContainsFunc(entries, func(e Entry) bool {
		return e.Kind == KindAssistant || e.Kind == KindToolCall
	})
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

func (s *Session) Resume(ctx context.Context) (string, error) {
	return s.Run(ctx, nil)
}

func (a *Agent) toolRequiresApproval(name string) bool {
	index := slices.IndexFunc(a.tools, func(tool Tool) bool { return tool.name == name })
	if index < 0 {
		return false
	}
	return a.tools[index].approvalNeeded
}

// openToolCall is a tool call that has no result yet, with the user's
// decision on it if one was recorded.
type openToolCall struct {
	call     *ToolCall
	seq      uint64 // of the call's entry
	decision *Approval
}

// key identifies the call within the session even when a provider reuses call IDs.
func (c openToolCall) key() string {
	return fmt.Sprintf("%d/%s", c.seq, c.call.ID)
}

// openToolCalls returns the tool calls without a result, in log order. A result
// or decision applies to the open call with its ID, not to every call that ever
// had it, so a provider that reuses call IDs across turns still gets each call
// run and answered.
func (s *Session) openToolCalls() []openToolCall {
	var open []openToolCall
	pending := make(map[string]int) // call ID -> index in open
	for _, entry := range s.logs {
		switch {
		case entry.Kind == KindToolCall && entry.ToolCall != nil && entry.ToolCall.ID != "":
			if _, dup := pending[entry.ToolCall.ID]; dup {
				continue // repeated while still open; answered once
			}
			pending[entry.ToolCall.ID] = len(open)
			open = append(open, openToolCall{call: entry.ToolCall, seq: entry.Seq})
		case entry.Kind == KindApproval && entry.Approval != nil:
			if i, ok := pending[entry.Approval.CallID]; ok {
				open[i].decision = entry.Approval
			}
		case entry.Kind == KindToolResult && entry.ToolResult != nil:
			if i, ok := pending[entry.ToolResult.CallID]; ok {
				open[i].call = nil
				delete(pending, entry.ToolResult.CallID)
			}
		}
	}
	return slices.DeleteFunc(open, func(c openToolCall) bool { return c.call == nil })
}

func (s *Session) hasUnexecutedToolCalls() bool {
	return len(s.openToolCalls()) > 0
}

func (s *Session) executeUnexecutedToolCalls(ctx context.Context) []Entry {
	if len(s.PendingApprovals()) > 0 {
		return nil
	}

	unexecuted := s.openToolCalls()
	if len(unexecuted) == 0 {
		return nil
	}

	// Calls from one model turn run concurrently, at most
	// maxConcurrentToolCalls at a time, and see the same state; their results
	// are recorded in the order the model requested them.
	const maxConcurrentToolCalls = 8
	state := s.StateSnapshot()
	results := make([][]Entry, len(unexecuted))
	infos := make([]*toolCallInfo, len(unexecuted))
	slots := make(chan struct{}, maxConcurrentToolCalls)
	var wg sync.WaitGroup
calls:
	for i, open := range unexecuted {
		if ctx.Err() != nil {
			break
		}

		call, dec := open.call, open.decision
		if dec != nil && !dec.Approved {
			reason := dec.Reason
			if reason == "" {
				reason = "tool execution declined by user"
			}
			result := ToolResult{
				CallID: call.ID,
				Error:  reason,
			}
			results[i] = []Entry{{Kind: KindToolResult, ToolResult: &result, At: time.Now().UTC()}}
			continue
		}

		select {
		case slots <- struct{}{}:
		case <-ctx.Done():
			break calls
		}
		infos[i] = &toolCallInfo{key: open.key()}
		wg.Go(func() {
			defer func() { <-slots }()
			results[i], _ = s.dispatch(context.WithValue(ctx, toolCallContextKey{}, infos[i]), call, state)
		})
	}
	wg.Wait()

	for _, info := range infos {
		switch {
		case info == nil:
		case info.waiting != nil:
			if s.children == nil {
				s.children = make(map[string]*Session)
			}
			s.children[info.key] = info.waiting
		default:
			delete(s.children, info.key)
		}
	}

	var entries []Entry
	for _, result := range results {
		entries = append(entries, result...)
	}
	return entries
}

// PendingApprovals returns copies of the tool calls waiting for Approve or
// Reject, including calls of subagents, in the order they were made. Agent
// names the agent that made each call. Changing them does not change what runs.
func (s *Session) PendingApprovals() []*ToolCall {
	var pending []*ToolCall
	for _, p := range s.pendingApprovals() {
		call := *p.call
		call.Args = slices.Clone(call.Args)
		call.Agent = p.owner.agent.name
		pending = append(pending, &call)
	}
	return pending
}

// pendingApproval is a call waiting for a decision, and the session, this one
// or a subagent's, that made it.
type pendingApproval struct {
	owner *Session
	call  *ToolCall
}

func (s *Session) pendingApprovals() []pendingApproval {
	var pending []pendingApproval
	for _, open := range s.openToolCalls() {
		if child := s.children[open.key()]; child != nil {
			pending = append(pending, child.pendingApprovals()...)
			continue
		}
		if open.decision == nil && s.agent.toolRequiresApproval(open.call.Name) {
			pending = append(pending, pendingApproval{owner: s, call: open.call})
		}
	}
	return pending
}

// Approve lets the pending tool call run on the next Resume.
func (s *Session) Approve(ctx context.Context, callID string) error {
	return s.decide(ctx, Approval{CallID: callID, Approved: true})
}

// Reject declines the pending tool call; the model is told reason, or that
// the user declined it.
func (s *Session) Reject(ctx context.Context, callID string, reason string) error {
	if reason == "" {
		reason = "tool execution declined by user"
	}
	return s.decide(ctx, Approval{CallID: callID, Approved: false, Reason: reason})
}

// decide records a decision in the session that made the call, which is a
// subagent's session for a call made by a subagent.
func (s *Session) decide(ctx context.Context, decision Approval) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if decision.CallID == "" {
		return errors.New("tool call ID cannot be empty")
	}

	var owners []*Session
	for _, p := range s.pendingApprovals() {
		if p.call.ID == decision.CallID {
			owners = append(owners, p.owner)
		}
	}
	switch len(owners) {
	case 0:
		return fmt.Errorf("tool call %q is not pending approval", decision.CallID)
	case 1:
		return owners[0].appendLogs(ctx, Entry{Kind: KindApproval, Approval: &decision})
	default:
		return fmt.Errorf("tool call ID %q is pending approval in %d sessions", decision.CallID, len(owners))
	}
}

// RunInto executes the agent and decodes its final response into target.
// target must be a non-nil pointer. Anything that is not text is decoded as JSON.
func (s *Session) RunInto(ctx context.Context, input any, target any) error {
	if target == nil {
		return errors.New("decode target cannot be nil")
	}

	rv := reflect.ValueOf(target)
	if rv.Kind() != reflect.Pointer || rv.IsNil() {
		return fmt.Errorf("decode target must be a non-nil pointer, got %T", target)
	}

	text, err := s.Run(ctx, input)
	if err != nil {
		return err
	}

	temp := reflect.New(rv.Elem().Type())
	if err := decodeInto(text, temp.Interface()); err != nil {
		return err
	}
	rv.Elem().Set(temp.Elem())
	return nil
}

// dispatch runs a tool call locally. A failure is reported to the model rather
// than returned, because the call is still owed an answer.
//
// dispatch returns the tool result entry and any state delta that the tool introduced.
func (s *Session) dispatch(ctx context.Context, call *ToolCall, snapshot map[string]any) ([]Entry, *StateDelta) {
	if call == nil {
		return nil, nil
	}

	result := ToolResult{CallID: call.ID}
	index := slices.IndexFunc(s.agent.tools, func(tool Tool) bool { return tool.name == call.Name })
	if index < 0 {
		result.Error = fmt.Sprintf("tool %q is not allowed", call.Name)
		return []Entry{{Kind: KindToolResult, ToolResult: &result}}, nil
	}

	tool := s.agent.tools[index]

	start := time.Now()
	toolCtx := context.WithValue(ContextWithState(ctx, snapshot), sessionContextKey{}, s)
	output, delta, err := invokeTool(toolCtx, tool, call.Args)
	duration := time.Since(start)
	if errors.Is(err, errToolWaiting) {
		return nil, nil // no result yet; the call runs again once it is decided
	}
	if err != nil {
		result.Error = err.Error()
	} else {
		result.Output = output
	}

	var resp = []Entry{
		{Kind: KindToolResult, ToolResult: &result, At: start.UTC(), Duration: duration},
	}

	if delta != nil {
		// A delta the store cannot encode would fail every write, so the tool
		// would run again on each Resume. Report it to the model instead.
		if _, err := json.Marshal(delta); err != nil {
			result.Output = ""
			result.Error = fmt.Sprintf("tool %q returned state that cannot be stored as JSON: %v", call.Name, err)
			return []Entry{{Kind: KindToolResult, ToolResult: &result, At: start.UTC(), Duration: duration}}, nil
		}
		// The tool may keep and change its maps, so the log holds its own copy.
		delta = &StateDelta{By: delta.By, Set: cloneState(delta.Set), Delete: slices.Clone(delta.Delete)}
		if delta.By == "" {
			delta.By = call.Name
		}

		resp = append(resp, Entry{Kind: KindStateDelta, Delta: delta, At: time.Now().UTC()})
	}

	return resp, delta
}

// invokeTool runs a tool and turns a panic into an error, so one faulty tool
// is reported to the model instead of crashing the program.
func invokeTool(ctx context.Context, tool Tool, args json.RawMessage) (output string, delta *StateDelta, err error) {
	defer func() {
		if r := recover(); r != nil {
			output, delta = "", nil
			err = fmt.Errorf("tool %q panicked: %v", tool.name, r)
		}
	}()
	return tool.invoke(ctx, args)
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
