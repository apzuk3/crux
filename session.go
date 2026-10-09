package crux

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"sync"
	"time"

	"crux.foo/internal/schema"
	"github.com/google/uuid"
	sjs "github.com/santhosh-tekuri/jsonschema/v5"
)

type Session struct {
	id       uuid.UUID
	parentID uuid.UUID // session whose tool call created this one; uuid.Nil at the top level
	agent    *Agent
	logs     []Entry // cache of the entries the store has accepted
	store    Store
	client   *http.Client        // nil uses the default client
	children map[string]*Session // subagent sessions waiting for approval, by openToolCall.key
	onEntry  []func(context.Context, *Session, Entry)
	entryMu  *sync.Mutex // serialises onEntry across the session and its subagents
}

// sessionContextKey carries the session running a tool, so sessions created
// inside that tool (such as subagents) are recorded as its children.
type sessionContextKey struct{}

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
	session.inheritParent(ctx)
	if len(session.onEntry) > 0 && session.entryMu == nil {
		session.entryMu = new(sync.Mutex)
	}
	if session.store == nil {
		session.store = NewMemoryStore()
	}
	if err := session.loadHistory(ctx); err != nil {
		return nil, err
	}
	return session, nil
}

// inheritParent records the session whose tool call creates this one as its
// parent and takes over its store, client and entry handlers.
func (s *Session) inheritParent(ctx context.Context) {
	parent, ok := ctx.Value(sessionContextKey{}).(*Session)
	if !ok || parent.id == s.id {
		return
	}
	s.parentID = parent.id
	if s.store == nil {
		s.store = parent.store
	}
	if s.client == nil {
		s.client = parent.client
	}
	if len(parent.onEntry) > 0 {
		// The parent's handlers see every descendant, before the child's own.
		s.onEntry = append(slices.Clone(parent.onEntry), s.onEntry...)
		s.entryMu = parent.entryMu
	}
}

// loadHistory continues from the history the store holds, or writes the
// seeded one to it.
func (s *Session) loadHistory(ctx context.Context) error {
	stored, err := s.store.Get(ctx, s.id)
	if err != nil && !errors.Is(err, ErrSessionNotFound) {
		return fmt.Errorf("load session %s: %w", s.id, err)
	}
	switch {
	case len(stored) > 0 && len(s.logs) > 0:
		return fmt.Errorf("session %s already has history; WithSessionLogs cannot replace it", s.id)
	case len(stored) > 0:
		s.logs = stored
		return s.loadWaitingChildren(ctx)
	case len(s.logs) > 0:
		normaliseSeeded(s.logs)
		if err := s.store.Append(ctx, s, s.logs...); err != nil {
			return fmt.Errorf("persist session logs: %w", err)
		}
	}
	return nil
}

// normaliseSeeded makes the sequence numbers of seeded entries increasing
// and gives the entries without a time the current one.
func normaliseSeeded(logs []Entry) {
	now := time.Now().UTC()
	var prev uint64
	for i := range logs {
		if logs[i].Seq <= prev {
			logs[i].Seq = prev + 1
		}
		prev = logs[i].Seq
		if logs[i].At.IsZero() {
			logs[i].At = now
		}
	}
}

// loadWaitingChildren finds the subagent sessions that stopped for approval
// inside the session's open tool calls, so PendingApprovals lists their calls
// once the session is loaded again.
func (s *Session) loadWaitingChildren(ctx context.Context) error {
	for _, open := range s.openToolCalls() {
		index := slices.IndexFunc(s.agent.tools, func(tool Tool) bool { return tool.name == open.call.Name })
		if index < 0 || s.agent.tools[index].kind != toolKindSubagent {
			continue
		}
		agent, _, err := s.subAgentFor(s.agent.tools[index], open.call)
		if err != nil {
			continue // the call fails when it runs
		}
		child, err := NewSession(context.WithValue(ctx, sessionContextKey{}, s), agent,
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

	if len(s.onEntry) > 0 {
		s.entryMu.Lock()
		defer s.entryMu.Unlock()
		for _, entry := range entries {
			for _, fn := range s.onEntry {
				fn(ctx, s, cloneEntries([]Entry{entry})[0])
			}
		}
	}
	return nil
}

// Run sends inputs as the next user message, continues the conversation, and
// returns the final text response. Inputs are strings, Attachments (File,
// Data, URL, ...) and other values sent as JSON, in order; with no inputs, Run
// continues the conversation as it is. User inputs, model entries, and tool
// results are retained even if a later step fails, so retry a failed Run with
// Resume: calling Run again with the same inputs would add them to the
// conversation twice. Run must not execute concurrently with other operations
// on the session.
func (s *Session) Run(ctx context.Context, inputs ...any) (string, error) {
	return s.run(ctx, inputs, nil)
}

func (s *Session) run(ctx context.Context, inputs []any, emit chunkSink) (text string, err error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}

	if len(s.PendingApprovals()) > 0 {
		return "", ErrApprovalNeeded
	}
	r, err := s.startRun(inputs)
	if err != nil {
		return "", err
	}
	if r.answered {
		return r.answer, nil
	}

	// From here the run does work, so it is recorded between a started and a
	// finished entry. The finished entry is kept even if ctx was cancelled.
	if err := s.appendLogs(ctx, Entry{Kind: KindRunStarted}); err != nil {
		return "", err
	}
	defer func() { text, err = s.finishRun(ctx, text, err) }()

	if err := s.openRun(ctx, r); err != nil {
		return "", err
	}
	// Repair requests do not count against maxTurns.
	first := r.userEntry != nil
	for turn := 0; turn < s.agent.maxTurns+s.agent.maxRepairs-r.repairsLeft; turn++ {
		if err := s.runTools(ctx); err != nil {
			return "", err
		}
		produced, err := s.requestTurn(ctx, turn, first, emit)
		if err != nil {
			return "", err
		}
		first = false
		answer, done, err := s.checkTurnOutcome(ctx, produced, r)
		if done {
			return answer, err
		}
	}
	return "", fmt.Errorf("%w (%d)", ErrMaxTurns, s.agent.maxTurns)
}

// runState is what a run keeps from its start through its turns.
type runState struct {
	validator   *sjs.Schema
	repairsLeft int
	userEntry   *Entry // the new input, recorded once the run has started
	repair      error  // a stored answer that failed validation, repaired like a new one
	answer      string // a stored answer that needs no work
	answered    bool
}

// startRun checks the input and compiles the output schema. Without input,
// a stored final answer is returned as it is when it is valid.
func (s *Session) startRun(inputs []any) (*runState, error) {
	hasInput := slices.ContainsFunc(inputs, func(v any) bool { return v != nil })
	if !hasInput && len(s.logs) == 0 {
		return nil, errors.New("cannot run agent with no input and empty history")
	}
	validator, err := schema.CompileOutput(s.agent.outputSchema)
	if err != nil {
		return nil, fmt.Errorf("invalid output schema: %w", err)
	}
	r := &runState{validator: validator, repairsLeft: s.agent.maxRepairs}
	if !hasInput {
		return r, s.storedAnswer(r)
	}
	userEntry, err := s.newUserEntry(inputs)
	if err != nil {
		return nil, err
	}
	r.userEntry = userEntry
	return r, nil
}

// storedAnswer takes the final answer the log already holds, if any: a
// valid one answers the run, one that fails validation is repaired when
// repairs are left, and is the run's error otherwise.
func (s *Session) storedAnswer(r *runState) error {
	text, ok := s.FinalOutput()
	if !ok {
		return nil
	}
	if r.validator == nil {
		r.answer, r.answered = text, true
		return nil
	}
	valErr := validateOutput(r.validator, text)
	switch {
	case valErr == nil:
		r.answer, r.answered = text, true
	case r.repairsLeft == 0:
		return valErr
	default:
		r.repair = valErr
	}
	return nil
}

// newUserEntry builds the user turn for inputs. It is part of the log, so
// every provider sees one shape and a resumed session needs nothing but its
// history.
func (s *Session) newUserEntry(inputs []any) (*Entry, error) {
	if s.hasUnexecutedToolCalls() {
		return nil, errors.New("cannot run agent with new user input while tool calls are pending execution; call Resume first")
	}
	entry, err := NewUserEntry(inputs...)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(entry.Text()) == "" && !slices.ContainsFunc(entry.Content, func(p ContentPart) bool { return p.Kind == ContentKindFile }) {
		return nil, errors.New("user input produced empty text")
	}
	return &entry, nil
}

// openRun records what the run starts from: the repair request for a stored
// answer, or the new user entry.
func (s *Session) openRun(ctx context.Context, r *runState) error {
	if r.repair != nil {
		r.repairsLeft--
		if err := s.requestRepair(ctx, r.repair); err != nil {
			return err
		}
	}
	if r.userEntry != nil {
		return s.appendLogs(ctx, *r.userEntry)
	}
	return nil
}

// finishRun records how the run ended, even if ctx was cancelled, and
// returns the run's result, or the write's failure joined to its error.
func (s *Session) finishRun(ctx context.Context, text string, err error) (string, error) {
	status := runStatus(err)
	if finishErr := s.appendLogs(context.WithoutCancel(ctx), Entry{Kind: KindRunFinished, Run: &status}); finishErr != nil {
		return "", errors.Join(err, finishErr)
	}
	return text, err
}

// runTools runs the open tool calls and records their results, and stops
// the run when a subagent waits for approval.
func (s *Session) runTools(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	toolResults, err := s.executeUnexecutedToolCalls(ctx)
	if err != nil {
		return err
	}
	if len(toolResults) > 0 {
		// The tools have already run, so their results are kept even if ctx
		// was cancelled meanwhile; otherwise the next Resume would run them again.
		if err := s.appendLogs(context.WithoutCancel(ctx), toolResults...); err != nil {
			return err
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(s.PendingApprovals()) > 0 {
		return ErrApprovalNeeded // a subagent stopped for approval
	}
	return nil
}

// requestTurn compacts the session when it has grown, sends the request and
// records what the model produced, with its timing on the last entry.
func (s *Session) requestTurn(ctx context.Context, turn int, first bool, emit chunkSink) ([]Entry, error) {
	// The request usually still fits when compaction fails, so it is sent
	// anyway; the failure is reported only if the request fails too.
	compactErr := s.maybeCompact(ctx)
	sink := &turnSink{emit: emit, turn: turn}
	produced, start, err := s.stepWithRetry(ctx, sink.sink(), first, compactErr)
	if err != nil {
		return nil, err
	}
	sink.stampTiming(produced, start)
	// Retain all model entries in history before dispatching local tools.
	if len(produced) > 0 {
		if err := s.appendLogs(ctx, produced...); err != nil {
			return nil, err
		}
	}
	return produced, nil
}

// stepWithRetry records the turn's start and sends the request, compacting
// and sending again when the model finds it too long. start is when the
// request that succeeded was sent.
func (s *Session) stepWithRetry(ctx context.Context, sink chunkSink, first bool, compactErr error) (produced []Entry, start time.Time, err error) {
	for attempt := 0; ; attempt++ {
		if err := s.appendLogs(ctx, Entry{Kind: KindTurnStarted, Turn: s.agent.turnInfo()}); err != nil {
			return nil, start, err
		}
		start = time.Now()
		produced, err = s.agent.step(ctx, s.client, s.logs, sink, first)
		if err == nil {
			return produced, start, nil
		}
		retry, stepErr := s.recoverContext(ctx, err, attempt)
		if retry {
			continue
		}
		if compactErr != nil {
			stepErr = errors.Join(stepErr, compactErr)
		}
		return nil, start, stepErr
	}
}

// turnSink forwards the streamed chunks of one turn and notes when the
// first arrived.
type turnSink struct {
	emit       chunkSink
	turn       int
	firstToken time.Time
}

// sink returns the sink to stream through, or nil when nothing listens.
func (t *turnSink) sink() chunkSink {
	if t.emit == nil {
		return nil
	}
	return t.send
}

func (t *turnSink) send(chunk Chunk) error {
	if t.firstToken.IsZero() {
		t.firstToken = time.Now()
	}
	chunk.Turn = t.turn + 1
	return t.emit(chunk)
}

// stampTiming records the response's duration and, when it was streamed,
// the time to its first token on the last produced entry.
func (t *turnSink) stampTiming(produced []Entry, start time.Time) {
	if len(produced) == 0 {
		return
	}
	last := &produced[len(produced)-1]
	last.Duration = time.Since(start)
	if t.firstToken.IsZero() {
		return
	}
	if last.Response == nil {
		last.Response = &ResponseInfo{}
	}
	last.Response.FirstTokenAfter = t.firstToken.Sub(start)
}

// checkTurnOutcome reports whether the run ends with what the turn
// produced: with an approval or a refusal, with the final answer, or with
// the answer's validation error once no repairs are left. An answer that
// fails validation with repairs left asks the model for a correction.
func (s *Session) checkTurnOutcome(ctx context.Context, produced []Entry, r *runState) (text string, done bool, err error) {
	if len(s.PendingApprovals()) > 0 {
		return "", true, ErrApprovalNeeded
	}
	if refusal, refused := latestRefusal(produced); refused {
		return "", true, refusalError(refusal)
	}
	text, ok := s.FinalOutput()
	if !ok {
		return "", false, nil
	}
	if r.validator == nil {
		return text, true, nil
	}
	valErr := validateOutput(r.validator, text)
	if valErr == nil {
		return text, true, nil
	}
	if r.repairsLeft == 0 {
		return "", true, valErr
	}
	r.repairsLeft--
	if err := s.requestRepair(ctx, valErr); err != nil {
		return "", true, err
	}
	return "", false, nil
}

func refusalError(refusal string) error {
	if refusal != "" {
		return fmt.Errorf("%w: %s", ErrRefused, refusal)
	}
	return ErrRefused
}

// runStatus describes how a run that returned err ended.
func runStatus(err error) RunStatus {
	var status RunStatus
	switch {
	case err == nil:
		return RunStatus{Outcome: RunAnswered}
	case errors.Is(err, ErrApprovalNeeded):
		status.Outcome = RunApprovalNeeded
	case errors.Is(err, ErrRefused):
		status.Outcome = RunRefused
	case errors.Is(err, ErrMaxTurns):
		status.Outcome = RunMaxTurns
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		status.Outcome = RunCancelled
	default:
		status.Outcome = RunFailed
	}
	status.Error = err.Error()
	return status
}

// requestRepair asks the model to correct an answer that failed validation.
func (s *Session) requestRepair(ctx context.Context, valErr error) error {
	entry, err := NewUserEntry(fmt.Sprintf("Return corrected JSON. Output validation failed: %v", valErr))
	if err != nil {
		return err
	}
	return s.appendLogs(ctx, entry)
}

// FinalOutput returns the final assistant text if the latest turn completed
// without requesting further tools, along with a boolean indicating completion.
func (s *Session) FinalOutput() (string, bool) {
	if len(s.logs) == 0 || len(s.PendingApprovals()) > 0 || s.hasUnexecutedToolCalls() {
		return "", false
	}
	latestTurn := s.logs[latestTurnStart(s.logs):]
	if !turnHasFinalAnswer(latestTurn) {
		return "", false
	}
	return finalText(latestTurn), true
}

// latestTurnStart returns the index of the first entry of the latest model
// turn, which follows the latest entry that ends the turn before it.
func latestTurnStart(log []Entry) int {
	for i, e := range slices.Backward(log) {
		if endsPreviousTurn(e) {
			return i + 1
		}
	}
	return 0
}

// endsPreviousTurn reports whether e closes the model turn before it: user
// input, a tool result, or a hidden entry other than the lifecycle records
// and compactions, which belong to no turn.
func endsPreviousTurn(e Entry) bool {
	kind := e.Kind
	return kind == KindUser || kind == KindToolResult || (e.HiddenFromModel() && !kind.lifecycle() && kind != KindCompaction)
}

// turnHasFinalAnswer reports whether the turn answered without refusing or
// requesting tools.
func turnHasFinalAnswer(turn []Entry) bool {
	if _, refused := latestRefusal(turn); refused {
		return false
	}
	var hasAssistant bool
	for _, e := range turn {
		if e.Kind == KindToolCall {
			return false
		}
		if e.Kind == KindAssistant {
			hasAssistant = true
		}
	}
	return hasAssistant
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
	return s.Run(ctx)
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

// maxConcurrentToolCalls is how many calls from one model turn run at once.
const maxConcurrentToolCalls = 8

func (s *Session) executeUnexecutedToolCalls(ctx context.Context) ([]Entry, error) {
	if len(s.PendingApprovals()) > 0 {
		return nil, nil
	}

	unexecuted := s.openToolCalls()
	if len(unexecuted) == 0 {
		return nil, nil
	}
	batch := &toolBatch{
		session: s,
		calls:   unexecuted,
		results: make([][]Entry, len(unexecuted)),
		ran:     make([]bool, len(unexecuted)),
		waiting: make([]*Session, len(unexecuted)),
		slots:   make(chan struct{}, maxConcurrentToolCalls),
	}
	run, err := batch.start(ctx)
	if err != nil {
		return nil, err
	}
	batch.state = s.StateSnapshot()
	sequential, spawns, parallel, spawnLimit := batch.partition(run)
	batch.runSpawns(ctx, spawns, spawnLimit)
	batch.runSequential(ctx, sequential)
	batch.runParallel(ctx, parallel)
	batch.wg.Wait()
	batch.recordWaiting()
	return slices.Concat(batch.results...), nil
}

// toolBatch runs the open tool calls of one model turn. Calls run
// concurrently, at most maxConcurrentToolCalls at a time, and see the same
// state; their results are recorded in the order the model requested them.
// Calls of sequential tools run one after another in that order, as one of
// the concurrent tasks, each seeing the state the calls before it left.
// Spawn calls have their own queue (WithSpawnConcurrency).
type toolBatch struct {
	session *Session
	calls   []openToolCall
	state   map[string]any
	results [][]Entry  // by call
	ran     []bool     // by call
	waiting []*Session // by call; a subagent that stopped for approval
	slots   chan struct{}
	wg      sync.WaitGroup
}

// start answers the rejected calls without running them and records the
// others as started, in one write, before any of them runs. It returns the
// indexes of the calls to run.
func (b *toolBatch) start(ctx context.Context) ([]int, error) {
	var run []int
	var started []Entry
	for i, open := range b.calls {
		if ctx.Err() != nil {
			break
		}
		if dec := open.decision; dec != nil && !dec.Approved {
			b.results[i] = []Entry{deniedResult(open.call.ID, dec.Reason)}
			continue
		}
		run = append(run, i)
		started = append(started, Entry{Kind: KindToolStarted, ToolCall: &ToolCall{ID: open.call.ID, Name: open.call.Name}})
	}
	if len(started) == 0 || ctx.Err() != nil {
		return nil, nil
	}
	if err := b.session.appendLogs(ctx, started...); err != nil {
		return nil, err
	}
	return run, nil
}

func deniedResult(callID, reason string) Entry {
	reason = cmp.Or(reason, "tool execution declined by user")
	result := ToolResult{CallID: callID, Error: reason, Denied: true}
	return Entry{Kind: KindToolResult, ToolResult: &result, At: time.Now().UTC()}
}

// partition splits the calls to run into those of sequential tools, the
// spawn calls with their concurrency limit, and the rest.
func (b *toolBatch) partition(run []int) (sequential, spawns, parallel []int, spawnLimit int) {
	agent := b.session.agent
	for _, i := range run {
		name := b.calls[i].call.Name
		if agent.toolIsSequential(name) {
			sequential = append(sequential, i)
		} else if limit, ok := agent.spawnConcurrency(name); ok {
			spawns = append(spawns, i)
			spawnLimit = limit
		} else {
			parallel = append(parallel, i)
		}
	}
	return sequential, spawns, parallel, spawnLimit
}

// run dispatches one call with the state it sees.
func (b *toolBatch) run(ctx context.Context, i int, state map[string]any) {
	b.results[i], b.waiting[i] = b.session.dispatch(ctx, b.calls[i], state)
}

func (b *toolBatch) runSpawns(ctx context.Context, spawns []int, limit int) {
	if len(spawns) == 0 {
		return
	}
	spawnSlots := make(chan struct{}, limit)
	b.wg.Go(func() {
		for _, i := range spawns {
			select {
			case spawnSlots <- struct{}{}:
			case <-ctx.Done():
				return
			}
			b.ran[i] = true
			b.wg.Go(func() {
				defer func() { <-spawnSlots }()
				b.run(ctx, i, b.state)
			})
		}
	})
}

func (b *toolBatch) runSequential(ctx context.Context, sequential []int) {
	if len(sequential) == 0 {
		return
	}
	select {
	case b.slots <- struct{}{}:
	case <-ctx.Done():
		return
	}
	b.wg.Go(func() {
		defer func() { <-b.slots }()
		state := cloneState(b.state)
		for _, i := range sequential {
			if ctx.Err() != nil {
				return
			}
			b.ran[i] = true
			b.run(ctx, i, state)
			applyResultDeltas(state, b.results[i])
		}
	})
}

func applyResultDeltas(state map[string]any, entries []Entry) {
	for _, e := range entries {
		if e.Kind == KindStateDelta {
			applyDelta(state, e.Delta)
		}
	}
}

func (b *toolBatch) runParallel(ctx context.Context, parallel []int) {
	for _, i := range parallel {
		select {
		case b.slots <- struct{}{}:
		case <-ctx.Done():
			return
		}
		b.ran[i] = true
		b.wg.Go(func() {
			defer func() { <-b.slots }()
			b.run(ctx, i, b.state)
		})
	}
}

// recordWaiting keeps the subagent sessions that stopped for approval under
// the calls that run them, and forgets those of calls that finished.
func (b *toolBatch) recordWaiting() {
	s := b.session
	for i, open := range b.calls {
		switch {
		case !b.ran[i]:
		case b.waiting[i] != nil:
			if s.children == nil {
				s.children = make(map[string]*Session)
			}
			s.children[open.key()] = b.waiting[i]
		default:
			delete(s.children, open.key())
		}
	}
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

// RunInto runs like Run and decodes the final response into target, which
// must be a non-nil pointer and comes before the inputs. Anything that is not
// text is decoded as JSON.
func (s *Session) RunInto(ctx context.Context, target any, inputs ...any) error {
	if target == nil {
		return errors.New("decode target cannot be nil")
	}

	rv := reflect.ValueOf(target)
	if rv.Kind() != reflect.Pointer || rv.IsNil() {
		return fmt.Errorf("decode target must be a non-nil pointer, got %T (RunInto takes the target before the inputs)", target)
	}

	text, err := s.Run(ctx, inputs...)
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

// dispatch runs a tool call locally and returns its result entry, followed by
// the state delta the tool introduced, if any. A failure is reported to the
// model rather than returned, because the call is still owed an answer.
//
// A subagent whose session stops for approval returns no entries and its
// session as waiting: the call gets no result and runs again once the
// approval is decided.
func (s *Session) dispatch(ctx context.Context, open openToolCall, snapshot map[string]any) (entries []Entry, waiting *Session) {
	call := open.call
	result := ToolResult{CallID: call.ID}
	index := slices.IndexFunc(s.agent.tools, func(tool Tool) bool { return tool.name == call.Name })
	if index < 0 {
		result.Error = fmt.Sprintf("tool %q is not allowed", call.Name)
		return []Entry{{Kind: KindToolResult, ToolResult: &result}}, nil
	}

	tool := s.agent.tools[index]

	start := time.Now()
	toolCtx := context.WithValue(ContextWithState(ctx, snapshot), sessionContextKey{}, s)
	var output string
	var delta *StateDelta
	var err error
	if tool.kind == toolKindSubagent {
		output, delta, waiting, err = s.runSubAgent(toolCtx, tool, open)
		if waiting != nil {
			return nil, waiting
		}
	} else {
		output, delta, err = invokeTool(toolCtx, tool, call.Args)
	}
	duration := time.Since(start)
	if err != nil {
		result.Error = err.Error()
	} else {
		result.Output = output
	}

	if delta != nil {
		// A delta the store cannot encode would fail every write, so the tool
		// would run again on each Resume. Report it to the model instead.
		if _, err := json.Marshal(delta); err != nil {
			result.Output = ""
			result.Error = fmt.Sprintf("tool %q returned state that cannot be stored as JSON: %v", call.Name, err)
			delta = nil
		}
	}

	var resp = []Entry{
		{Kind: KindToolResult, ToolResult: &result, At: start.UTC(), Duration: duration},
	}

	if delta != nil {
		// The tool may keep and change its maps, so the log holds its own copy.
		delta = &StateDelta{By: delta.By, Set: cloneState(delta.Set), Delete: slices.Clone(delta.Delete)}
		if delta.By == "" {
			delta.By = call.Name
		}

		resp = append(resp, Entry{Kind: KindStateDelta, Delta: delta, At: time.Now().UTC()})
	}

	return resp, nil
}

// runSubAgent runs the subagent of a WithSubAgent or spawn tool on the task
// in the call's arguments. The subagent's session ID derives from the call,
// so a call that stopped for approval, or whose result was not stored,
// continues in the same session instead of starting over. When that session
// stops for approval it is returned as waiting.
func (s *Session) runSubAgent(ctx context.Context, tool Tool, open openToolCall) (output string, delta *StateDelta, waiting *Session, err error) {
	agent, task, err := s.subAgentFor(tool, open.call)
	if err != nil {
		return "", nil, nil, err
	}

	child, err := NewSession(ctx, agent, WithSessionID(childSessionID(s.id, open.key())))
	if err != nil {
		return "", nil, nil, err
	}
	if len(child.logs) > 0 {
		output, err = child.Resume(ctx)
	} else {
		output, err = child.Run(ctx, task)
	}
	if errors.Is(err, ErrApprovalNeeded) {
		return "", nil, child, nil
	}
	if err != nil {
		return "", nil, nil, err
	}
	if tool.spawn != nil {
		return output, nil, nil, nil // the model chose the name, so it is no state key
	}
	return output, &StateDelta{Set: map[string]any{agent.name: output}}, nil, nil
}

// subAgentFor returns the agent a subagent tool call runs and its task.
func (s *Session) subAgentFor(tool Tool, call *ToolCall) (*Agent, string, error) {
	if tool.spawn != nil {
		return s.agent.spawnedAgent(tool, call.Args)
	}
	input, err := schema.DecodeArgs[subAgentInput](tool.name, call.Args, subAgentArgsValidator())
	if err != nil {
		return nil, "", err
	}
	if strings.TrimSpace(input.Task) == "" {
		return nil, "", fmt.Errorf("subagent %q needs a non-empty task", tool.subAgent.name)
	}
	return tool.subAgent, input.Task, nil
}

// invokeTool runs a tool and turns a panic into an error, so one faulty tool
// is reported to the model instead of crashing the program.
// invokeTool runs the tool, stopping waiting for it after its timeout.
func invokeTool(ctx context.Context, tool Tool, args json.RawMessage) (string, *StateDelta, error) {
	if tool.timeout <= 0 {
		return callTool(ctx, tool, args)
	}
	timedOut := fmt.Errorf("tool %q timed out after %v", tool.name, tool.timeout)
	ctx, cancel := context.WithTimeoutCause(ctx, tool.timeout, timedOut)
	defer cancel()
	type result struct {
		output string
		delta  *StateDelta
		err    error
	}
	done := make(chan result, 1)
	go func() {
		output, delta, err := callTool(ctx, tool, args)
		done <- result{output, delta, err}
	}()
	select {
	case r := <-done:
		if r.err != nil && context.Cause(ctx) == timedOut {
			return "", nil, timedOut
		}
		return r.output, r.delta, r.err
	case <-ctx.Done():
		if context.Cause(ctx) == timedOut {
			return "", nil, timedOut
		}
		// The run was cancelled: wait for the tool, as without a timeout.
		r := <-done
		return r.output, r.delta, r.err
	}
}

// callTool runs the tool, turning a panic into an error.
func callTool(ctx context.Context, tool Tool, args json.RawMessage) (output string, delta *StateDelta, err error) {
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

// SessionOption configures a Session in NewSession.
type SessionOption func(*Session) error

// WithSessionID sets the session ID. If the store already holds a session with
// this ID, NewSession loads its history and the conversation continues.
func WithSessionID(id uuid.UUID) SessionOption {
	return func(s *Session) error {
		s.id = id
		return nil
	}
}

// WithSessionLogs seeds a new session with existing history.
func WithSessionLogs(logs []Entry) SessionOption {
	return func(s *Session) error { s.logs = cloneEntries(logs); return nil }
}

// WithHTTPClient sends the session's provider requests through client,
// including compaction and the sessions of its subagents, which inherit it.
// Nil uses the default client. In tests, pass cruxtest's Mock.Client.
func WithHTTPClient(client *http.Client) SessionOption {
	return func(s *Session) error { s.client = client; return nil }
}

// WithStore persists the session in store. The default is a MemoryStore.
func WithStore(store Store) SessionOption {
	return func(s *Session) error {
		if store == nil {
			return errors.New("store cannot be nil")
		}
		s.store = store
		return nil
	}
}

// WithEntryHandler calls fn with every entry the session appends to its log,
// once the store has accepted it and in log order. It can be given more than
// once; handlers run in the order they were added, each with its own copy of
// the entry. Subagent sessions report to the same handlers; s is the session
// that appended the entry. Calls are serialised, so fn need not be safe for
// concurrent use, but it runs on the session's goroutine and must not call
// back into the session. History seeded with WithSessionLogs is not reported.
func WithEntryHandler(fn func(ctx context.Context, s *Session, e Entry)) SessionOption {
	return func(s *Session) error {
		if fn == nil {
			return errors.New("entry handler cannot be nil")
		}
		s.onEntry = append(s.onEntry, fn)
		return nil
	}
}

// Fork creates a session with the full retained history and inherited
// configuration, then applies opts in order to the session's agent.
// Changing providers clears opaque data and removes provider-only tool events
// from the copied history. Portable entries remain in their original order.
//
// History and mutable search/tool-selection settings are copied. State values
// follow StateSnapshot's copying rules. Bound tools are inherited unless tool
// options are supplied, which resolve a fresh selection from the registry.
// The tools registry and output schema remain shared unless overridden.
// Token usage is not copied: the fork's Usage counts only its own requests.
// Fork must not run concurrently with writes to the session.
func (s *Session) Fork(ctx context.Context, opts ...AgentOption) (*Session, error) {
	return s.forkFrom(ctx, len(s.logs), opts...)
}

// forkFrom is Fork with only logs[:from] copied. from is an exclusive slice
// offset, not an Entry.Seq. Empty history is valid; a prefix with an
// unfinished local tool exchange or an unmatched tool result is not.
func (s *Session) forkFrom(ctx context.Context, from int, opts ...AgentOption) (*Session, error) {
	return s.forkWith(ctx, from, nil, opts...)
}

// forkWith is forkFrom with extra options for the new session.
func (s *Session) forkWith(ctx context.Context, from int, sessionOpts []SessionOption, opts ...AgentOption) (*Session, error) {
	if from < 0 || from > len(s.logs) {
		return nil, fmt.Errorf("cannot fork at offset %d: must be between 0 and %d", from, len(s.logs))
	}
	if err := validateForkHistory(s.logs[:from]); err != nil {
		return nil, fmt.Errorf("cannot fork at offset %d: %w", from, err)
	}

	clonedAgent, err := s.agent.clone(opts...)
	if err != nil {
		return nil, err
	}

	forkedLogs := cloneEntries(s.logs[:from])
	for i := range forkedLogs {
		forkedLogs[i].Usage = nil
	}
	if clonedAgent.provider != s.agent.provider {
		for i := range forkedLogs {
			forkedLogs[i].Opaque = nil
		}
		forkedLogs = slices.DeleteFunc(forkedLogs, func(entry Entry) bool {
			return entry.Kind == KindProviderTool
		})
	}

	sessionOpts = append([]SessionOption{WithSessionLogs(forkedLogs), WithStore(s.store), WithHTTPClient(s.client)}, sessionOpts...)
	return NewSession(ctx, clonedAgent, sessionOpts...)
}

func validateForkHistory(entries []Entry) error {
	pending := make(map[string]bool)
	for _, entry := range entries {
		switch entry.Kind {
		case KindToolCall:
			if entry.ToolCall == nil || entry.ToolCall.ID == "" {
				return fmt.Errorf("tool call has no ID")
			}
			// A call repeated while still open is answered once, as in the run loop.
			pending[entry.ToolCall.ID] = true
		case KindToolResult:
			if entry.ToolResult == nil || !pending[entry.ToolResult.CallID] {
				return fmt.Errorf("tool result has no matching pending call")
			}
			delete(pending, entry.ToolResult.CallID)
		}
	}
	// Report in log order so errors are deterministic for parallel batches.
	for _, entry := range entries {
		if entry.Kind == KindToolCall && pending[entry.ToolCall.ID] {
			return fmt.Errorf("tool call %q has no result in retained history", entry.ToolCall.ID)
		}
	}
	return nil
}

// ChunkKind identifies readable content in a provider stream.
type ChunkKind string

const (
	ChunkText      ChunkKind = "text"
	ChunkReasoning ChunkKind = "reasoning"
)

// Chunk is provisional incremental content. Turn is one-based within this run.
// Reasoning contains only readable reasoning or summaries exposed by the provider.
type Chunk struct {
	Kind  ChunkKind
	Delta string
	Turn  int
}

type chunkSink func(Chunk) error

// Stream runs the conversation when iterated, yielding text and reasoning deltas.
// Each iterator is single-use. Breaking iteration cancels the active request and
// stops execution. Completed steps are retained; an interrupted step is not.
// Inputs, tools and approvals behave as in Run; call it with no inputs to resume
// after approval.
// Deltas may include intermediate commentary and invalid output before a repair.
// After successful iteration, FinalOutput returns the authoritative final answer.
// Stream must not execute concurrently with other operations on the session.
func (s *Session) Stream(ctx context.Context, inputs ...any) iter.Seq2[Chunk, error] {
	used := false
	return func(yield func(Chunk, error) bool) {
		if used {
			yield(Chunk{}, errors.New("stream iterator already consumed"))
			return
		}
		used = true
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		stopped := false
		_, err := s.run(ctx, inputs, func(chunk Chunk) error {
			if !yield(chunk, nil) {
				stopped = true
				cancel()
				return ctx.Err()
			}
			return ctx.Err()
		})
		if err != nil && !stopped {
			yield(Chunk{}, err)
		}
	}
}

type stateContextKey struct{}

// StateSnapshot reconstructs the current state by replaying state deltas in log
// order. Set replaces whole values; Delete is applied after Set in each delta.
// A session without state deltas returns an empty, non-nil map.
//
// A session loaded from a persistent store such as GORMStore has its values
// decoded from JSON: numbers become float64, structs become map[string]any.
//
// Snapshots recursively copy JSON-like values (map[string]any and []any), byte
// slices, and json.RawMessage. Other values are copied by assignment; callers
// must treat other reference-bearing types as immutable. Values must be acyclic.
// StateSnapshot must not run concurrently with writes to the session's logs.
func (s *Session) StateSnapshot() map[string]any {
	state := make(map[string]any)
	for _, entry := range s.logs {
		if entry.Kind == KindStateDelta {
			applyDelta(state, entry.Delta)
		}
	}
	return cloneState(state)
}

// applyDelta changes state as delta says. The values are not copied.
func applyDelta(state map[string]any, delta *StateDelta) {
	if delta == nil {
		return
	}
	maps.Copy(state, delta.Set)
	for _, key := range delta.Delete {
		delete(state, key)
	}
}

// ContextWithState attaches a snapshot of state to a child context. Copying
// follows StateSnapshot's value rules. A nil map is a present, nil snapshot.
func ContextWithState(ctx context.Context, state map[string]any) context.Context {
	return context.WithValue(ctx, stateContextKey{}, cloneState(state))
}

// StateFromContext retrieves a copy of the injected snapshot, using
// StateSnapshot's value rules. It returns (nil, false) when no state is present.
func StateFromContext(ctx context.Context) (map[string]any, bool) {
	state, ok := ctx.Value(stateContextKey{}).(map[string]any)
	if !ok {
		return nil, false
	}
	return cloneState(state), true
}
