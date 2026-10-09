package crux

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"
)

// Compaction keeps a long session within the model's context window. The
// log is never changed: a KindCompaction entry records what the model no
// longer sees in full, and modelView applies it to every request.
//
// Before each request, a session whose context has grown past CompactAt of
// the window first omits large tool outputs and files the model has already
// read. If that is not enough, the model summarises the older part of the
// conversation. A provider that rejects a request as too long gets the same
// treatment and the request is sent again.

const (
	defaultCompactAt = 0.8
	compactTarget    = 0.5     // compaction aims for this fraction of the window
	compactTail      = 0.25    // the most recent entries a summary leaves as they are
	omitOver         = 2048    // tool outputs longer than this, in bytes, are omitted
	transcriptCap    = 2000    // bytes of each tool output a summary is written from
	transcriptMax    = 200_000 // bytes of the whole transcript a summary is written from
)

// CompactionOptions configures compaction; see WithCompaction.
type CompactionOptions struct {
	Off   bool    `json:"off,omitempty"`
	At    float64 `json:"at,omitempty"`    // fraction of the window; 0 uses 0.8
	Model string  `json:"model,omitempty"` // writes summaries; "" uses the agent's model
}

type CompactionOption func(*CompactionOptions)

// CompactAt compacts once the context passes fraction of the model's window,
// which must be between 0 and 1. The default is 0.8.
func CompactAt(fraction float64) CompactionOption {
	return func(o *CompactionOptions) { o.At = fraction }
}

// CompactWith has model write the summaries, such as a cheaper model of the
// same provider. A model of another provider uses that provider's API key
// from the environment.
func CompactWith(model string) CompactionOption {
	return func(o *CompactionOptions) { o.Model = model }
}

// WithCompaction configures compaction, which is on by default: a session
// close to the model's context window omits old tool outputs and, when that
// is not enough, summarises older turns. Each call replaces the configuration.
func WithCompaction(opts ...CompactionOption) AgentOption {
	return func(a *Agent) error {
		var c CompactionOptions
		for _, opt := range opts {
			opt(&c)
		}
		if c.At < 0 || c.At >= 1 {
			return fmt.Errorf("compaction threshold must be between 0 and 1, got %v", c.At)
		}
		a.compaction = c
		return nil
	}
}

// WithoutCompaction turns compaction off: a request larger than the model
// accepts fails with ErrContextTooLong. Session.Compact still works.
func WithoutCompaction() AgentOption {
	return func(a *Agent) error {
		a.compaction = CompactionOptions{Off: true}
		return nil
	}
}

// WithContextWindow sets the model's context window in tokens, for models
// crux doesn't know, such as local Ollama models. Without one, compaction
// only starts when the provider rejects a request as too long.
func WithContextWindow(tokens int) AgentOption {
	return func(a *Agent) error {
		if tokens < 0 {
			return fmt.Errorf("context window cannot be negative, got %d", tokens)
		}
		a.contextWindow = tokens
		return nil
	}
}

// window returns the agent's context window in tokens, or 0 when unknown.
func (a *Agent) window() int {
	if a.contextWindow > 0 {
		return a.contextWindow
	}
	if spec, ok := providerSpecs[a.provider]; ok && spec.contextWindow != nil {
		return spec.contextWindow(a.model)
	}
	return 0
}

// Compact summarises the older part of the conversation now, whatever its
// size, as a /compact command would. It does nothing when there is nothing
// to summarise yet.
func (s *Session) Compact(ctx context.Context) error {
	_, err := s.summarise(ctx, true)
	return err
}

// maybeCompact compacts before a request when the context has grown past
// the agent's threshold.
func (s *Session) maybeCompact(ctx context.Context) error {
	window := s.agent.window()
	if s.agent.compaction.Off || window == 0 {
		return nil
	}
	at := s.agent.compaction.At
	if at == 0 {
		at = defaultCompactAt
	}
	if s.contextTokens() <= int(at*float64(window)) {
		return nil
	}
	_, err := s.compact(ctx, false)
	return err
}

// recoverContext compacts after the provider rejected a request as too long,
// and reports whether to send it again. The first attempt may only omit
// outputs; the second summarises.
func (s *Session) recoverContext(ctx context.Context, err error, attempt int) (bool, error) {
	if !errors.Is(err, ErrContextTooLong) || s.agent.compaction.Off || attempt >= 2 {
		return false, err
	}
	compacted, cerr := s.compact(ctx, attempt > 0)
	if cerr != nil {
		return false, errors.Join(err, cerr)
	}
	return compacted, err
}

// compact omits large outputs the model has read when that brings the
// context under the target, and otherwise summarises.
func (s *Session) compact(ctx context.Context, summarise bool) (bool, error) {
	if !summarise {
		if through := lastResponseSeq(s.logs); through > omittedThrough(s.logs) && hasLargeOutputs(s.logs, omittedThrough(s.logs), through) {
			entry := Entry{Kind: KindCompaction, Compaction: &Compaction{Through: through}}
			window := s.agent.window()
			after := estimateTokens(modelView(slices.Concat(s.logs, []Entry{entry}))) + s.agent.fixedTokens()
			if window == 0 || after <= int(compactTarget*float64(window)) {
				return true, s.appendLogs(ctx, entry)
			}
		}
	}
	return s.summarise(ctx, false)
}

// summarise replaces the conversation up to a turn boundary with a summary
// written by the compaction model. An explicit request summarises everything
// before the latest user message.
func (s *Session) summarise(ctx context.Context, explicit bool) (bool, error) {
	from, cut := s.summaryRange(explicit)
	if cut <= from {
		return false, nil
	}
	compactor, err := s.agent.compactor()
	if err != nil {
		return false, err
	}
	prompt, err := NewUserEntry("Summarise this conversation:\n\n<conversation>\n" +
		transcript(latestSummary(s.logs), s.logs[from:cut], transcriptBudget(compactor.window())) + "</conversation>")
	if err != nil {
		return false, err
	}
	produced, err := compactor.step(ctx, s.client, []Entry{prompt}, nil, false)
	if err != nil {
		return false, fmt.Errorf("compact session: %w", err)
	}
	summary := strings.TrimSpace(finalText(produced))
	if summary == "" {
		return false, errors.New("compact session: the model returned an empty summary")
	}
	entry := Entry{Kind: KindCompaction, Compaction: &Compaction{Through: s.logs[cut-1].Seq, Summary: summary}}
	if n := len(produced); n > 0 {
		entry.Usage = produced[n-1].Usage
	}
	return true, s.appendLogs(ctx, entry)
}

// summaryRange returns the entries a new summary covers, s.logs[from:cut]:
// those after the previous summary, up to a turn boundary. A boundary is a
// user message or the start of a turn, where no tool call waits for its
// result; the last entry is never one, because it may start the request
// being retried. The cut keeps as much as fits in compactTail of the window;
// when nothing fits, or the request is explicit, it keeps the latest user
// message, or else the latest turn.
func (s *Session) summaryRange(explicit bool) (from, cut int) {
	if len(s.logs) < 2 {
		return 0, 0
	}
	summarised := summarisedThrough(s.logs)
	for from < len(s.logs) && s.logs[from].Seq <= summarised {
		from++
	}
	limit := int(compactTail * float64(s.agent.window()))
	fits, latestUser, latest := -1, -1, -1
	tail := estimateTokens(s.logs[len(s.logs)-1:])
	for i := len(s.logs) - 2; i > from; i-- {
		tail += estimateTokens(s.logs[i : i+1])
		kind := s.logs[i].Kind
		if kind != KindUser && kind != KindTurnStarted {
			continue
		}
		if latest < 0 {
			latest = i
		}
		if kind == KindUser && latestUser < 0 {
			latestUser = i
		}
		if tail > limit {
			if fits >= 0 {
				break
			}
			continue
		}
		fits = i
	}
	if explicit {
		fits = -1
	}
	for _, cut := range []int{fits, latestUser, latest} {
		if cut >= 0 && slices.ContainsFunc(s.logs[from:cut], func(e Entry) bool { return !e.HiddenFromModel() }) {
			return from, cut
		}
	}
	return from, from
}

// compactor returns the agent that writes summaries.
func (a *Agent) compactor() (*Agent, error) {
	opts := []AgentOption{WithInstructions(summaryInstructions), WithoutCompaction()}
	model := a.compaction.Model
	if model == "" {
		model = a.model
	}
	if p := inferProvider(model); model == a.model || p == "" || p == a.provider {
		opts = append(opts, WithProvider(a.provider), WithAPIKey(a.apiKey), WithBaseURL(a.baseURL))
	}
	if model == a.model {
		opts = append(opts, WithContextWindow(a.contextWindow))
	}
	compactor, err := New(a.name+"-compactor", model, opts...)
	if err != nil {
		return nil, fmt.Errorf("compaction model: %w", err)
	}
	compactor.maxRetries = a.maxRetries
	return compactor, nil
}

const summaryInstructions = `You summarise a conversation between a user and an AI agent, so the agent can continue the work from your summary alone. The conversation may start with an earlier summary.

Start with the language the user writes in, such as "Conversation language: English". Then cover, concisely but completely:
- the user's goals and requests, quoting the latest request exactly;
- decisions, preferences and constraints the user gave;
- what the agent has done: the exact names of the tools it called and the results that matter (names, IDs, numbers, paths, URLs, errors);
- what is still open or in progress, and the next step.

End with a section titled "Durable facts" listing every concrete detail the user stated: identifiers, names, dates, numbers, chosen options and the reasons given for them, each copied verbatim. Carry every durable fact of an earlier summary forward unchanged, however old it is. Never drop or generalise a durable fact to save space; shorten the rest instead.

Include only what the conversation says, and call the user "the user". Reply with the summary alone.`

// modelView returns the entries the model sees: those not hidden from it,
// after the latest summary, with outputs and files up to the latest
// compaction omitted.
func modelView(log []Entry) []Entry {
	summarised, omitted := summarisedThrough(log), omittedThrough(log)
	view := make([]Entry, 0, len(log)+1)
	if summary := latestSummary(log); summary != "" {
		view = append(view, Entry{Kind: KindUser, Content: []ContentPart{{Kind: ContentKindText, Text: "<conversation_summary>\n" + summary +
			"\n</conversation_summary>\nThe earlier conversation was summarised above to save context. Continue from where it left off."}}})
	}
	for _, e := range log {
		if e.HiddenFromModel() || (summarised > 0 && e.Seq <= summarised) {
			continue
		}
		if omitted > 0 && e.Seq <= omitted {
			e = omitLarge(e)
		}
		view = append(view, e)
	}
	return view
}

// omitLarge returns e with its large tool output and its files replaced by a
// short note. It does not change e's slices or pointers.
func omitLarge(e Entry) Entry {
	if r := e.ToolResult; r != nil && len(r.Output) > omitOver {
		omitted := *r
		omitted.Output = fmt.Sprintf("[output omitted to save context: %d bytes. Run the tool again if you need it.]", len(r.Output))
		e.ToolResult = &omitted
	}
	if slices.ContainsFunc(e.Content, func(p ContentPart) bool { return p.Kind == ContentKindFile }) {
		content := make([]ContentPart, len(e.Content))
		for i, part := range e.Content {
			if part.Kind == ContentKindFile {
				part = ContentPart{Kind: ContentKindText, Text: fmt.Sprintf("[file %s omitted to save context]", fileLabel(part))}
			}
			content[i] = part
		}
		e.Content = content
	}
	return e
}

func fileLabel(p ContentPart) string {
	name := p.Name
	if name == "" {
		name = p.URL
	}
	if name == "" {
		return "(" + p.MIME + ")"
	}
	return name + " (" + p.MIME + ")"
}

// omittedThrough returns the Seq up to which outputs are omitted.
func omittedThrough(log []Entry) uint64 {
	var through uint64
	for _, e := range log {
		if e.Kind == KindCompaction && e.Compaction != nil {
			through = max(through, e.Compaction.Through)
		}
	}
	return through
}

// summarisedThrough returns the Seq of the last entry the latest summary covers.
func summarisedThrough(log []Entry) uint64 {
	for _, e := range slices.Backward(log) {
		if e.Kind == KindCompaction && e.Compaction != nil && e.Compaction.Summary != "" {
			return e.Compaction.Through
		}
	}
	return 0
}

func latestSummary(log []Entry) string {
	for _, e := range slices.Backward(log) {
		if e.Kind == KindCompaction && e.Compaction != nil && e.Compaction.Summary != "" {
			return e.Compaction.Summary
		}
	}
	return ""
}

// lastResponseSeq returns the Seq of the last entry of the latest model
// response: outputs up to it have been read by the model.
func lastResponseSeq(log []Entry) uint64 {
	for _, e := range slices.Backward(log) {
		if e.Kind != KindCompaction && (e.Response != nil || e.Usage != nil) {
			return e.Seq
		}
	}
	return 0
}

func hasLargeOutputs(log []Entry, after, through uint64) bool {
	return slices.ContainsFunc(log, func(e Entry) bool {
		if e.Seq <= after || e.Seq > through {
			return false
		}
		return (e.ToolResult != nil && len(e.ToolResult.Output) > omitOver) ||
			slices.ContainsFunc(e.Content, func(p ContentPart) bool { return p.Kind == ContentKindFile })
	})
}

// contextTokens estimates the size of the next request. The latest response
// says exactly how large the context was then; only what came after it is
// estimated. After a compaction, the whole view is estimated.
func (s *Session) contextTokens() int {
	for i, e := range slices.Backward(s.logs) {
		if e.Kind == KindCompaction {
			break
		}
		if e.Usage != nil {
			return e.Usage.InputTokens + e.Usage.OutputTokens + estimateTokens(modelView(s.logs[i+1:]))
		}
	}
	return estimateTokens(modelView(s.logs)) + s.agent.fixedTokens()
}

// fixedTokens estimates the instructions and tool definitions sent with
// every request.
func (a *Agent) fixedTokens() int {
	n := len(a.instructions)
	for _, t := range a.tools {
		n += len(t.name) + len(t.description) + len(fmt.Sprint(t.schema))
	}
	return n / 4
}

// estimateTokens estimates the tokens of entries, at about four characters
// per token. It only has to be good enough to decide when to compact.
func estimateTokens(entries []Entry) int {
	n := 0
	for _, e := range entries {
		if e.HiddenFromModel() {
			continue
		}
		for _, p := range e.Content {
			switch {
			case p.Kind != ContentKindFile:
				n += len(p.Text) / 4
			case strings.HasPrefix(p.MIME, "image/"):
				n += 1600
			case strings.HasPrefix(p.MIME, "text/"):
				n += len(p.Data) / 4
			default:
				n += 2000 + len(p.Data)/200
			}
		}
		if c := e.ToolCall; c != nil {
			n += (len(c.Name) + len(c.Args)) / 4
		}
		if r := e.ToolResult; r != nil {
			n += (len(r.Output) + len(r.Error)) / 4
		}
		if r := e.Reasoning; r != nil {
			n += len(r.Summary) / 4
		}
		n += 4 // message framing
	}
	return n
}

// transcriptBudget returns the size in bytes of the transcript a summary is
// written from: transcriptMax, or half of a smaller window at about four
// bytes per token.
func transcriptBudget(window int) int {
	if window > 0 {
		return min(transcriptMax, window*2)
	}
	return transcriptMax
}

// transcript renders entries as text for the summary request, starting with
// the previous summary, in at most about budget bytes. Long tool outputs are
// cut short. When that is not enough, every message and output is cut to an
// equal share, and if even that is too long the oldest entries are left out.
func transcript(previous string, entries []Entry, budget int) string {
	var head string
	if previous != "" {
		head = fmt.Sprintf("[summary of the conversation before this point]\n%s\n\n", previous)
	}
	budget -= len(head)
	blocks := transcriptBlocks(entries, 0, transcriptCap)
	if len(blocks) > 0 && blocksSize(blocks) > budget {
		share := max(budget/len(blocks)-transcriptLabel, transcriptLabel)
		blocks = transcriptBlocks(entries, share, min(share, transcriptCap))
	}
	first, size := 0, blocksSize(blocks)
	for ; first < len(blocks) && size > budget; first++ {
		size -= len(blocks[first])
	}
	var b strings.Builder
	b.WriteString(head)
	if first > 0 {
		fmt.Fprintf(&b, "[%d earlier messages left out]\n", first)
	}
	for _, block := range blocks[first:] {
		b.WriteString(block)
	}
	return b.String()
}

// transcriptLabel is room for the label and note around a block's text.
const transcriptLabel = 100

// transcriptBlocks renders each message, tool call and tool result as a
// block, cutting texts to textCap and tool contents to toolCap bytes; a cap
// of 0 leaves them whole.
func transcriptBlocks(entries []Entry, textCap, toolCap int) []string {
	var blocks []string
	names := make(map[string]string)
	for _, e := range entries {
		if e.HiddenFromModel() {
			continue
		}
		switch e.Kind {
		case KindUser, KindAssistant:
			role := "user"
			if e.Kind == KindAssistant {
				role = "assistant"
			}
			for _, p := range e.Content {
				if p.Kind == ContentKindFile {
					blocks = append(blocks, fmt.Sprintf("[%s attached file %s]\n", role, fileLabel(p)))
				} else if text := strings.TrimSpace(p.Text); text != "" {
					blocks = append(blocks, fmt.Sprintf("[%s]\n%s\n", role, truncate(text, textCap)))
				}
			}
		case KindToolCall:
			if c := e.ToolCall; c != nil {
				names[c.ID] = c.Name
				blocks = append(blocks, fmt.Sprintf("[tool call %s]\n%s\n", c.Name, truncate(string(c.Args), toolCap)))
			}
		case KindToolResult:
			if r := e.ToolResult; r != nil {
				if r.Error != "" {
					blocks = append(blocks, fmt.Sprintf("[tool error %s]\n%s\n", names[r.CallID], truncate(r.Error, toolCap)))
				} else {
					blocks = append(blocks, fmt.Sprintf("[tool result %s]\n%s\n", names[r.CallID], truncate(r.Output, toolCap)))
				}
			}
		}
	}
	return blocks
}

func blocksSize(blocks []string) int {
	n := 0
	for _, b := range blocks {
		n += len(b)
	}
	return n
}

// truncate cuts s to at most n bytes, at a rune boundary, noting how much was
// left out. An n of 0 leaves s whole.
func truncate(s string, n int) string {
	if n <= 0 || len(s) <= n {
		return s
	}
	cut := n
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + fmt.Sprintf("… [%d more bytes]", len(s)-cut)
}
