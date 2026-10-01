package crux

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"sync"

	"github.com/apzuk3/crux/internal/tui"
	"github.com/google/uuid"
)

// CLI opens an interactive chat with agent in the terminal and blocks until
// the user quits. The chat streams answers, shows each tool call and subagent
// as it runs, and asks the user to approve tools registered with
// WithApprovalNeeded. opts configure the session: for example, WithStore and
// WithSessionID continue a stored conversation.
func CLI(agent *Agent, opts ...SessionOption) error {
	if agent == nil {
		return errors.New("agent cannot be nil")
	}
	ctx := context.Background()
	c := &cli{children: map[uuid.UUID]string{}}
	opts = append(opts, WithEntryHandler(c.onEntry))
	session, err := NewSession(ctx, agent, opts...)
	if err != nil {
		return err
	}
	c.session = session

	info := tui.Info{
		Agent:     agent.name,
		Provider:  string(agent.provider),
		Model:     agent.model,
		SessionID: session.id.String(),
		MaxTurns:  agent.maxTurns,
		Tools:     cliTools(agent.tools),
		Models:    cliModels(agent),
	}
	for _, e := range session.logs {
		c.track(session, e)
		if e.Kind.lifecycle() {
			continue
		}
		if ev, ok := c.event(session, e); ok {
			info.History = append(info.History, ev)
		}
	}

	c.app = tui.New(info, c)
	return c.app.Run()
}

// cli connects a session to the terminal chat.
type cli struct {
	session *Session
	app     *tui.App

	mu       sync.Mutex
	children map[uuid.UUID]string // subagent session ID -> the call that runs it
}

func (c *cli) Run(ctx context.Context, input string) error {
	var in any
	if input != "" {
		in = input
	}
	var err error
	for chunk, streamErr := range c.session.Stream(ctx, in) {
		if streamErr != nil {
			err = streamErr
			continue
		}
		kind := tui.EventText
		if chunk.Kind == ChunkReasoning {
			kind = tui.EventReasoningText
		}
		c.app.Emit(tui.Event{Kind: kind, Agent: c.session.agent.name, Text: chunk.Delta})
	}
	if errors.Is(err, ErrApprovalNeeded) || errors.Is(err, ErrRefused) {
		return nil // the chat shows both from the log
	}
	return err
}

func (c *cli) Pending() []tui.Call {
	var calls []tui.Call
	for _, call := range c.session.PendingApprovals() {
		calls = append(calls, tui.Call{ID: call.ID, Name: call.Name, Args: string(call.Args), Agent: call.Agent})
	}
	return calls
}

func (c *cli) Decide(ctx context.Context, id string, approve bool) error {
	if approve {
		return c.session.Approve(ctx, id)
	}
	return c.session.Reject(ctx, id, "")
}

// SwitchModel continues the conversation with another model, in a fork of
// the session that keeps the history and every entry handler.
func (c *cli) SwitchModel(ctx context.Context, provider, model string) (string, error) {
	handlers := func(s *Session) error {
		s.onEntry = slices.Clone(c.session.onEntry)
		return nil
	}
	fork, err := c.session.forkWith(ctx, len(c.session.logs), []SessionOption{handlers},
		WithProvider(Provider(provider)), WithModel(model))
	if err != nil {
		return "", err
	}
	c.session = fork
	return fork.id.String(), nil
}

func (c *cli) onEntry(_ context.Context, s *Session, e Entry) {
	c.track(s, e)
	if ev, ok := c.event(s, e); ok {
		c.app.Emit(ev)
	}
}

// track remembers which call each subagent session runs for, so the chat can
// show a subagent's work inside the card of the call that started it.
func (c *cli) track(s *Session, e Entry) {
	if e.Kind != KindToolCall || e.ToolCall == nil {
		return
	}
	for _, t := range s.agent.tools {
		if t.name == e.ToolCall.Name && t.kind == toolKindSubagent {
			key := openToolCall{call: e.ToolCall, seq: e.Seq}.key()
			c.mu.Lock()
			c.children[childSessionID(s.id, key)] = e.ToolCall.ID
			c.mu.Unlock()
			return
		}
	}
}

// event describes an entry to the chat. It reports false for entries the
// chat does not show and that carry no token usage.
func (c *cli) event(s *Session, e Entry) (tui.Event, bool) {
	ev := tui.Event{Agent: s.agent.name}
	if s != c.session {
		c.mu.Lock()
		ev.Parent = c.children[s.id]
		c.mu.Unlock()
	}
	if e.Usage != nil {
		ev.Usage = &tui.Usage{In: e.Usage.InputTokens, Out: e.Usage.OutputTokens, CacheRead: e.Usage.CacheReadTokens}
	}
	if e.Response != nil {
		ev.FirstToken = e.Response.FirstTokenAfter
	}

	switch e.Kind {
	case KindRunStarted:
		ev.Kind = tui.EventRunStarted
	case KindRunFinished:
		ev.Kind = tui.EventRunFinished
		if e.Run != nil {
			ev.Outcome, ev.Err = string(e.Run.Outcome), e.Run.Error
		}
	case KindTurnStarted:
		ev.Kind = tui.EventTurnStarted
		if e.Turn != nil {
			ev.Provider, ev.Model = string(e.Turn.Provider), e.Turn.Model
		}
	case KindUser:
		ev.Kind, ev.Text = tui.EventUser, e.Text()
	case KindAssistant:
		ev.Kind, ev.Text = tui.EventAssistant, e.Text()
		for _, part := range e.Content {
			if part.Kind == ContentKindRefusal && ev.Text == "" {
				ev.Kind, ev.Text = tui.EventRefusal, part.Text
			}
		}
	case KindReasoning:
		ev.Kind = tui.EventReasoning
		if e.Reasoning != nil {
			ev.Text = e.Reasoning.Summary
		}
	case KindToolCall, KindToolStarted:
		if e.ToolCall == nil {
			return ev, false
		}
		ev.Kind = tui.EventToolCall
		if e.Kind == KindToolStarted {
			ev.Kind = tui.EventToolStarted
		}
		ev.Call = tui.Call{ID: e.ToolCall.ID, Name: e.ToolCall.Name, Args: string(e.ToolCall.Args), Agent: s.agent.name}
	case KindToolResult:
		if e.ToolResult == nil {
			return ev, false
		}
		ev.Kind = tui.EventToolResult
		ev.Call.ID = e.ToolResult.CallID
		ev.Output, ev.Err, ev.Denied, ev.Duration = e.ToolResult.Output, e.ToolResult.Error, e.ToolResult.Denied, e.Duration
	case KindApproval:
		if e.Approval == nil {
			return ev, false
		}
		ev.Kind, ev.Call.ID, ev.Approved = tui.EventApproval, e.Approval.CallID, e.Approval.Approved
	default:
		ev.Kind = tui.EventOther
		return ev, ev.Usage != nil
	}
	return ev, true
}

// cliTools describes tools, and each subagent's own tools, for the sidebar.
func cliTools(tools []Tool) []tui.Tool {
	out := make([]tui.Tool, 0, len(tools))
	for _, t := range tools {
		info := tui.Tool{
			Name:        t.name,
			Description: t.description,
			Approval:    t.approvalNeeded,
			Toolset:     t.toolset,
		}
		if t.kind == toolKindSubagent && t.subAgent != nil {
			info.IsSubAgent = true
			info.SubAgent = cliTools(t.subAgent.tools)
			if info.Description == "" {
				info.Description = fmt.Sprintf("Subagent %s (%s)", t.subAgent.name, t.subAgent.model)
			}
		}
		out = append(out, info)
	}
	return out
}

// snapshot matches model IDs pinned to a release date, which the model picker
// leaves out.
var snapshot = regexp.MustCompile(`(19|20)\d{2}-?\d{2}-?\d{2}`)

// cliModels lists the models the chat can switch to: those of the agent's
// provider and of every provider with an API key in the environment.
func cliModels(agent *Agent) []tui.Model {
	var providers []Provider
	for p, spec := range providerSpecs {
		if p == agent.provider || firstEnv(spec.envVars) != "" {
			providers = append(providers, p)
		}
	}
	slices.Sort(providers)
	slices.SortStableFunc(providers, func(a, b Provider) int {
		switch {
		case a == agent.provider:
			return -1
		case b == agent.provider:
			return 1
		}
		return 0
	})

	models := []tui.Model{{Provider: string(agent.provider), Name: agent.model}}
	for _, p := range providers {
		for _, m := range providerSpecs[p].models {
			if (p == agent.provider && m == agent.model) || snapshot.MatchString(m) {
				continue
			}
			models = append(models, tui.Model{Provider: string(p), Name: m})
		}
	}
	return models
}
