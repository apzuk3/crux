// Package tui is the terminal chat behind crux.CLI. It knows nothing about
// crux: the crux package describes the agent with Info, drives it through a
// Backend and reports what happens as Events.
package tui

import (
	"context"
	"errors"
	"sync"
	"time"

	tea "charm.land/bubbletea/v2"
)

// Info describes the agent the chat talks to.
type Info struct {
	Agent     string
	Provider  string
	Model     string
	SessionID string
	MaxTurns  int
	Tools     []Tool
	Models    []Model // models the user can switch to, the current one first
	History   []Event // earlier conversation, shown before the first prompt
}

// Model is a model the user can switch to.
type Model struct {
	Provider, Name string
}

// Tool is a tool the agent may call. SubAgent lists a subagent's own tools
// and is nil for ordinary tools.
type Tool struct {
	Name        string
	Description string
	Approval    bool
	Toolset     string
	SubAgent    []Tool
	IsSubAgent  bool
}

// Call is a tool call.
type Call struct {
	ID    string
	Name  string
	Args  string
	Agent string // agent that made the call
}

// Usage counts tokens.
type Usage struct {
	In, Out, CacheRead int
}

// EventKind identifies an Event.
type EventKind int

const (
	EventOther         EventKind = iota // carries only usage
	EventRunStarted                     // Outcome, Err on EventRunFinished
	EventRunFinished                    //
	EventTurnStarted                    // Provider, Model
	EventUser                           // Text
	EventText                           // streamed answer delta
	EventReasoningText                  // streamed reasoning delta
	EventAssistant                      // Text: a stored model message
	EventReasoning                      // Text: a stored reasoning summary
	EventRefusal                        // Text
	EventToolCall                       // Call
	EventToolStarted                    // Call.ID
	EventToolResult                     // Call.ID, Output, Err, Denied, Duration
	EventApproval                       // Call.ID, Approved
)

// Event is something that happened in the session or one of its subagents.
// Parent is the ID of the subagent call whose session produced it, empty for
// the top-level session.
type Event struct {
	Kind   EventKind
	Agent  string
	Parent string

	Text     string
	Call     Call
	Output   string
	Err      string
	Denied   bool
	Approved bool
	Duration time.Duration

	Outcome  string
	Provider string
	Model    string

	Usage      *Usage
	FirstToken time.Duration
}

// Backend runs the agent. Its methods are called one at a time.
type Backend interface {
	// Run sends input to the agent, or resumes the run when input is empty,
	// and reports progress through App.Emit until it returns.
	Run(ctx context.Context, input string) error
	// Pending lists the tool calls waiting for the user's decision.
	Pending() []Call
	// Decide approves or rejects a pending call.
	Decide(ctx context.Context, id string, approve bool) error
	// SwitchModel continues the conversation with another model and returns
	// the ID of the session it continues in.
	SwitchModel(ctx context.Context, provider, model string) (sessionID string, err error)
}

// App is a running chat.
type App struct {
	program *tea.Program
	runs    sync.WaitGroup
	cancel  context.CancelFunc
}

// New prepares the chat.
func New(info Info, backend Backend) *App {
	ctx, cancel := context.WithCancel(context.Background())
	app := &App{cancel: cancel}
	m := newModel(info, backend, ctx, &app.runs)
	app.program = tea.NewProgram(m)
	return app
}

// Emit reports an event to the chat. It may be called from any goroutine.
func (a *App) Emit(e Event) {
	a.program.Send(eventMsg(e))
}

// Run shows the chat and blocks until the user quits. A run still in progress
// is cancelled and waited for, so it records how it ended.
func (a *App) Run() error {
	_, err := a.program.Run()
	a.cancel()
	a.runs.Wait()
	if errors.Is(err, tea.ErrInterrupted) {
		return nil
	}
	return err
}
