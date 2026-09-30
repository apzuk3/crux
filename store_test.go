package crux

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

func TestAgentDeterministicID(t *testing.T) {
	agent1 := Must(New("store-agent", OpenAIGPT4, WithInstructions("You manage server stores."), WithMaxTurns(10)))
	agent2 := Must(New("store-agent", OpenAIGPT4, WithInstructions("You manage server stores."), WithMaxTurns(10)))

	// Identical blueprints must have identical IDs
	require.NotEqual(t, uuid.Nil, agent1.ID())
	require.Equal(t, agent1.ID(), agent2.ID())

	// Different instructions must produce a different ID
	agent3 := Must(New("store-agent", OpenAIGPT4, WithInstructions("You manage databases."), WithMaxTurns(10)))
	require.NotEqual(t, agent1.ID(), agent3.ID())

	// A tool with the same name but another definition produces a different ID
	withTool := func(description string) *Agent {
		reg := NewToolsRegistry()
		RegisterToolWithRegistry(reg, "lookup", description, func(ctx context.Context, in struct{}) (string, *StateDelta, error) { return "", nil, nil })
		return Must(New("store-agent", OpenAIGPT4, WithToolsRegistry([]string{"lookup"}, reg)))
	}
	require.Equal(t, withTool("Looks up a record.").ID(), withTool("Looks up a record.").ID())
	require.NotEqual(t, withTool("Looks up a record.").ID(), withTool("Deletes a record.").ID())
}

func TestMemoryStore_AppendAndGet(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()
	agent := Must(New("bot", OpenAIGPT4))
	session1, session2 := uuid.New(), uuid.New()

	_, err := store.Get(ctx, session1)
	require.ErrorIs(t, err, ErrSessionNotFound)

	e1, _ := NewUserEntry("Session 1 message")
	e2, _ := NewUserEntry("Session 2 message")
	require.NoError(t, store.Append(ctx, &Session{id: session1, agent: agent}, e1))
	require.NoError(t, store.Append(ctx, &Session{id: session2, agent: agent}, e2))

	logs1, err := store.Get(ctx, session1)
	require.NoError(t, err)
	require.Len(t, logs1, 1)
	require.Equal(t, "Session 1 message", logs1[0].Text())

	// Returned entries are copies.
	logs1[0].Content[0].Text = "mutated"
	again, err := store.Get(ctx, session1)
	require.NoError(t, err)
	require.Equal(t, "Session 1 message", again[0].Text())

	logs2, err := store.Get(ctx, session2)
	require.NoError(t, err)
	require.Equal(t, "Session 2 message", logs2[0].Text())
}

func TestNewSession_DefaultsToMemoryStore(t *testing.T) {
	session, err := NewSession(context.Background(), Must(New("bot", OpenAIGPT4)))
	require.NoError(t, err)
	require.IsType(t, &MemoryStore{}, session.Store())
}

func TestNewSession_RejectsNilStore(t *testing.T) {
	_, err := NewSession(context.Background(), Must(New("bot", OpenAIGPT4)), WithStore(nil))
	require.Error(t, err)
}

func TestNewSession_LoadsExistingSession(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()
	agent := Must(New("resume-agent", OpenAIGPT4))

	session, err := NewSession(ctx, agent, WithStore(store))
	require.NoError(t, err)

	entry1, _ := NewUserEntry("Initial question")
	entry2 := Entry{
		At:      time.Now().UTC(),
		Kind:    KindAssistant,
		Content: []ContentPart{{Kind: ContentKindText, Text: "Initial answer"}},
	}
	require.NoError(t, session.appendLogs(ctx, entry1, entry2))

	resumed, err := NewSession(ctx, agent, WithStore(store), WithSessionID(session.ID()))
	require.NoError(t, err)
	require.Equal(t, session.ID(), resumed.ID())
	require.Len(t, resumed.Logs(), 2)
	require.Equal(t, "Initial answer", resumed.Logs()[1].Text())

	entry3, _ := NewUserEntry("Follow up")
	require.NoError(t, resumed.appendLogs(ctx, entry3))
	stored, err := store.Get(ctx, session.ID())
	require.NoError(t, err)
	require.Len(t, stored, 3)
	require.Equal(t, uint64(3), stored[2].Seq)

	// Seeding history into a session that already has some is ambiguous.
	_, err = NewSession(ctx, agent, WithStore(store), WithSessionID(session.ID()), WithSessionLogs([]Entry{entry3}))
	require.Error(t, err)
}

func TestNewSession_UnknownIDStartsFresh(t *testing.T) {
	id := uuid.New()
	session, err := NewSession(context.Background(), Must(New("bot", OpenAIGPT4)), WithSessionID(id))
	require.NoError(t, err)
	require.Equal(t, id, session.ID())
	require.Empty(t, session.Logs())
}

func TestSession_ForkPersistsCopiedHistory(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()
	agent := Must(New("fork-agent", OpenAIGPT4))

	session, err := NewSession(ctx, agent, WithStore(store))
	require.NoError(t, err)
	entry1, _ := NewUserEntry("Message 1")
	entry2, _ := NewUserEntry("Message 2")
	require.NoError(t, session.appendLogs(ctx, entry1, entry2))

	forked, err := session.Fork(ctx)
	require.NoError(t, err)
	require.NotEqual(t, session.ID(), forked.ID())

	entry3, _ := NewUserEntry("Forked message 3")
	require.NoError(t, forked.appendLogs(ctx, entry3))

	// The fork can be resumed with its whole history, not just what came after the fork.
	resumed, err := NewSession(ctx, forked.Agent(), WithStore(store), WithSessionID(forked.ID()))
	require.NoError(t, err)
	require.Len(t, resumed.Logs(), 3)
	require.Equal(t, "Forked message 3", resumed.Logs()[2].Text())

	original, err := store.Get(ctx, session.ID())
	require.NoError(t, err)
	require.Len(t, original, 2)
}

func TestSessionOutsideToolHasNoParent(t *testing.T) {
	session, err := NewSession(context.Background(), Must(New("bot", OpenAIGPT4)))
	require.NoError(t, err)
	require.Equal(t, uuid.Nil, session.parentID)
}

type flakyStore struct {
	*MemoryStore
	fail bool
}

func (f *flakyStore) Append(ctx context.Context, session *Session, entries ...Entry) error {
	if f.fail {
		return errors.New("database unavailable")
	}
	return f.MemoryStore.Append(ctx, session, entries...)
}

func TestFailedAppendLeavesSessionUnchanged(t *testing.T) {
	ctx := context.Background()
	store := &flakyStore{MemoryStore: NewMemoryStore()}
	session := MustSession(NewSession(ctx, Must(New("bot", OpenAIGPT4)), WithStore(store)))

	user, _ := NewUserEntry("hi")
	require.NoError(t, session.appendLogs(ctx, user))

	store.fail = true
	err := session.appendLogs(ctx, Entry{Kind: KindToolCall, ToolCall: &ToolCall{ID: "c1", Name: "lookup"}})
	require.ErrorContains(t, err, "database unavailable")
	require.Len(t, session.Logs(), 1, "an entry the store rejected must not join the session")

	store.fail = false
	require.NoError(t, session.appendLogs(ctx,
		Entry{Kind: KindToolCall, ToolCall: &ToolCall{ID: "c1", Name: "lookup"}},
		Entry{Kind: KindToolResult, ToolResult: &ToolResult{CallID: "c1"}},
	))

	stored, err := store.Get(ctx, session.ID())
	require.NoError(t, err)
	require.Equal(t, session.Logs(), stored)
	require.Equal(t, []uint64{1, 2, 3}, []uint64{stored[0].Seq, stored[1].Seq, stored[2].Seq})
}

func TestEntryHandlerSeesOnlyStoredEntries(t *testing.T) {
	ctx := context.Background()
	store := &flakyStore{MemoryStore: NewMemoryStore()}
	var seen []Entry
	session := MustSession(NewSession(ctx, Must(New("bot", OpenAIGPT4)), WithStore(store),
		WithEntryHandler(func(_ context.Context, _ *Session, e Entry) { seen = append(seen, e) })))

	store.fail = true
	_, err := session.Run(ctx, "hi")
	require.ErrorContains(t, err, "database unavailable")
	require.Empty(t, seen, "an entry the store rejected must not be reported")

	store.fail = false
	user, _ := NewUserEntry("hi")
	require.NoError(t, session.appendLogs(ctx, user))
	require.Equal(t, session.Logs(), seen)
}
