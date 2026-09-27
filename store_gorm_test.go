package crux

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

func setupGORMTestDB(t *testing.T) *GORMStore {
	t.Helper()

	store, err := NewInMemoryStore()
	require.NoError(t, err)

	return store
}

func TestNewInMemoryStore(t *testing.T) {
	store1, err := NewInMemoryStore()
	require.NoError(t, err)
	require.NotNil(t, store1)

	store2, err := NewMemoryStore()
	require.NoError(t, err)
	require.NotNil(t, store2)

	// Ensure stores are separate instances
	require.NotEqual(t, store1.DB(), store2.DB())
}

func TestAgentDeterministicID(t *testing.T) {
	agent1 := Must(New(
		"store-agent",
		ChatModelGPT4,
		WithInstructions("You manage server stores."),
		WithMaxTurns(10),
	))

	agent2 := Must(New(
		"store-agent",
		ChatModelGPT4,
		WithInstructions("You manage server stores."),
		WithMaxTurns(10),
	))

	// Identical blueprints must have identical IDs
	require.NotEmpty(t, agent1.ID())
	require.NotEqual(t, uuid.Nil, agent1.ID())
	require.Equal(t, agent1.ID(), agent2.ID())

	// Different instructions must produce a different ID
	agent3 := Must(New(
		"store-agent",
		ChatModelGPT4,
		WithInstructions("You manage databases."),
		WithMaxTurns(10),
	))
	require.NotEqual(t, agent1.ID(), agent3.ID())

	// Custom ID override
	customID := uuid.New()
	agentCustom := Must(New(
		"store-agent",
		ChatModelGPT4,
		WithAgentID(customID),
	))
	require.Equal(t, customID, agentCustom.ID())
}

func TestGORMStore_AppendAndGet(t *testing.T) {
	store := setupGORMTestDB(t)
	ctx := context.Background()

	agent := Must(New(
		"test-assistant",
		ChatModelGPT4,
		WithInstructions("Assist users with testing."),
		WithMaxTurns(5),
	))

	sessionID := uuid.New()

	entry1, err := NewUserEntry("Hello server")
	require.NoError(t, err)
	entry1.Seq = 1

	entry2 := Entry{
		Seq:  2,
		At:   time.Now().UTC(),
		Kind: KindAssistant,
		Content: []ContentPart{
			{Kind: ContentKindText, Text: "Hello! How can I help?"},
		},
	}

	// 1. Initial Append
	err = store.Append(ctx, sessionID, agent, entry1, entry2)
	require.NoError(t, err)

	// Verify agent record created
	var agentRec AgentRecord
	err = store.DB().First(&agentRec, "id = ?", agent.ID()).Error
	require.NoError(t, err)
	require.Equal(t, agent.ID(), agentRec.ID)

	var meta map[string]any
	err = json.Unmarshal(agentRec.Data, &meta)
	require.NoError(t, err)
	require.Equal(t, "Assist users with testing.", meta["instructions"])

	// Verify session record created
	var sessRec SessionRecord
	err = store.DB().First(&sessRec, "id = ?", sessionID).Error
	require.NoError(t, err)
	require.Equal(t, sessionID, sessRec.ID)
	require.Equal(t, agent.ID(), sessRec.AgentID)

	// Verify log records created
	var logCount int64
	err = store.DB().Model(&LogRecord{}).Where("session_id = ?", sessionID).Count(&logCount).Error
	require.NoError(t, err)
	require.Equal(t, int64(2), logCount)

	// 2. Subsequent Append to same session
	entry3 := Entry{
		Seq:  3,
		At:   time.Now().UTC(),
		Kind: KindToolCall,
		ToolCall: &ToolCall{
			ID:   "call_123",
			Name: "check_disk",
			Args: []byte(`{"path":"/"}`),
		},
	}

	time.Sleep(10 * time.Millisecond) // ensure timestamp progresses
	err = store.Append(ctx, sessionID, agent, entry3)
	require.NoError(t, err)

	// Verify session was updated (not duplicated)
	var sessionCount int64
	err = store.DB().Model(&SessionRecord{}).Where("id = ?", sessionID).Count(&sessionCount).Error
	require.NoError(t, err)
	require.Equal(t, int64(1), sessionCount)

	var updatedSess SessionRecord
	err = store.DB().First(&updatedSess, "id = ?", sessionID).Error
	require.NoError(t, err)
	require.True(t, updatedSess.UpdatedAt.After(sessRec.CreatedAt) || updatedSess.UpdatedAt.Equal(sessRec.CreatedAt))

	// 3. Get all entries
	entries, err := store.Get(ctx, sessionID)
	require.NoError(t, err)
	require.Len(t, entries, 3)

	require.Equal(t, uint64(1), entries[0].Seq)
	require.Equal(t, KindUser, entries[0].Kind)
	require.Equal(t, "Hello server", entries[0].Text())

	require.Equal(t, uint64(2), entries[1].Seq)
	require.Equal(t, KindAssistant, entries[1].Kind)
	require.Equal(t, "Hello! How can I help?", entries[1].Text())

	require.Equal(t, uint64(3), entries[2].Seq)
	require.Equal(t, KindToolCall, entries[2].Kind)
	require.Equal(t, "check_disk", entries[2].ToolCall.Name)
	require.Equal(t, `{"path":"/"}`, string(entries[2].ToolCall.Args))
}

func TestGORMStore_SessionIsolation(t *testing.T) {
	store := setupGORMTestDB(t)
	ctx := context.Background()

	agent := Must(New("bot", ChatModelGPT4))

	session1 := uuid.New()
	session2 := uuid.New()

	e1, _ := NewUserEntry("Session 1 message")
	e1.Seq = 1
	e2, _ := NewUserEntry("Session 2 message")
	e2.Seq = 1

	require.NoError(t, store.Append(ctx, session1, agent, e1))
	require.NoError(t, store.Append(ctx, session2, agent, e2))

	logs1, err := store.Get(ctx, session1)
	require.NoError(t, err)
	require.Len(t, logs1, 1)
	require.Equal(t, "Session 1 message", logs1[0].Text())

	logs2, err := store.Get(ctx, session2)
	require.NoError(t, err)
	require.Len(t, logs2, 1)
	require.Equal(t, "Session 2 message", logs2[0].Text())
}

func TestNewSession_CreationWithStore(t *testing.T) {
	store := setupGORMTestDB(t)
	ctx := context.Background()

	agent := Must(New("storage-agent", ChatModelGPT4))

	// 1. NewSession with store
	session, err := NewSession(ctx, agent, WithStore(store))
	require.NoError(t, err)
	require.NotNil(t, session)

	// 2. On first turn/append, records are created in DB
	entry, _ := NewUserEntry("ping")
	require.NoError(t, session.appendLogs(ctx, entry))

	var sessRec SessionRecord
	err = store.DB().First(&sessRec, "id = ?", session.ID()).Error
	require.NoError(t, err)
	require.Equal(t, session.ID(), sessRec.ID)
	require.Equal(t, agent.ID(), sessRec.AgentID)

	var agentRec AgentRecord
	err = store.DB().First(&agentRec, "id = ?", agent.ID()).Error
	require.NoError(t, err)
	require.Equal(t, agent.ID(), agentRec.ID)

	// 3. NewSession with custom SessionID
	customID := uuid.New()
	customSession, err := NewSession(ctx, agent, WithStore(store), WithSessionID(customID))
	require.NoError(t, err)
	require.Equal(t, customID, customSession.ID())
}

func TestResumeSession_SuccessAndNotFound(t *testing.T) {
	store := setupGORMTestDB(t)
	ctx := context.Background()

	agent := Must(New("resume-agent", ChatModelGPT4))

	// 1. Resuming non-existent session returns ErrSessionNotFound
	_, err := ResumeSession(ctx, uuid.New(), store, agent)
	require.ErrorIs(t, err, ErrSessionNotFound)

	// 2. Create session, append logs, then resume
	session, err := NewSession(ctx, agent, WithStore(store))
	require.NoError(t, err)

	entry1, _ := NewUserEntry("Initial question")
	entry2 := Entry{
		Seq:     2,
		At:      time.Now().UTC(),
		Kind:    KindAssistant,
		Content: []ContentPart{{Kind: ContentKindText, Text: "Initial answer"}},
	}

	err = session.appendLogs(ctx, entry1, entry2)
	require.NoError(t, err)

	// 3. Resume the session
	resumed, err := ResumeSession(ctx, session.ID(), store, agent)
	require.NoError(t, err)
	require.Equal(t, session.ID(), resumed.ID())
	require.Equal(t, agent, resumed.Agent())

	resumedLogs := resumed.Logs()
	require.Len(t, resumedLogs, 2)
	require.Equal(t, "Initial question", resumedLogs[0].Text())
	require.Equal(t, "Initial answer", resumedLogs[1].Text())

	// 4. Appending to resumed session persists to the same session in DB
	entry3, _ := NewUserEntry("Follow up")
	err = resumed.appendLogs(ctx, entry3)
	require.NoError(t, err)

	dbEntries, err := store.Get(ctx, session.ID())
	require.NoError(t, err)
	require.Len(t, dbEntries, 3)
	require.Equal(t, "Follow up", dbEntries[2].Text())
}

func TestSession_ForkWithStore(t *testing.T) {
	store := setupGORMTestDB(t)
	ctx := context.Background()

	agent := Must(New("fork-agent", ChatModelGPT4))

	session, err := NewSession(ctx, agent, WithStore(store))
	require.NoError(t, err)

	entry1, _ := NewUserEntry("Message 1")
	entry2, _ := NewUserEntry("Message 2")
	require.NoError(t, session.appendLogs(ctx, entry1, entry2))

	// Fork the session
	forked, err := session.Fork(ctx)
	require.NoError(t, err)
	require.NotEqual(t, session.ID(), forked.ID())
	require.Len(t, forked.Logs(), 2)

	// Appending to forked session persists to store under forked.ID()
	entry3, _ := NewUserEntry("Forked message 3")
	require.NoError(t, forked.appendLogs(ctx, entry3))

	forkedDBLogs, err := store.Get(ctx, forked.ID())
	require.NoError(t, err)
	require.Len(t, forkedDBLogs, 1)
	require.Equal(t, "Forked message 3", forkedDBLogs[0].Text())

	// Verify original session logs are untouched
	origDBLogs, err := store.Get(ctx, session.ID())
	require.NoError(t, err)
	require.Len(t, origDBLogs, 2)
}
