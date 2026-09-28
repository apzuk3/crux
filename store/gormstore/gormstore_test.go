package gormstore_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/apzuk3/crux"
	"github.com/apzuk3/crux/store/gormstore"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

func setupGORMTestDB(t *testing.T) *gormstore.Store {
	t.Helper()

	db, err := gorm.Open(sqlite.Open("file:"+uuid.NewString()+"?mode=memory&cache=shared"), &gorm.Config{})
	require.NoError(t, err)

	store, err := gormstore.New(db)
	require.NoError(t, err)

	return store
}

func TestGORMStore_AppendAndGet(t *testing.T) {
	store := setupGORMTestDB(t)
	ctx := context.Background()

	agent := crux.Must(crux.New(
		"test-assistant",
		crux.ChatModelGPT4,
		crux.WithInstructions("Assist users with testing."),
		crux.WithMaxTurns(5),
	))

	sessionID := uuid.New()

	entry1, err := crux.NewUserEntry("Hello server")
	require.NoError(t, err)
	entry1.Seq = 1

	entry2 := crux.Entry{
		Seq:  2,
		At:   time.Now().UTC(),
		Kind: crux.KindAssistant,
		Content: []crux.ContentPart{
			{Kind: crux.ContentKindText, Text: "Hello! How can I help?"},
		},
	}

	// 1. Initial Append
	err = store.Append(ctx, sessionID, agent, entry1, entry2)
	require.NoError(t, err)

	// Verify agent record created
	var agentRec gormstore.AgentRecord
	err = store.DB().First(&agentRec, "id = ?", agent.ID()).Error
	require.NoError(t, err)
	require.Equal(t, agent.ID(), agentRec.ID)

	var meta map[string]any
	err = json.Unmarshal(agentRec.Data, &meta)
	require.NoError(t, err)
	require.Equal(t, "Assist users with testing.", meta["instructions"])

	// Verify session record created
	var sessRec gormstore.SessionRecord
	err = store.DB().First(&sessRec, "id = ?", sessionID).Error
	require.NoError(t, err)
	require.Equal(t, sessionID, sessRec.ID)
	require.Equal(t, agent.ID(), sessRec.AgentID)

	// Verify log records created
	var logCount int64
	err = store.DB().Model(&gormstore.LogRecord{}).Where("session_id = ?", sessionID).Count(&logCount).Error
	require.NoError(t, err)
	require.Equal(t, int64(2), logCount)

	// 2. Subsequent Append to same session
	entry3 := crux.Entry{
		Seq:  3,
		At:   time.Now().UTC(),
		Kind: crux.KindToolCall,
		ToolCall: &crux.ToolCall{
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
	err = store.DB().Model(&gormstore.SessionRecord{}).Where("id = ?", sessionID).Count(&sessionCount).Error
	require.NoError(t, err)
	require.Equal(t, int64(1), sessionCount)

	var updatedSess gormstore.SessionRecord
	err = store.DB().First(&updatedSess, "id = ?", sessionID).Error
	require.NoError(t, err)
	require.True(t, updatedSess.UpdatedAt.After(sessRec.CreatedAt) || updatedSess.UpdatedAt.Equal(sessRec.CreatedAt))

	// 3. Get all entries
	entries, err := store.Get(ctx, sessionID)
	require.NoError(t, err)
	require.Len(t, entries, 3)

	require.Equal(t, uint64(1), entries[0].Seq)
	require.Equal(t, crux.KindUser, entries[0].Kind)
	require.Equal(t, "Hello server", entries[0].Text())

	require.Equal(t, uint64(2), entries[1].Seq)
	require.Equal(t, crux.KindAssistant, entries[1].Kind)
	require.Equal(t, "Hello! How can I help?", entries[1].Text())

	require.Equal(t, uint64(3), entries[2].Seq)
	require.Equal(t, crux.KindToolCall, entries[2].Kind)
	require.Equal(t, "check_disk", entries[2].ToolCall.Name)
	require.Equal(t, `{"path":"/"}`, string(entries[2].ToolCall.Args))
}

func TestGORMStore_SessionIsolation(t *testing.T) {
	store := setupGORMTestDB(t)
	ctx := context.Background()

	agent := crux.Must(crux.New("bot", crux.ChatModelGPT4))

	session1 := uuid.New()
	session2 := uuid.New()

	e1, _ := crux.NewUserEntry("Session 1 message")
	e1.Seq = 1
	e2, _ := crux.NewUserEntry("Session 2 message")
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

func TestGORMStore_GetMissingSession(t *testing.T) {
	store := setupGORMTestDB(t)

	_, err := store.Get(context.Background(), uuid.New())
	require.ErrorIs(t, err, crux.ErrSessionNotFound)
}

func TestGORMStore_ResumeSession(t *testing.T) {
	store := setupGORMTestDB(t)
	ctx := context.Background()
	agent := crux.Must(crux.New("resume-agent", crux.ChatModelGPT4, crux.WithAPIKey("test")))

	e1, _ := crux.NewUserEntry("Initial question")
	e1.Seq = 1
	sessionID := uuid.New()
	require.NoError(t, store.Append(ctx, sessionID, agent, e1))

	resumed, err := crux.NewSession(ctx, agent, crux.WithStore(store), crux.WithSessionID(sessionID))
	require.NoError(t, err)
	require.Equal(t, sessionID, resumed.ID())
	require.Len(t, resumed.Logs(), 1)
	require.Equal(t, "Initial question", resumed.Logs()[0].Text())
}
