package crux

import (
	"context"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/glebarez/sqlite"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

// openTestDB opens a file-backed SQLite database, so concurrent writers wait
// for each other instead of failing with a locked table.
func openTestDB(t *testing.T) *gorm.DB {
	t.Helper()
	dsn := "file:" + filepath.Join(t.TempDir(), "crux.db") + "?_pragma=busy_timeout(10000)"
	db, err := gorm.Open(sqlite.Open(dsn), &gorm.Config{Logger: logger.Discard})
	require.NoError(t, err)
	// Close before t.TempDir removes the directory: Windows can't delete an
	// open file.
	sqlDB, err := db.DB()
	require.NoError(t, err)
	t.Cleanup(func() { sqlDB.Close() })
	return db
}

func testEntries(seqs ...uint64) []Entry {
	entries := make([]Entry, len(seqs))
	for i, seq := range seqs {
		entries[i] = Entry{Seq: seq, At: time.Now().UTC(), Kind: KindUser, Content: []ContentPart{{Kind: ContentKindText, Text: "hi"}}}
	}
	return entries
}

func TestGORMStoreAppendUpsertsAgentAndSession(t *testing.T) {
	ctx := context.Background()
	db := openTestDB(t)
	store, err := NewGORMStore(db)
	require.NoError(t, err)
	agent := Must(New("support", OpenAIGPT4))
	session := &Session{id: uuid.New(), agent: agent}

	require.NoError(t, store.Append(ctx, session, testEntries(1, 2)...))
	require.NoError(t, store.Append(ctx, session, testEntries(3)...))
	require.NoError(t, store.Append(ctx, session), "no entries still records the session")

	var agents, sessions, logs int64
	require.NoError(t, db.Model(&agentRecord{}).Count(&agents).Error)
	require.NoError(t, db.Model(&sessionRecord{}).Count(&sessions).Error)
	require.NoError(t, db.Model(&logRecord{}).Count(&logs).Error)
	require.Equal(t, []int64{1, 1, 3}, []int64{agents, sessions, logs})

	var stored sessionRecord
	require.NoError(t, db.First(&stored, "id = ?", dbUUID(session.id)).Error)
	require.Equal(t, dbUUID(agent.ID()), stored.AgentID)
	require.Nil(t, stored.ParentID)

	entries, err := store.Get(ctx, session.id)
	require.NoError(t, err)
	require.Len(t, entries, 3)
	for i, e := range entries {
		require.Equal(t, uint64(i+1), e.Seq)
		require.Equal(t, "hi", e.Text())
	}
}

func TestGORMStoreAppendRecordsParent(t *testing.T) {
	ctx := context.Background()
	db := openTestDB(t)
	store, err := NewGORMStore(db)
	require.NoError(t, err)
	agent := Must(New("support", OpenAIGPT4))
	parent := &Session{id: uuid.New(), agent: agent}
	child := &Session{id: uuid.New(), parentID: parent.id, agent: agent}

	require.NoError(t, store.Append(ctx, parent, testEntries(1)...))
	require.NoError(t, store.Append(ctx, child, testEntries(1)...))

	var stored sessionRecord
	require.NoError(t, db.First(&stored, "id = ?", dbUUID(child.id)).Error)
	require.NotNil(t, stored.ParentID)
	require.Equal(t, parent.id, uuid.UUID(*stored.ParentID))
}

func TestGORMStoreAppendRejectsStaleSeq(t *testing.T) {
	ctx := context.Background()
	store, err := NewGORMStore(openTestDB(t))
	require.NoError(t, err)
	session := &Session{id: uuid.New(), agent: Must(New("support", OpenAIGPT4))}

	require.NoError(t, store.Append(ctx, session, testEntries(1, 2)...))
	err = store.Append(ctx, session, testEntries(2, 3)...)
	require.ErrorIs(t, err, ErrSessionConflict)
	require.ErrorContains(t, err, "already has entry 2")

	entries, err := store.Get(ctx, session.id)
	require.NoError(t, err)
	require.Len(t, entries, 2, "a conflicting write stores nothing")
}

func TestGORMStoreAppendRejectsConcurrentDuplicate(t *testing.T) {
	ctx := context.Background()
	store, err := NewGORMStore(openTestDB(t))
	require.NoError(t, err)
	session := &Session{id: uuid.New(), agent: Must(New("support", OpenAIGPT4))}
	require.NoError(t, store.Append(ctx, session, testEntries(1)...))

	const writers = 4
	errs := make([]error, writers)
	var wg sync.WaitGroup
	for i := range writers {
		wg.Go(func() { errs[i] = store.Append(ctx, session, testEntries(2, 3)...) })
	}
	wg.Wait()

	var won int
	for _, err := range errs {
		if err == nil {
			won++
			continue
		}
		require.ErrorIs(t, err, ErrSessionConflict)
	}
	require.Equal(t, 1, won, "exactly one writer appends entry 2")

	entries, err := store.Get(ctx, session.id)
	require.NoError(t, err)
	require.Len(t, entries, 3)
}

// TestGORMStoreAppendReportsUniqueIndexAsConflict drives the path where two
// writers pass the seq check and the unique index rejects the second. The
// duplicate row is inserted from a create callback, and the store runs on an
// outer transaction without nesting, so the row survives the failed write
// the way a committed concurrent write would.
func TestGORMStoreAppendReportsUniqueIndexAsConflict(t *testing.T) {
	ctx := context.Background()
	db := openTestDB(t)
	require.NoError(t, db.AutoMigrate(&agentRecord{}, &sessionRecord{}, &logRecord{}))
	session := &Session{id: uuid.New(), agent: Must(New("support", OpenAIGPT4))}

	outer := db.Session(&gorm.Session{DisableNestedTransaction: true}).Begin()
	require.NoError(t, outer.Error)
	t.Cleanup(func() { outer.Rollback() })
	store := &GORMStore{db: outer}

	var once sync.Once
	require.NoError(t, db.Callback().Create().Before("gorm:create").Register("crux_test_duplicate", func(tx *gorm.DB) {
		if tx.Statement.Table != (logRecord{}).TableName() {
			return
		}
		once.Do(func() {
			tx.AddError(tx.Session(&gorm.Session{NewDB: true}).Exec(
				"INSERT INTO crux_session_logs (session_id, seq, data, created_at) VALUES (?, ?, ?, ?)",
				dbUUID(session.id), 1, []byte("{}"), time.Now()).Error)
		})
	}))

	err := store.Append(ctx, session, testEntries(1)...)
	require.ErrorIs(t, err, ErrSessionConflict)
	require.ErrorContains(t, err, "already has entry 1")
}
