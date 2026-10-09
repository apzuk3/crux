package crux

import (
	"context"
	"database/sql/driver"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	"gorm.io/gorm/schema"
)

// Store defines the persistence contract for agent session state and logs.
type Store interface {
	// Append writes newly produced entries for the session. The session also
	// gives access to its ID and agent for recording session metadata.
	// Append must be all-or-nothing. Entries join the session only after
	// Append succeeds, so after an error the session is unchanged and the
	// entries are produced again: tool calls whose results were not stored
	// run again on the next Run or Resume. Append must fail with
	// ErrSessionConflict when the first entry's Seq is not greater than every
	// Seq already stored, which means another writer got there first.
	Append(ctx context.Context, session *Session, entries ...Entry) error

	// Get retrieves all log entries for the session in sequential order.
	// It returns ErrSessionNotFound when nothing was ever appended for the session.
	Get(ctx context.Context, sessionID uuid.UUID) ([]Entry, error)
}

var _ Store = (*MemoryStore)(nil)

// MemoryStore keeps session logs in process memory. It is the default store
// for new sessions and is safe for concurrent use. Its contents are lost when
// the process exits; use a persistent store such as GORMStore to keep them.
type MemoryStore struct {
	mu       sync.RWMutex
	sessions map[uuid.UUID]*memorySession
}

type memorySession struct {
	parentID uuid.UUID
	entries  []Entry
}

// NewMemoryStore returns an empty in-memory store.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{sessions: make(map[uuid.UUID]*memorySession)}
}

// Append records entries for the session, creating it on first use.
func (m *MemoryStore) Append(ctx context.Context, session *Session, entries ...Entry) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if session == nil {
		return errors.New("session cannot be nil")
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	stored, ok := m.sessions[session.id]
	if !ok {
		stored = &memorySession{parentID: session.parentID}
		m.sessions[session.id] = stored
	}
	if n := len(stored.entries); n > 0 && len(entries) > 0 && entries[0].Seq <= stored.entries[n-1].Seq {
		return fmt.Errorf("%w: session %s already has entry %d", ErrSessionConflict, session.id, entries[0].Seq)
	}
	stored.entries = append(stored.entries, cloneEntries(entries)...)
	return nil
}

// Get returns a copy of the session's entries in the order they were appended.
func (m *MemoryStore) Get(ctx context.Context, sessionID uuid.UUID) ([]Entry, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	m.mu.RLock()
	defer m.mu.RUnlock()
	stored, ok := m.sessions[sessionID]
	if !ok {
		return nil, ErrSessionNotFound
	}
	return cloneEntries(stored.entries), nil
}

var _ Store = (*GORMStore)(nil)

// The tables are internal; query them through DB() with your own structs if
// you need to.
type agentRecord struct {
	ID        dbUUID    `gorm:"primaryKey"`
	Data      []byte    `gorm:"type:bytes;not null"`
	CreatedAt time.Time `gorm:"not null"`
	UpdatedAt time.Time `gorm:"not null"`
}

func (agentRecord) TableName() string { return "crux_agents" }

type sessionRecord struct {
	ID        dbUUID    `gorm:"primaryKey"`
	AgentID   dbUUID    `gorm:"not null;index"`
	ParentID  *dbUUID   `gorm:"index"` // session whose tool call created this one, for tracing
	CreatedAt time.Time `gorm:"not null"`
	UpdatedAt time.Time `gorm:"not null;index"`
}

func (sessionRecord) TableName() string { return "crux_sessions" }

// bySessionID is the GORM condition that selects one session's log entries.
const bySessionID = "session_id = ?"

type logRecord struct {
	ID        uint64    `gorm:"primaryKey;autoIncrement"`
	SessionID dbUUID    `gorm:"not null;uniqueIndex:idx_crux_session_logs_seq"`
	Seq       uint64    `gorm:"not null;uniqueIndex:idx_crux_session_logs_seq"`
	Data      []byte    `gorm:"type:bytes;not null"`
	CreatedAt time.Time `gorm:"not null"`
}

func (logRecord) TableName() string { return "crux_session_logs" }

// dbUUID stores a UUID in the column type each database supports.
type dbUUID uuid.UUID

func (u dbUUID) Value() (driver.Value, error) { return uuid.UUID(u).String(), nil }

func (u *dbUUID) Scan(src any) error {
	var id uuid.UUID
	if err := id.Scan(src); err != nil {
		return err
	}
	*u = dbUUID(id)
	return nil
}

func (dbUUID) GormDBDataType(db *gorm.DB, _ *schema.Field) string {
	switch db.Dialector.Name() {
	case "postgres":
		return "uuid"
	case "sqlserver":
		return "uniqueidentifier"
	default:
		return "char(36)"
	}
}

// GORMStore persists sessions in any database GORM supports. Bring your own
// driver, for example github.com/glebarez/sqlite (pure Go),
// gorm.io/driver/sqlite (cgo) or gorm.io/driver/postgres:
//
//	db, _ := gorm.Open(sqlite.Open("db"), &gorm.Config{})
//	store, _ := NewGORMStore(db)
//	session, _ := NewSession(ctx, agent, WithStore(store))
type GORMStore struct {
	db *gorm.DB
}

// NewGORMStore initializes the store and automatically runs migrations for the
// crux_agents, crux_sessions, and crux_session_logs tables.
func NewGORMStore(db *gorm.DB) (*GORMStore, error) {
	if db == nil {
		return nil, errors.New("db cannot be nil")
	}

	if err := db.AutoMigrate(&agentRecord{}, &sessionRecord{}, &logRecord{}); err != nil {
		return nil, fmt.Errorf("crux gorm automigrate: %w", err)
	}

	return &GORMStore{db: db}, nil
}

// Append persists newly produced log entries for a session within a single transaction,
// creating or updating the corresponding crux_agents and crux_sessions records.
func (s *GORMStore) Append(ctx context.Context, session *Session, entries ...Entry) error {
	if session == nil {
		return errors.New("session cannot be nil")
	}
	if session.agent == nil {
		return errors.New("agent cannot be nil")
	}
	err := s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		return appendIn(tx, session, entries)
	})
	return s.asConflict(ctx, session.id, entries, err)
}

// appendIn writes the session's records and its new entries in tx, refusing
// entries another writer already stored.
func appendIn(tx *gorm.DB, session *Session, entries []Entry) error {
	now := time.Now().UTC()
	if err := upsertAgentAndSession(tx, session, now); err != nil {
		return err
	}
	if len(entries) == 0 {
		return nil
	}
	last, err := lastLogSeq(tx, session.id)
	if err != nil {
		return fmt.Errorf("read last log seq: %w", err)
	}
	if entries[0].Seq <= last {
		return seqConflict(session.id, entries[0].Seq)
	}
	logs, err := logRecords(session.id, entries, now)
	if err != nil {
		return err
	}
	return insertLogs(tx, logs)
}

func upsertAgentAndSession(tx *gorm.DB, session *Session, now time.Time) error {
	agent := session.agent
	agentRec := agentRecord{
		ID:        dbUUID(agent.ID()),
		Data:      agent.canonicalData(),
		CreatedAt: now,
		UpdatedAt: now,
	}
	if err := tx.Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "id"}},
		DoUpdates: clause.AssignmentColumns([]string{"data", "updated_at"}),
	}).Create(&agentRec).Error; err != nil {
		return fmt.Errorf("upsert agent record: %w", err)
	}

	sessRec := sessionRecord{
		ID:        dbUUID(session.id),
		AgentID:   dbUUID(agent.ID()),
		CreatedAt: now,
		UpdatedAt: now,
	}
	if session.parentID != uuid.Nil {
		parentID := dbUUID(session.parentID)
		sessRec.ParentID = &parentID
	}
	if err := tx.Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "id"}},
		DoUpdates: clause.AssignmentColumns([]string{"updated_at"}),
	}).Create(&sessRec).Error; err != nil {
		return fmt.Errorf("upsert session record: %w", err)
	}
	return nil
}

// lastLogSeq returns the highest Seq stored for the session, or 0.
func lastLogSeq(db *gorm.DB, sessionID uuid.UUID) (uint64, error) {
	var last uint64
	err := db.Model(&logRecord{}).
		Where(bySessionID, dbUUID(sessionID)).
		Select("COALESCE(MAX(seq), 0)").
		Scan(&last).Error
	return last, err
}

func seqConflict(sessionID uuid.UUID, seq uint64) error {
	return fmt.Errorf("%w: session %s already has entry %d", ErrSessionConflict, sessionID, seq)
}

func logRecords(sessionID uuid.UUID, entries []Entry, now time.Time) ([]logRecord, error) {
	logs := make([]logRecord, len(entries))
	for i, entry := range entries {
		data, err := json.Marshal(entry)
		if err != nil {
			return nil, fmt.Errorf("marshal log entry seq %d: %w", entry.Seq, err)
		}
		at := entry.At
		if at.IsZero() {
			at = now
		}
		logs[i] = logRecord{
			SessionID: dbUUID(sessionID),
			Seq:       entry.Seq,
			Data:      data,
			CreatedAt: at,
		}
	}
	return logs, nil
}

// insertLogs writes the records in batches, which keep each INSERT under
// the database's limit on bound parameters (999 on older SQLite); the
// transaction keeps the write all-or-nothing.
func insertLogs(tx *gorm.DB, logs []logRecord) error {
	if err := tx.CreateInBatches(&logs, 200).Error; err != nil {
		return fmt.Errorf("insert session logs: %w", err)
	}
	return nil
}

// asConflict reports err as ErrSessionConflict when another writer stored
// the entries meanwhile: two writers can both pass the seq check, and the
// unique (session_id, seq) index then rejects the second with a
// driver-specific error.
func (s *GORMStore) asConflict(ctx context.Context, sessionID uuid.UUID, entries []Entry, err error) error {
	if err == nil || errors.Is(err, ErrSessionConflict) || len(entries) == 0 {
		return err
	}
	if last, readErr := lastLogSeq(s.db.WithContext(ctx), sessionID); readErr == nil && entries[0].Seq <= last {
		return seqConflict(sessionID, entries[0].Seq)
	}
	return err
}

// Get retrieves all log entries for the specified session in sequential order.
// If the session does not exist in the store, it returns ErrSessionNotFound.
func (s *GORMStore) Get(ctx context.Context, sessionID uuid.UUID) ([]Entry, error) {
	var count int64
	if err := s.db.WithContext(ctx).
		Model(&sessionRecord{}).
		Where("id = ?", dbUUID(sessionID)).
		Count(&count).Error; err != nil {
		return nil, fmt.Errorf("check session existence: %w", err)
	}
	if count == 0 {
		return nil, ErrSessionNotFound
	}

	var records []logRecord
	if err := s.db.WithContext(ctx).
		Where(bySessionID, dbUUID(sessionID)).
		Order("seq ASC, id ASC").
		Find(&records).Error; err != nil {
		return nil, fmt.Errorf("get session logs: %w", err)
	}

	entries := make([]Entry, len(records))
	for i, r := range records {
		if err := json.Unmarshal(r.Data, &entries[i]); err != nil {
			return nil, fmt.Errorf("unmarshal log entry id %d: %w", r.ID, err)
		}
	}

	return entries, nil
}
