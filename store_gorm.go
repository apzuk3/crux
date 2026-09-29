package crux

import (
	"context"
	"database/sql/driver"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	"gorm.io/gorm/schema"
)

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

// DB returns the underlying *gorm.DB instance.
func (s *GORMStore) DB() *gorm.DB {
	return s.db
}

// Append persists newly produced log entries for a session within a single transaction,
// creating or updating the corresponding crux_agents and crux_sessions records.
func (s *GORMStore) Append(ctx context.Context, session *Session, entries ...Entry) error {
	if session == nil {
		return errors.New("session cannot be nil")
	}
	sessionID, agent := session.id, session.agent
	if agent == nil {
		return errors.New("agent cannot be nil")
	}

	err := s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		now := time.Now().UTC()

		// 1. Upsert crux_agents
		agentRec := agentRecord{
			ID:        dbUUID(agent.ID()),
			Data:      agent.CanonicalData(),
			CreatedAt: now,
			UpdatedAt: now,
		}
		if err := tx.Clauses(clause.OnConflict{
			Columns:   []clause.Column{{Name: "id"}},
			DoUpdates: clause.AssignmentColumns([]string{"data", "updated_at"}),
		}).Create(&agentRec).Error; err != nil {
			return fmt.Errorf("upsert agent record: %w", err)
		}

		// 2. Upsert crux_sessions
		sessRec := sessionRecord{
			ID:        dbUUID(sessionID),
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

		// 3. Insert crux_session_logs, refusing entries another writer already
		// stored. The unique (session_id, seq) index catches concurrent writers.
		if len(entries) > 0 {
			var last uint64
			if err := tx.Model(&logRecord{}).
				Where("session_id = ?", dbUUID(sessionID)).
				Select("COALESCE(MAX(seq), 0)").
				Scan(&last).Error; err != nil {
				return fmt.Errorf("read last log seq: %w", err)
			}
			if entries[0].Seq <= last {
				return fmt.Errorf("%w: session %s already has entry %d", ErrSessionConflict, sessionID, entries[0].Seq)
			}

			logs := make([]logRecord, len(entries))
			for i, entry := range entries {
				data, err := json.Marshal(entry)
				if err != nil {
					return fmt.Errorf("marshal log entry seq %d: %w", entry.Seq, err)
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

			// Batches keep each INSERT under the database's limit on bound
			// parameters (999 on older SQLite); the transaction keeps the
			// write all-or-nothing.
			if err := tx.CreateInBatches(&logs, 200).Error; err != nil {
				return fmt.Errorf("insert session logs: %w", err)
			}
		}

		return nil
	})
	if err == nil || errors.Is(err, ErrSessionConflict) || len(entries) == 0 {
		return err
	}
	// Two writers can both pass the seq check; the unique index then rejects
	// the second with a driver-specific error. Report it as a conflict.
	var last uint64
	if s.db.WithContext(ctx).Model(&logRecord{}).
		Where("session_id = ?", dbUUID(sessionID)).
		Select("COALESCE(MAX(seq), 0)").
		Scan(&last).Error == nil && entries[0].Seq <= last {
		return fmt.Errorf("%w: session %s already has entry %d", ErrSessionConflict, sessionID, entries[0].Seq)
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
		Where("session_id = ?", dbUUID(sessionID)).
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
