// Package gormstore persists crux sessions in any database supported by GORM.
//
// Bring your own driver, for example gorm.io/driver/sqlite (cgo),
// github.com/glebarez/sqlite (pure Go) or gorm.io/driver/postgres:
//
//	db, _ := gorm.Open(sqlite.Open("crux.db"), &gorm.Config{})
//	store, _ := gormstore.New(db)
//	session, _ := crux.NewSession(ctx, agent, crux.WithStore(store))
package gormstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/apzuk3/crux"
	"github.com/google/uuid"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

var _ crux.Store = (*Store)(nil)

// AgentRecord stores the immutable blueprint definition for an agent in GORM.
type AgentRecord struct {
	ID        uuid.UUID `gorm:"type:uuid;primaryKey"`
	Data      []byte    `gorm:"type:bytes;not null"`
	CreatedAt time.Time `gorm:"not null"`
	UpdatedAt time.Time `gorm:"not null"`
}

func (AgentRecord) TableName() string { return "crux_agents" }

// SessionRecord stores active conversation instances linked to an agent blueprint in GORM.
type SessionRecord struct {
	ID        uuid.UUID `gorm:"type:uuid;primaryKey"`
	AgentID   uuid.UUID `gorm:"type:uuid;not null;index"`
	CreatedAt time.Time `gorm:"not null"`
	UpdatedAt time.Time `gorm:"not null;index"`
}

func (SessionRecord) TableName() string { return "crux_sessions" }

// LogRecord stores sequential log entries belonging to a session in GORM.
type LogRecord struct {
	ID        uint64    `gorm:"primaryKey;autoIncrement"`
	SessionID uuid.UUID `gorm:"type:uuid;not null;index:idx_session_seq"`
	Seq       uint64    `gorm:"not null;index:idx_session_seq"`
	Data      []byte    `gorm:"type:bytes;not null"`
	CreatedAt time.Time `gorm:"not null"`
}

func (LogRecord) TableName() string { return "crux_session_logs" }

// Store implements crux.Store backed by a GORM database.
type Store struct {
	db *gorm.DB
}

// New initializes the store and automatically runs migrations for the
// crux_agents, crux_sessions, and crux_session_logs tables.
func New(db *gorm.DB) (*Store, error) {
	if db == nil {
		return nil, errors.New("db cannot be nil")
	}

	if err := db.AutoMigrate(&AgentRecord{}, &SessionRecord{}, &LogRecord{}); err != nil {
		return nil, fmt.Errorf("crux gorm automigrate: %w", err)
	}

	return &Store{db: db}, nil
}

// DB returns the underlying *gorm.DB instance.
func (s *Store) DB() *gorm.DB {
	return s.db
}

// Append persists newly produced log entries for a session within a single transaction,
// creating or updating the corresponding crux_agents and crux_sessions records.
func (s *Store) Append(ctx context.Context, sessionID uuid.UUID, agent *crux.Agent, entries ...crux.Entry) error {
	if agent == nil {
		return errors.New("agent cannot be nil")
	}

	return s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		now := time.Now().UTC()

		// 1. Upsert crux_agents
		agentRec := AgentRecord{
			ID:        agent.ID(),
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
		sessRec := SessionRecord{
			ID:        sessionID,
			AgentID:   agent.ID(),
			CreatedAt: now,
			UpdatedAt: now,
		}
		if err := tx.Clauses(clause.OnConflict{
			Columns:   []clause.Column{{Name: "id"}},
			DoUpdates: clause.AssignmentColumns([]string{"updated_at"}),
		}).Create(&sessRec).Error; err != nil {
			return fmt.Errorf("upsert session record: %w", err)
		}

		// 3. Insert crux_session_logs
		if len(entries) > 0 {
			logs := make([]LogRecord, len(entries))
			for i, entry := range entries {
				data, err := json.Marshal(entry)
				if err != nil {
					return fmt.Errorf("marshal log entry seq %d: %w", entry.Seq, err)
				}
				at := entry.At
				if at.IsZero() {
					at = now
				}
				logs[i] = LogRecord{
					SessionID: sessionID,
					Seq:       entry.Seq,
					Data:      data,
					CreatedAt: at,
				}
			}

			if err := tx.Create(&logs).Error; err != nil {
				return fmt.Errorf("insert session logs: %w", err)
			}
		}

		return nil
	})
}

// Get retrieves all log entries for the specified session in sequential order.
// If the session does not exist in the store, it returns crux.ErrSessionNotFound.
func (s *Store) Get(ctx context.Context, sessionID uuid.UUID) ([]crux.Entry, error) {
	var count int64
	if err := s.db.WithContext(ctx).
		Model(&SessionRecord{}).
		Where("id = ?", sessionID).
		Count(&count).Error; err != nil {
		return nil, fmt.Errorf("check session existence: %w", err)
	}
	if count == 0 {
		return nil, crux.ErrSessionNotFound
	}

	var records []LogRecord
	if err := s.db.WithContext(ctx).
		Where("session_id = ?", sessionID).
		Order("seq ASC").
		Find(&records).Error; err != nil {
		return nil, fmt.Errorf("get session logs: %w", err)
	}

	entries := make([]crux.Entry, len(records))
	for i, r := range records {
		if err := json.Unmarshal(r.Data, &entries[i]); err != nil {
			return nil, fmt.Errorf("unmarshal log entry id %d: %w", r.ID, err)
		}
	}

	return entries, nil
}
