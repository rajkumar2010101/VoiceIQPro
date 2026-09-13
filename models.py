"""
models.py  —  VoiceIQ Pro database models
SQLite via SQLAlchemy — zero config, single file DB
Tables: User, ConversationRecord, LearningFeedback, AlertThreshold, APIConfig
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = "users"
    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(80), unique=True, nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password   = db.Column(db.String(200), nullable=False)
    role       = db.Column(db.String(20), default="analyst")   # owner | analyst
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active  = db.Column(db.Boolean, default=True)

    def __repr__(self):
        return f"<User {self.username} [{self.role}]>"


class ConversationRecord(db.Model):
    __tablename__ = "conversations"
    id                   = db.Column(db.Integer, primary_key=True)
    conversation_id      = db.Column(db.String(50))
    date                 = db.Column(db.String(20))
    agent                = db.Column(db.String(100))
    shift                = db.Column(db.String(30))
    channel              = db.Column(db.String(30))
    customer_message     = db.Column(db.Text)
    sentiment            = db.Column(db.String(20))
    sentiment_score      = db.Column(db.Float)
    emotion              = db.Column(db.String(30))
    topic                = db.Column(db.String(50))
    escalation_score     = db.Column(db.Integer)
    escalation_level     = db.Column(db.String(10))
    escalation_reasons   = db.Column(db.String(200))
    aspects              = db.Column(db.Text)   # JSON
    csat_score           = db.Column(db.Float)
    resolved             = db.Column(db.Boolean)
    wait_time_mins       = db.Column(db.Integer)
    handle_time_mins     = db.Column(db.Integer)
    source               = db.Column(db.String(50), default="upload")   # upload | api | crm
    imported_at          = db.Column(db.DateTime, default=datetime.utcnow)
    uploaded_by          = db.Column(db.Integer, db.ForeignKey("users.id"))


class LearningFeedback(db.Model):
    """Reinforcement learning feedback — human corrections on AI predictions."""
    __tablename__ = "learning_feedback"
    id                   = db.Column(db.Integer, primary_key=True)
    conversation_id      = db.Column(db.String(50))
    original_message     = db.Column(db.Text)
    ai_sentiment         = db.Column(db.String(20))
    correct_sentiment    = db.Column(db.String(20))
    ai_topic             = db.Column(db.String(50))
    correct_topic        = db.Column(db.String(50))
    ai_emotion           = db.Column(db.String(30))
    correct_emotion      = db.Column(db.String(30))
    ai_escalation        = db.Column(db.String(10))
    correct_escalation   = db.Column(db.String(10))
    feedback_note        = db.Column(db.Text)
    submitted_by         = db.Column(db.Integer, db.ForeignKey("users.id"))
    submitted_at         = db.Column(db.DateTime, default=datetime.utcnow)
    applied_to_model     = db.Column(db.Boolean, default=False)


class AlertThreshold(db.Model):
    """User-configurable alert thresholds for real-time monitoring."""
    __tablename__ = "alert_thresholds"
    id                   = db.Column(db.Integer, primary_key=True)
    owner_id             = db.Column(db.Integer, db.ForeignKey("users.id"))
    metric               = db.Column(db.String(50))  # e.g. neg_sentiment_pct, high_esc_pct, aht_mins
    threshold_value      = db.Column(db.Float)
    operator             = db.Column(db.String(5), default=">")  # > or <
    alert_label          = db.Column(db.String(100))
    severity             = db.Column(db.String(20), default="warning")  # critical | warning | info
    enabled              = db.Column(db.Boolean, default=True)
    created_at           = db.Column(db.DateTime, default=datetime.utcnow)


class APIConfig(db.Model):
    """Stored CRM/NLP API connections."""
    __tablename__ = "api_configs"
    id           = db.Column(db.Integer, primary_key=True)
    owner_id     = db.Column(db.Integer, db.ForeignKey("users.id"))
    name         = db.Column(db.String(100))         # e.g. "Five9 Production"
    provider     = db.Column(db.String(50))           # five9 | salesforce | freshdesk | zendesk | custom
    api_type     = db.Column(db.String(20))           # crm | nlp
    endpoint_url = db.Column(db.String(300))
    api_key_hint = db.Column(db.String(50))           # last 4 chars only — never store full key
    field_map    = db.Column(db.Text)                 # JSON mapping their fields → our fields
    is_active    = db.Column(db.Boolean, default=True)
    last_sync    = db.Column(db.DateTime)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)


class RealtimeMetric(db.Model):
    """Snapshot of real-time contact center metrics (pushed via API or simulated)."""
    __tablename__ = "realtime_metrics"
    id              = db.Column(db.Integer, primary_key=True)
    recorded_at     = db.Column(db.DateTime, default=datetime.utcnow)
    queue_name      = db.Column(db.String(100))
    agents_available= db.Column(db.Integer)
    agents_on_call  = db.Column(db.Integer)
    agents_in_aux   = db.Column(db.Integer)
    calls_waiting   = db.Column(db.Integer)
    longest_wait_sec= db.Column(db.Integer)
    aht_sec         = db.Column(db.Integer)
    asa_sec         = db.Column(db.Integer)
    service_level   = db.Column(db.Float)
    abandonment_rate= db.Column(db.Float)
    aux_code        = db.Column(db.String(50))
    source          = db.Column(db.String(50), default="simulated")
