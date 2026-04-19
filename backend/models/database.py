"""
MedTermQuest — Database Models
SQLite + SQLAlchemy for portability. Switch to PostgreSQL for production.
"""

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean,
    DateTime, Text, JSON, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import uuid

DATABASE_URL = "sqlite:///./medterm.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def new_id() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=new_id)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, default="")
    role = Column(String, default="student")  # student, resident, attending
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

    sessions = relationship("QuizSession", back_populates="user", cascade="all, delete-orphan")
    progress = relationship("UserProgress", back_populates="user", uselist=False, cascade="all, delete-orphan")


class UserProgress(Base):
    """Persistent belief state and cumulative stats per user."""
    __tablename__ = "user_progress"

    id = Column(String, primary_key=True, default=new_id)
    user_id = Column(String, ForeignKey("users.id"), unique=True, nullable=False)

    # Serialized belief state (JSON dict: term_id -> float)
    beliefs = Column(JSON, default={})
    # Set of term IDs the user has been tested on (ever)
    tested_ids = Column(JSON, default=[])

    # Cumulative statistics
    total_questions = Column(Integer, default=0)
    total_correct = Column(Integer, default=0)
    total_sessions = Column(Integer, default=0)
    total_time_ms = Column(Integer, default=0)

    # Entropy trajectory (list of floats, one per session)
    entropy_history = Column(JSON, default=[])
    # Mastery per category
    category_mastery = Column(JSON, default={})

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="progress")


class QuizSession(Base):
    """Individual quiz session record."""
    __tablename__ = "quiz_sessions"

    id = Column(String, primary_key=True, default=new_id)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)

    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    mode = Column(String, default="adaptive")  # adaptive, category, difficulty
    category_filter = Column(String, nullable=True)
    difficulty_filter = Column(Integer, nullable=True)

    # Session metrics
    questions_asked = Column(Integer, default=0)
    correct_answers = Column(Integer, default=0)
    entropy_start = Column(Float, default=0.0)
    entropy_end = Column(Float, nullable=True)
    total_information_gained = Column(Float, default=0.0)
    avg_response_time_ms = Column(Integer, default=0)

    # Serialized results list
    results = Column(JSON, default=[])

    user = relationship("User", back_populates="sessions")
    answers = relationship("Answer", back_populates="session", cascade="all, delete-orphan")


class Answer(Base):
    """Individual answer record within a session."""
    __tablename__ = "answers"

    id = Column(String, primary_key=True, default=new_id)
    session_id = Column(String, ForeignKey("quiz_sessions.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)

    term_id = Column(String, nullable=False)
    term_name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    difficulty = Column(Integer, nullable=False)

    is_correct = Column(Boolean, nullable=False)
    selected_option = Column(String, nullable=True)
    correct_option = Column(String, nullable=False)
    response_time_ms = Column(Integer, default=0)

    entropy_before = Column(Float, default=0.0)
    entropy_after = Column(Float, default=0.0)
    information_gained = Column(Float, default=0.0)
    belief_after = Column(Float, default=0.0)
    eig_score = Column(Float, default=0.0)   # EIG that caused this term to be selected

    answered_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("QuizSession", back_populates="answers")


def create_tables():
    Base.metadata.create_all(bind=engine)