"""
MedTermQuest — Progress Router
Per-user analytics, history, leaderboard, and knowledge state.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from datetime import datetime, timedelta

from models.database import get_db, User, UserProgress, QuizSession, Answer
from core.entropy_engine import EntropyEngine
from data.medical_terms import MEDICAL_TERMS, TERMS_BY_CATEGORY
from routers.auth import get_current_user
from routers.quiz import build_engine_from_user

router = APIRouter(prefix="/progress", tags=["Progress"])


@router.get("/overview")
def get_progress_overview(current_user: User = Depends(get_current_user),
                          db: Session = Depends(get_db)):
    """Full progress dashboard data for the current user."""
    progress = db.query(UserProgress).filter(
        UserProgress.user_id == current_user.id
    ).first()

    engine = build_engine_from_user(current_user.id, db)

    sessions = (db.query(QuizSession)
                .filter(QuizSession.user_id == current_user.id,
                        QuizSession.ended_at.isnot(None))
                .order_by(QuizSession.started_at.desc())
                .limit(20).all())

    # Accuracy trend (last 10 sessions)
    accuracy_trend = []
    for s in reversed(sessions[:10]):
        if s.questions_asked > 0:
            accuracy_trend.append({
                "session_id": s.id,
                "date": s.started_at.isoformat(),
                "accuracy": round(s.correct_answers / s.questions_asked * 100, 1),
                "questions": s.questions_asked,
                "entropy_delta": round((s.entropy_start or 0) - (s.entropy_end or 0), 4),
            })

    # Entropy trajectory
    entropy_history = []
    if progress and progress.entropy_history:
        entropy_history = progress.entropy_history[-20:]

    # Category breakdown
    category_mastery = engine.get_category_mastery()

    # Weakest terms (lowest belief, not yet mastered)
    weak_terms = sorted(
        [{"term": MEDICAL_TERMS[i]["term"],
          "category": MEDICAL_TERMS[i]["category"],
          "difficulty": MEDICAL_TERMS[i]["difficulty"],
          "mastery": round(engine.beliefs[MEDICAL_TERMS[i]["id"]] * 100, 1),
          "tested": MEDICAL_TERMS[i]["id"] in engine.asked_ids}
         for i in range(len(MEDICAL_TERMS))],
        key=lambda x: x["mastery"]
    )[:8]

    # Strong terms (highest belief)
    strong_terms = sorted(
        [{"term": MEDICAL_TERMS[i]["term"],
          "category": MEDICAL_TERMS[i]["category"],
          "mastery": round(engine.beliefs[MEDICAL_TERMS[i]["id"]] * 100, 1)}
         for i in range(len(MEDICAL_TERMS))],
        key=lambda x: x["mastery"], reverse=True
    )[:8]

    # Streak calculation
    streak = _calculate_streak(current_user.id, db)

    # Overall stats
    total_q = progress.total_questions if progress else 0
    total_c = progress.total_correct if progress else 0
    overall_accuracy = round(total_c / total_q * 100, 1) if total_q else 0

    return {
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "full_name": current_user.full_name,
            "role": current_user.role,
        },
        "stats": {
            "total_sessions": progress.total_sessions if progress else 0,
            "total_questions": total_q,
            "total_correct": total_c,
            "overall_accuracy": overall_accuracy,
            "overall_mastery": engine.overall_mastery(),
            "current_entropy": round(engine.get_knowledge_entropy(), 4),
            "terms_tested": len(engine.asked_ids),
            "terms_total": len(MEDICAL_TERMS),
            "streak_days": streak,
        },
        "category_mastery": category_mastery,
        "accuracy_trend": accuracy_trend,
        "entropy_history": entropy_history,
        "weak_terms": weak_terms,
        "strong_terms": strong_terms,
        "weakest_areas": engine.get_weakest_areas(3),
        "strongest_areas": engine.get_strongest_areas(3),
        "recent_sessions": [_format_session(s) for s in sessions[:5]],
    }


@router.get("/sessions")
def get_session_history(current_user: User = Depends(get_current_user),
                        db: Session = Depends(get_db),
                        limit: int = 20, offset: int = 0):
    """Paginated session history."""
    sessions = (db.query(QuizSession)
                .filter(QuizSession.user_id == current_user.id,
                        QuizSession.ended_at.isnot(None))
                .order_by(QuizSession.started_at.desc())
                .offset(offset).limit(limit).all())

    total = (db.query(func.count(QuizSession.id))
             .filter(QuizSession.user_id == current_user.id,
                     QuizSession.ended_at.isnot(None))
             .scalar())

    return {
        "sessions": [_format_session(s) for s in sessions],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/sessions/{session_id}")
def get_session_detail(session_id: str,
                       current_user: User = Depends(get_current_user),
                       db: Session = Depends(get_db)):
    """Detailed view of one session including per-answer breakdown."""
    session = db.query(QuizSession).filter(
        QuizSession.id == session_id,
        QuizSession.user_id == current_user.id,
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    answers = (db.query(Answer)
               .filter(Answer.session_id == session_id)
               .order_by(Answer.answered_at).all())

    return {
        **_format_session(session),
        "answers": [_format_answer(a) for a in answers],
    }


@router.get("/leaderboard")
def get_leaderboard(db: Session = Depends(get_db), limit: int = 10):
    """Top users by overall mastery score."""
    progresses = (db.query(UserProgress, User)
                  .join(User, UserProgress.user_id == User.id)
                  .filter(User.is_active == True)
                  .all())

    board = []
    for prog, user in progresses:
        if prog.total_questions >= 5:  # minimum questions to appear
            engine = EntropyEngine()
            if prog.beliefs:
                engine.beliefs.update(prog.beliefs)
            board.append({
                "rank": 0,
                "username": user.username,
                "full_name": user.full_name or user.username,
                "role": user.role,
                "overall_mastery": engine.overall_mastery(),
                "total_sessions": prog.total_sessions,
                "total_questions": prog.total_questions,
                "accuracy": round(prog.total_correct / prog.total_questions * 100, 1)
                            if prog.total_questions else 0,
                "entropy": round(engine.get_knowledge_entropy(), 3),
            })

    board.sort(key=lambda x: x["overall_mastery"], reverse=True)
    for i, entry in enumerate(board[:limit]):
        entry["rank"] = i + 1

    return {"leaderboard": board[:limit], "total_users": len(board)}


@router.get("/knowledge-map")
def get_knowledge_map(current_user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    """Full per-term belief state for the knowledge map visualization."""
    engine = build_engine_from_user(current_user.id, db)

    terms_data = []
    for term in MEDICAL_TERMS:
        terms_data.append({
            "id": term["id"],
            "term": term["term"],
            "category": term["category"],
            "difficulty": term["difficulty"],
            "mastery": round(engine.beliefs[term["id"]] * 100, 1),
            "tested": term["id"] in engine.asked_ids,
            "entropy": round(
                -(engine.beliefs[term["id"]] * __import__('math').log2(max(1e-9, engine.beliefs[term["id"]]))
                  + (1 - engine.beliefs[term["id"]]) * __import__('math').log2(max(1e-9, 1 - engine.beliefs[term["id"]]))),
                3
            ),
        })

    return {
        "terms": terms_data,
        "overall_entropy": round(engine.get_knowledge_entropy(), 4),
        "overall_mastery": engine.overall_mastery(),
        "category_summary": engine.get_category_mastery(),
    }


def _format_session(s: QuizSession) -> dict:
    accuracy = round(s.correct_answers / s.questions_asked * 100, 1) if s.questions_asked else 0
    return {
        "id": s.id,
        "mode": s.mode,
        "category_filter": s.category_filter,
        "difficulty_filter": s.difficulty_filter,
        "started_at": s.started_at.isoformat() if s.started_at else None,
        "ended_at": s.ended_at.isoformat() if s.ended_at else None,
        "questions_asked": s.questions_asked,
        "correct_answers": s.correct_answers,
        "accuracy": accuracy,
        "entropy_start": round(s.entropy_start or 0, 4),
        "entropy_end": round(s.entropy_end or 0, 4),
        "information_gained": round(s.total_information_gained or 0, 4),
    }


def _format_answer(a: Answer) -> dict:
    return {
        "term_id": a.term_id,
        "term_name": a.term_name,
        "category": a.category,
        "difficulty": a.difficulty,
        "is_correct": a.is_correct,
        "selected_option": a.selected_option,
        "correct_option": a.correct_option,
        "response_time_ms": a.response_time_ms,
        "entropy_before": round(a.entropy_before or 0, 4),
        "entropy_after": round(a.entropy_after or 0, 4),
        "information_gained": round(a.information_gained or 0, 4),
        "belief_after": round(a.belief_after or 0, 3),
        "eig_score": round(a.eig_score or 0, 4),
        "answered_at": a.answered_at.isoformat() if a.answered_at else None,
    }


def _calculate_streak(user_id: str, db: Session) -> int:
    """Calculate consecutive days with at least one completed session."""
    sessions = (db.query(QuizSession.started_at)
                .filter(QuizSession.user_id == user_id,
                        QuizSession.ended_at.isnot(None))
                .order_by(QuizSession.started_at.desc()).all())

    if not sessions:
        return 0

    streak = 0
    today = datetime.utcnow().date()
    check_date = today

    session_dates = {s.started_at.date() for s in sessions}

    while check_date in session_dates:
        streak += 1
        check_date -= timedelta(days=1)

    return streak