"""
MedTermQuest — Quiz Router
Adaptive question selection using the entropy engine.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uuid, random

from models.database import get_db, User, UserProgress, QuizSession, Answer
from core.entropy_engine import EntropyEngine
from data.medical_terms import MEDICAL_TERMS, TERM_BY_ID, CATEGORIES
from routers.auth import get_current_user

router = APIRouter(prefix="/quiz", tags=["Quiz"])


# ── MCQ Generation ────────────────────────────────────────────────────────────

def generate_mcq(term: dict, all_terms: list) -> dict:
    """
    Generate a 4-option multiple choice question for a term.
    Distractors are selected from the same category (harder) or across categories.
    """
    correct_answer = term["definition"]

    # Prefer same-category distractors for harder discrimination
    same_cat = [t for t in all_terms
                if t["category"] == term["category"] and t["id"] != term["id"]]
    other_cat = [t for t in all_terms
                 if t["category"] != term["category"] and t["id"] != term["id"]]

    distractor_pool = same_cat + other_cat[:max(0, 3 - len(same_cat))]
    random.shuffle(distractor_pool)
    distractors = [t["definition"] for t in distractor_pool[:3]]

    # If not enough, pad with other cat
    while len(distractors) < 3:
        t = random.choice(other_cat)
        if t["definition"] not in distractors:
            distractors.append(t["definition"])

    options = distractors[:3] + [correct_answer]
    random.shuffle(options)

    return {
        "question": f"Which best defines: **{term['term']}**?",
        "options": options,
        "correct": correct_answer,
        "option_labels": ["A", "B", "C", "D"],
    }


def build_engine_from_user(user_id: str, db: Session) -> EntropyEngine:
    """Reconstruct EntropyEngine from persisted belief state."""
    progress = db.query(UserProgress).filter(UserProgress.user_id == user_id).first()
    engine = EntropyEngine()
    if progress and progress.beliefs:
        engine.beliefs.update(progress.beliefs)
    if progress and progress.tested_ids:
        engine.asked_ids = set(progress.tested_ids)
    return engine


def save_engine_to_user(engine: EntropyEngine, user_id: str, db: Session):
    """Persist updated belief state back to DB."""
    progress = db.query(UserProgress).filter(UserProgress.user_id == user_id).first()
    if progress:
        progress.beliefs = engine.beliefs
        progress.tested_ids = list(engine.asked_ids)
        progress.updated_at = datetime.utcnow()
        db.commit()


# ── Schemas ───────────────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    mode: str = "adaptive"          # adaptive | category | difficulty | review
    category_filter: Optional[str] = None
    difficulty_filter: Optional[int] = None
    num_questions: int = 10

class SubmitAnswerRequest(BaseModel):
    session_id: str
    term_id: str
    selected_answer: str
    response_time_ms: int = 0

class NextQuestionResponse(BaseModel):
    term_id: str
    term_name: str
    category: str
    difficulty: int
    root: str
    mnemonic: str
    question: str
    options: List[str]
    eig_score: float
    entropy_before: float
    questions_remaining: int
    session_progress: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/start")
def start_session(req: StartSessionRequest,
                  current_user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    """Start a new adaptive quiz session."""
    engine = build_engine_from_user(current_user.id, db)
    entropy_start = engine.get_knowledge_entropy()

    # Validate category filter
    if req.category_filter and req.category_filter not in CATEGORIES:
        raise HTTPException(status_code=400,
                            detail=f"Invalid category. Valid: {CATEGORIES}")

    session = QuizSession(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        mode=req.mode,
        category_filter=req.category_filter,
        difficulty_filter=req.difficulty_filter,
        entropy_start=entropy_start,
        questions_asked=0,
        correct_answers=0,
        results=[],
    )

    # Store num_questions in results as metadata (simple approach)
    session.results = [{"meta": True, "num_questions": req.num_questions}]
    db.add(session)
    db.commit()
    db.refresh(session)

    # Select first question
    next_term = engine.select_next_term(
        category_filter=req.category_filter,
        difficulty_filter=req.difficulty_filter,
    )

    if not next_term:
        raise HTTPException(status_code=404,
                            detail="No terms available for the selected filters.")

    eig = engine.expected_information_gain(next_term["id"])
    mcq = generate_mcq(next_term, MEDICAL_TERMS)

    return {
        "session_id": session.id,
        "mode": req.mode,
        "num_questions": req.num_questions,
        "entropy_start": round(entropy_start, 4),
        "category_mastery": engine.get_category_mastery(),
        "first_question": {
            "term_id": next_term["id"],
            "term_name": next_term["term"],
            "category": next_term["category"],
            "difficulty": next_term["difficulty"],
            "root": next_term.get("root", ""),
            "mnemonic": next_term.get("mnemonic", ""),
            "question": mcq["question"],
            "options": mcq["options"],
            "correct": mcq["correct"],
            "eig_score": round(eig, 4),
            "entropy_before": round(entropy_start, 4),
        }
    }


@router.post("/answer")
def submit_answer(req: SubmitAnswerRequest,
                  current_user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    """Submit an answer, get feedback + next question."""
    session = db.query(QuizSession).filter(
        QuizSession.id == req.session_id,
        QuizSession.user_id == current_user.id,
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.ended_at:
        raise HTTPException(status_code=400, detail="Session already ended")

    term = TERM_BY_ID.get(req.term_id)
    if not term:
        raise HTTPException(status_code=404, detail="Term not found")

    engine = build_engine_from_user(current_user.id, db)
    eig_score = engine.expected_information_gain(req.term_id)

    # Check correctness
    is_correct = req.selected_answer.strip().lower() == term["definition"].strip().lower()

    # Record in engine
    result = engine.record_answer(req.term_id, is_correct, req.response_time_ms)

    # Save answer to DB
    answer = Answer(
        id=str(uuid.uuid4()),
        session_id=session.id,
        user_id=current_user.id,
        term_id=req.term_id,
        term_name=term["term"],
        category=term["category"],
        difficulty=term["difficulty"],
        is_correct=is_correct,
        selected_option=req.selected_answer[:500],
        correct_option=term["definition"],
        response_time_ms=req.response_time_ms,
        entropy_before=result["entropy_before"],
        entropy_after=result["entropy_after"],
        information_gained=result["information_gained"],
        belief_after=result["belief_after"],
        eig_score=eig_score,
    )
    db.add(answer)

    # Update session stats
    session.questions_asked += 1
    if is_correct:
        session.correct_answers += 1
    session.total_information_gained += result["information_gained"]

    results_list = [r for r in (session.results or []) if not r.get("meta")]
    meta = [r for r in (session.results or []) if r.get("meta")]
    num_questions = meta[0]["num_questions"] if meta else 10

    results_list.append({
        "term_id": req.term_id,
        "correct": is_correct,
        "entropy_before": result["entropy_before"],
        "entropy_after": result["entropy_after"],
    })
    session.results = meta + results_list

    # Check if session is done
    session_complete = session.questions_asked >= num_questions

    if session_complete:
        session.ended_at = datetime.utcnow()
        session.entropy_end = engine.get_knowledge_entropy()

        # Update user cumulative progress
        progress = db.query(UserProgress).filter(
            UserProgress.user_id == current_user.id
        ).first()
        if progress:
            progress.total_questions += session.questions_asked
            progress.total_correct += session.correct_answers
            progress.total_sessions += 1
            progress.total_time_ms += sum(
                a.response_time_ms for a in session.answers
            ) + req.response_time_ms

            entropy_hist = progress.entropy_history or []
            entropy_hist.append({
                "session_id": session.id,
                "entropy": session.entropy_end,
                "timestamp": datetime.utcnow().isoformat(),
            })
            progress.entropy_history = entropy_hist
            progress.category_mastery = engine.get_category_mastery()

        save_engine_to_user(engine, current_user.id, db)
        db.commit()

        return {
            "feedback": _build_feedback(term, is_correct, result),
            "session_complete": True,
            "session_summary": _build_summary(session, engine, current_user.id, db),
            "next_question": None,
        }

    # Persist updated beliefs between questions
    save_engine_to_user(engine, current_user.id, db)
    db.commit()

    # Select next question
    next_term = engine.select_next_term(
        category_filter=session.category_filter,
        difficulty_filter=session.difficulty_filter,
    )

    if not next_term:
        session.ended_at = datetime.utcnow()
        db.commit()
        return {
            "feedback": _build_feedback(term, is_correct, result),
            "session_complete": True,
            "session_summary": _build_summary(session, engine, current_user.id, db),
            "next_question": None,
        }

    next_eig = engine.expected_information_gain(next_term["id"])
    next_mcq = generate_mcq(next_term, MEDICAL_TERMS)

    return {
        "feedback": _build_feedback(term, is_correct, result),
        "session_complete": False,
        "next_question": {
            "term_id": next_term["id"],
            "term_name": next_term["term"],
            "category": next_term["category"],
            "difficulty": next_term["difficulty"],
            "root": next_term.get("root", ""),
            "mnemonic": next_term.get("mnemonic", ""),
            "question": next_mcq["question"],
            "options": next_mcq["options"],
            "correct": next_mcq["correct"],
            "eig_score": round(next_eig, 4),
            "entropy_before": round(result["entropy_after"], 4),
            "questions_asked": session.questions_asked,
            "questions_total": num_questions,
        }
    }


@router.get("/categories")
def get_categories():
    return {"categories": CATEGORIES, "total": len(CATEGORIES)}


@router.get("/terms/sample")
def get_sample_terms():
    sample = random.sample(MEDICAL_TERMS, min(5, len(MEDICAL_TERMS)))
    return [{"id": t["id"], "term": t["term"], "category": t["category"],
             "difficulty": t["difficulty"]} for t in sample]


def _build_feedback(term: dict, is_correct: bool, result: dict) -> dict:
    return {
        "is_correct": is_correct,
        "correct_answer": term["definition"],
        "term": term["term"],
        "category": term["category"],
        "root": term.get("root", ""),
        "mnemonic": term.get("mnemonic", ""),
        "synonyms": term.get("synonyms", []),
        "related_terms": term.get("related", []),
        "entropy_delta": result["information_gained"],
        "belief_after": result["belief_after"],
        "explanation": (
            f"{'✓ Correct!' if is_correct else '✗ Incorrect.'} "
            f"The system's confidence in your knowledge of '{term['term']}' "
            f"is now {result['belief_after']*100:.0f}%."
        ),
    }


def _build_summary(session: QuizSession, engine: EntropyEngine,
                   user_id: str, db: Session) -> dict:
    accuracy = (session.correct_answers / session.questions_asked * 100
                if session.questions_asked else 0)
    return {
        "session_id": session.id,
        "questions_asked": session.questions_asked,
        "correct_answers": session.correct_answers,
        "accuracy": round(accuracy, 1),
        "entropy_start": round(session.entropy_start, 4),
        "entropy_end": round(session.entropy_end or engine.get_knowledge_entropy(), 4),
        "total_information_gained": round(session.total_information_gained, 4),
        "overall_mastery": engine.overall_mastery(),
        "weakest_areas": engine.get_weakest_areas(3),
        "strongest_areas": engine.get_strongest_areas(3),
        "category_mastery": engine.get_category_mastery(),
    }
