"""
MedTermQuest — Main FastAPI Application
Greedy Entropy Maximization for Medical Lexical Knowledge Assessment
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.database import create_tables
from routers import auth, quiz, progress
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

app = FastAPI(
    title="MedTermQuest API",
    description=(
        "Adaptive medical terminology learning system using "
        "Greedy Entropy Maximization for Lexical Constraint Satisfaction. "
        "Each question is selected to maximize Shannon information gain "
        "about the user's knowledge state."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create DB tables on startup
create_tables()

# Register routers
app.include_router(auth.router)
app.include_router(quiz.router)
app.include_router(progress.router)


@app.get("/")
def root():
    return {
        "service": "MedTermQuest API",
        "version": "1.0.0",
        "algorithm": "Greedy Entropy Maximization for Lexical Constraint Satisfaction",
        "domain": "Medical NLP & Cognitive Computing",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    from data.medical_terms import MEDICAL_TERMS, CATEGORIES
    return {
        "status": "healthy",
        "terms_loaded": len(MEDICAL_TERMS),
        "categories": len(CATEGORIES),
    }