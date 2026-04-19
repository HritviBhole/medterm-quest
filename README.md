# MedTermQuest

**Greedy Entropy Maximization for Medical Lexical Constraint Satisfaction**
*Cognitive Computing and Natural Language Processing*

MedTermQuest models medical terminology learning as a lexical constraint satisfaction problem. The system infers which terms a user knows based on their answers, maintains a Bayesian belief distribution over all terms, and updates it after each response. Using Shannon entropy, it measures uncertainty in the user's knowledge state and applies greedy entropy maximization to select the next question with the highest expected information gain. Each question reduces uncertainty efficiently, allowing the system to converge quickly to an accurate representation of the user's knowledge.

---

## What is this?

MedTermQuest is an adaptive medical terminology learning system. Instead of quizzing randomly, it maintains a **Bayesian belief distribution** over knowledge of 47 medical terms across 13 domains, then uses **Shannon entropy maximization** to greedily pick the next question that will reveal the most about the user's knowledge state.

The system treats medical knowledge as a probability distribution and applies information theory to learn the user as efficiently as possible, converging on the true knowledge state in the fewest possible questions.

---

## File Structure

```
medterm-quest/
|
|-- README.md                  <- This file
|-- .env.example               <- Copy to .env and configure
|-- .gitignore
|-- docker-compose.yml         <- One-command full-stack launch
|-- nginx.conf                 <- Nginx reverse proxy config
|
|-- backend/                   <- Python FastAPI backend
|   |-- Dockerfile
|   |-- requirements.txt
|   |-- main.py                <- FastAPI app entry point
|   |
|   |-- core/
|   |   |-- __init__.py
|   |   `-- entropy_engine.py  <- THE ALGORITHM (read this first)
|   |
|   |-- data/
|   |   |-- __init__.py
|   |   `-- medical_terms.py   <- 47 terms, 13 categories, mnemonics
|   |
|   |-- models/
|   |   |-- __init__.py
|   |   `-- database.py        <- SQLAlchemy models (User, Session, Answer...)
|   |
|   `-- routers/
|       |-- __init__.py
|       |-- auth.py            <- JWT login/signup
|       |-- quiz.py            <- Adaptive session management
|       `-- progress.py        <- Analytics, history, leaderboard
|
`-- frontend/
    |-- package.json
    `-- index.html             <- Complete SPA (no build step needed)
```

---

## Core Algorithm - Greedy Entropy Maximization

The heart of the project is `backend/core/entropy_engine.py`.

### The Problem (Lexical Constraint Satisfaction)

Given a medical lexicon L of n terms, determine which subset S is known by a user, subject to the constraint that each observed answer is consistent with their true knowledge state.

This is a **constraint satisfaction problem** over a probabilistic belief space.

### The Solution (Greedy Entropy Maximization)

Maintain a belief vector **B in [0,1]^n** where `B[i] = P(user knows term i)`.

**Initialization** - difficulty-based priors:

```
P(knows | difficulty=1) = 0.80   # basic terms
P(knows | difficulty=2) = 0.60
P(knows | difficulty=3) = 0.40
P(knows | difficulty=4) = 0.25   # expert terms
```

**Greedy selection** - pick the term maximizing Expected Information Gain:

```
EIG(term) = H(B) - E[H(B | outcome)]
          = H(B) - [P(correct)*H(B|correct) + P(wrong)*H(B|wrong)]

Next question = argmax_{term not in asked} EIG(term)
```

where `H(B) = -sum p*log2(p) + (1-p)*log2(1-p)` is average binary entropy.

**Bayesian update** - after observing an answer:

```
P(knows | correct) proportional to P(correct | knows) * P(knows)
                   = 0.92 * P(knows)

P(knows | wrong) proportional to P(wrong | knows) * P(knows)
                 = 0.08 * P(knows)
```

**Category correlation propagation** - knowing one cardiovascular term raises the prior for related cardiovascular terms (correlation coefficient = 0.12), because domain knowledge is correlated.

### Why greedy?

Optimal lookahead over the full question graph is NP-hard. Greedy selection is O(n) per step and provably converges; each question strictly reduces expected entropy over the belief distribution.

---

## Medical Domains

| Category | Terms | Difficulty Range |
| --- | --- | --- |
| Cardiovascular | 5 | 1-3 |
| Neurology | 4 | 2-3 |
| Respiratory | 4 | 1-3 |
| Endocrinology | 3 | 1-2 |
| Pharmacology | 3 | 2-3 |
| Pathology | 3 | 1-2 |
| Microbiology | 2 | 2-3 |
| Psychiatry | 1 | 2 |
| Immunology | 2 | 2 |
| Hematology | 1 | 1 |
| Gastroenterology | 1 | 2 |
| Nephrology | 1 | 2 |
| Anatomy | 1 | 3 |

Each term includes definition, etymology/root, mnemonic, related terms, and diagnostic constraints.

---

## API Reference

All endpoints are documented at `http://localhost:8000/docs` (Swagger UI).

### Auth

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/auth/signup` | Create account, returns JWT |
| POST | `/auth/login` | Login, returns JWT |
| GET | `/auth/me` | Get current user profile |
| POST | `/auth/logout` | Logout (client deletes token) |

### Quiz

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/quiz/start` | Start adaptive session, first question |
| POST | `/quiz/answer` | Submit answer, feedback and next question |
| GET | `/quiz/categories` | List all 13 categories |
| GET | `/quiz/terms/sample` | Random sample of 5 terms |

**Start session request:**

```json
{
  "mode": "adaptive",
  "category_filter": null,
  "difficulty_filter": null,
  "num_questions": 10
}
```

**Answer response (key fields):**

```json
{
  "feedback": {
    "is_correct": true,
    "correct_answer": "Heart rate exceeding 100 bpm...",
    "mnemonic": "TACHY = Too Aggressively...",
    "entropy_delta": 0.0143,
    "belief_after": 0.927
  },
  "next_question": {
    "term_name": "Myocardial Infarction",
    "eig_score": 0.0098,
    "entropy_before": 0.9056,
    "options": ["...", "...", "...", "..."]
  }
}
```

### Progress

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/progress/overview` | Full dashboard, mastery, trends, weak/strong areas |
| GET | `/progress/sessions` | Paginated session history |
| GET | `/progress/sessions/{id}` | Single session with per-answer breakdown |
| GET | `/progress/leaderboard` | All users ranked by mastery |
| GET | `/progress/knowledge-map` | Per-term belief state for visualization |

---

## Running the Project

### Option A: Docker (recommended)

```bash
git clone <repo>
cd medterm-quest

# Configure environment
cp .env.example .env
# Edit .env if needed (defaults work for local dev)

# Start everything
docker-compose up --build

# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### Option B: Manual (development)

```bash
# Terminal 1 - Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
python3 -m http.server 3000
# Open http://localhost:3000
```

### Option C: Frontend Only (no backend)

Open `frontend/index.html` directly in a browser. The frontend works completely standalone using browser `localStorage`. The full entropy algorithm runs client-side in JavaScript, so no server or installation is needed.

---

## Database Schema

```
users
  id, email, username, hashed_password, full_name, role, created_at, last_login

user_progress                        <- one row per user, updated after every session
  user_id, beliefs (JSON),           <- Bayesian belief vector B[term_id] = float
  tested_ids (JSON),                 <- set of term IDs the user has answered
  total_questions, total_correct,
  entropy_history (JSON),            <- list of {session_id, entropy, timestamp}
  category_mastery (JSON)            <- {category: {mastery_score, entropy, confidence}}

quiz_sessions
  id, user_id, mode, category_filter, difficulty_filter,
  questions_asked, correct_answers,
  entropy_start, entropy_end,        <- knowledge entropy before/after session
  total_information_gained           <- sum of EIG across all questions

answers
  session_id, term_id, term_name, category, difficulty,
  is_correct, selected_option, correct_option,
  response_time_ms,
  entropy_before, entropy_after,     <- global belief entropy before/after this answer
  information_gained,                <- entropy_before - entropy_after
  belief_after,                      <- P(knows this term) after update
  eig_score                          <- EIG score that caused this term to be selected
```

---

## Switching to PostgreSQL (production)

In `.env`, change:

```
DATABASE_URL=postgresql://user:password@localhost:5432/medterm_db
```

Create the database:

```bash
psql -U postgres -c "CREATE DATABASE medterm_db;"
```

The ORM handles all schema creation automatically on first run.

---

## Key Design Decisions

**Why not a React app?** The frontend is a single `index.html` with no build step. It has zero dependencies and opens in any browser. The full entropy algorithm is implemented in both Python (backend) and JavaScript (frontend), so the app works completely offline.

**Why SQLite by default?** Zero-config for development and demos. The SQLAlchemy ORM makes switching to PostgreSQL a one-line change in `.env`.

**Why not a larger question bank?** Each term requires a definition, etymology, mnemonic, and related terms. The entropy algorithm extracts maximum information from each answer, so 47 high-quality terms outperform 500 low-quality ones.

**Why greedy and not optimal?** Optimal adaptive testing (via POMDP) is NP-hard for n greater than about 20 terms. Greedy EIG is O(n) per step, produces near-optimal results in practice, and is theoretically sound and convergent.

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| Backend language | Python 3.11 |
| API framework | FastAPI 0.111 |
| ORM | SQLAlchemy 2.0 |
| Auth | JWT (python-jose) + bcrypt |
| Database (dev) | SQLite |
| Database (prod) | PostgreSQL |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
| Containerization | Docker + Docker Compose |
| Reverse proxy | Nginx |