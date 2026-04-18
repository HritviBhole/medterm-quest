"""
MedTermQuest — Greedy Entropy Maximization Engine
=========================================================
Theme: Greedy Entropy Maximization for Lexical Constraint Satisfaction
Subject: Cognitive Computing & Natural Language Processing

This module implements the core NLP algorithm that treats a user's medical
knowledge as a probability distribution, and greedily selects the next term
to test that maximizes information gain about the user's knowledge state.

Shannon Entropy: H(X) = -Σ p(xᵢ) · log₂(p(xᵢ))

The system maintains a belief vector B ∈ [0,1]ⁿ where B[i] represents
P(user knows term i). After each answer, Bayes' rule updates the belief:
  P(knows|correct) ∝ P(correct|knows) · P(knows)
  P(knows|wrong) ∝ P(wrong|knows) · P(knows)

Greedy selection: pick the term whose outcome will most reduce uncertainty
across the entire knowledge graph (not just the individual term).
"""

import math
from typing import List, Dict, Tuple, Optional
from data.medical_terms import MEDICAL_TERMS, TERMS_BY_CATEGORY, CATEGORIES


# ── Constants ────────────────────────────────────────────────────────────────
# Likelihood of correct answer given true knowledge state
P_CORRECT_IF_KNOWS = 0.92   # P(correct | user knows term)
P_CORRECT_IF_UNKNOWN = 0.18  # P(correct | user doesn't know) — random guess on MCQ

# Difficulty-based prior probability of knowing a term
DIFFICULTY_PRIOR = {1: 0.80, 2: 0.60, 3: 0.40, 4: 0.25, 5: 0.15}

# Category correlation coefficient — knowing one term raises related terms
CATEGORY_CORRELATION = 0.15
RELATED_TERM_CORRELATION = 0.08


class EntropyEngine:
    """
    Greedy entropy-based knowledge state estimator.

    Maintains a belief distribution over all medical terms and uses
    information-theoretic scoring to select the most informative next question.
    """

    def __init__(self, user_history: List[Dict] = None):
        self.beliefs = self._initialize_beliefs()
        self.asked_ids = set()
        self.session_results = []

        # Apply historical evidence if available
        if user_history:
            for record in user_history:
                self._update_belief(record["term_id"], record["correct"],
                                    learning_event=False)
                self.asked_ids.add(record["term_id"])

    def _initialize_beliefs(self) -> Dict[str, float]:
        """Initialize P(knows term) for each term using difficulty-based prior."""
        return {
            t["id"]: DIFFICULTY_PRIOR.get(t["difficulty"], 0.5)
            for t in MEDICAL_TERMS
        }

    def get_knowledge_entropy(self, term_ids: List[str] = None) -> float:
        """
        Compute Shannon entropy of the current belief distribution.
        H = -Σ p·log₂(p) + (1-p)·log₂(1-p)   [binary entropy per term]

        High entropy = high uncertainty about user's knowledge.
        Low entropy = confident model of what user knows/doesn't know.
        """
        ids = term_ids or list(self.beliefs.keys())
        total_entropy = 0.0
        for tid in ids:
            p = self.beliefs.get(tid, 0.5)
            p = max(1e-9, min(1 - 1e-9, p))
            # Binary entropy
            total_entropy += -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        return total_entropy / len(ids) if ids else 0.0

    def expected_information_gain(self, term_id: str) -> float:
        """
        Compute expected information gain from testing this term.

        EIG(term) = H(beliefs) - E[H(beliefs | outcome)]
                  = H(beliefs) - [P(correct)·H(beliefs|correct) + P(wrong)·H(beliefs|wrong)]

        This is the core greedy selection criterion: pick term with highest EIG.
        """
        p_knows = self.beliefs.get(term_id, 0.5)

        # P(correct) by law of total probability
        p_correct = (P_CORRECT_IF_KNOWS * p_knows +
                     P_CORRECT_IF_UNKNOWN * (1 - p_knows))
        p_wrong = 1 - p_correct

        # Simulate belief update for each outcome
        beliefs_if_correct = self._simulate_update(term_id, correct=True)
        beliefs_if_wrong = self._simulate_update(term_id, correct=False)

        # Compute entropy delta
        h_current = self.get_knowledge_entropy()
        h_if_correct = self._entropy_of_beliefs(beliefs_if_correct)
        h_if_wrong = self._entropy_of_beliefs(beliefs_if_wrong)

        expected_h_after = p_correct * h_if_correct + p_wrong * h_if_wrong
        return h_current - expected_h_after

    def _simulate_update(self, term_id: str, correct: bool) -> Dict[str, float]:
        """Simulate Bayesian belief update without mutating state."""
        beliefs = self.beliefs.copy()
        term = next((t for t in MEDICAL_TERMS if t["id"] == term_id), None)
        if not term:
            return beliefs

        p_knows = beliefs[term_id]

        # Bayes update for the tested term
        if correct:
            likelihood = P_CORRECT_IF_KNOWS
            likelihood_neg = P_CORRECT_IF_UNKNOWN
        else:
            likelihood = 1 - P_CORRECT_IF_KNOWS
            likelihood_neg = 1 - P_CORRECT_IF_UNKNOWN

        p_evidence = likelihood * p_knows + likelihood_neg * (1 - p_knows)
        if p_evidence > 0:
            beliefs[term_id] = (likelihood * p_knows) / p_evidence

        # Propagate correlation to category peers
        same_category = [t["id"] for t in MEDICAL_TERMS
                         if t["category"] == term["category"] and t["id"] != term_id]

        delta = CATEGORY_CORRELATION * (beliefs[term_id] - p_knows)
        for peer_id in same_category:
            beliefs[peer_id] = max(0.01, min(0.99, beliefs[peer_id] + delta * 0.5))

        return beliefs

    def _entropy_of_beliefs(self, beliefs: Dict[str, float]) -> float:
        """Compute average binary entropy of a belief dict."""
        total = 0.0
        for p in beliefs.values():
            p = max(1e-9, min(1 - 1e-9, p))
            total += -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        return total / len(beliefs) if beliefs else 0.0

    def _update_belief(self, term_id: str, correct: bool,
                       learning_event: bool = True):
        """Apply Bayesian update to beliefs after observing an answer."""
        term = next((t for t in MEDICAL_TERMS if t["id"] == term_id), None)
        if not term:
            return

        p_knows = self.beliefs[term_id]

        if correct:
            likelihood = P_CORRECT_IF_KNOWS
            likelihood_neg = P_CORRECT_IF_UNKNOWN
        else:
            likelihood = 1 - P_CORRECT_IF_KNOWS
            likelihood_neg = 1 - P_CORRECT_IF_UNKNOWN

        p_evidence = likelihood * p_knows + likelihood_neg * (1 - p_knows)
        if p_evidence > 0:
            self.beliefs[term_id] = (likelihood * p_knows) / p_evidence

        # Propagate to category peers
        same_category = [t["id"] for t in MEDICAL_TERMS
                         if t["category"] == term["category"] and t["id"] != term_id]
        delta = self.beliefs[term_id] - p_knows
        for peer_id in same_category:
            self.beliefs[peer_id] = max(0.01, min(0.99,
                                                   self.beliefs[peer_id] + delta * CATEGORY_CORRELATION))

        if learning_event:
            # Correct answers slightly boost knowledge of related terms
            if correct:
                for rel_name in term.get("related", []):
                    for t in MEDICAL_TERMS:
                        if t["term"].lower() in rel_name.lower():
                            self.beliefs[t["id"]] = min(0.99,
                                self.beliefs[t["id"]] + RELATED_TERM_CORRELATION)

    def select_next_term(self,
                         category_filter: Optional[str] = None,
                         difficulty_filter: Optional[int] = None) -> Optional[Dict]:
        """
        GREEDY SELECTION: pick the unasked term with highest Expected Information Gain.

        This is the core NLP/cognitive computing algorithm — treating knowledge
        assessment as an information-theoretic optimization problem.
        """
        candidates = [t for t in MEDICAL_TERMS
                      if t["id"] not in self.asked_ids]

        if category_filter:
            candidates = [t for t in candidates if t["category"] == category_filter]
        if difficulty_filter:
            candidates = [t for t in candidates if t["difficulty"] == difficulty_filter]

        if not candidates:
            return None

        # Score each candidate by expected information gain
        scored = []
        for term in candidates:
            eig = self.expected_information_gain(term["id"])
            # Tie-break: prefer terms with beliefs near 0.5 (maximum uncertainty)
            p = self.beliefs[term["id"]]
            uncertainty_bonus = 1.0 - abs(p - 0.5) * 2  # max at p=0.5
            scored.append((eig + uncertainty_bonus * 0.05, term))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None

    def record_answer(self, term_id: str, correct: bool,
                      response_time_ms: int = 0) -> Dict:
        """
        Record user answer, update beliefs, return feedback data.
        Returns entropy metrics for display.
        """
        old_entropy = self.get_knowledge_entropy()
        self._update_belief(term_id, correct)
        self.asked_ids.add(term_id)
        new_entropy = self.get_knowledge_entropy()

        result = {
            "term_id": term_id,
            "correct": correct,
            "response_time_ms": response_time_ms,
            "entropy_before": round(old_entropy, 4),
            "entropy_after": round(new_entropy, 4),
            "information_gained": round(old_entropy - new_entropy, 4),
            "belief_after": round(self.beliefs[term_id], 3),
        }
        self.session_results.append(result)
        return result

    def get_category_mastery(self) -> Dict[str, Dict]:
        """Return per-category knowledge summary for the progress dashboard."""
        summary = {}
        for cat, terms in TERMS_BY_CATEGORY.items():
            term_ids = [t["id"] for t in terms]
            beliefs = [self.beliefs[tid] for tid in term_ids]
            avg_belief = sum(beliefs) / len(beliefs)
            entropy = self.get_knowledge_entropy(term_ids)
            asked = [tid for tid in term_ids if tid in self.asked_ids]
            summary[cat] = {
                "mastery_score": round(avg_belief * 100, 1),
                "entropy": round(entropy, 3),
                "terms_total": len(terms),
                "terms_tested": len(asked),
                "confidence": "high" if entropy < 0.3 else "medium" if entropy < 0.6 else "low",
            }
        return summary

    def get_weakest_areas(self, n: int = 3) -> List[Dict]:
        """Return N categories/terms with lowest belief (most to learn)."""
        category_scores = []
        for cat, terms in TERMS_BY_CATEGORY.items():
            avg = sum(self.beliefs[t["id"]] for t in terms) / len(terms)
            category_scores.append({"category": cat, "mastery": round(avg * 100, 1)})
        return sorted(category_scores, key=lambda x: x["mastery"])[:n]

    def get_strongest_areas(self, n: int = 3) -> List[Dict]:
        """Return N categories with highest belief."""
        category_scores = []
        for cat, terms in TERMS_BY_CATEGORY.items():
            avg = sum(self.beliefs[t["id"]] for t in terms) / len(terms)
            category_scores.append({"category": cat, "mastery": round(avg * 100, 1)})
        return sorted(category_scores, key=lambda x: x["mastery"], reverse=True)[:n]

    def overall_mastery(self) -> float:
        """Overall knowledge mastery score 0–100."""
        if not self.beliefs:
            return 0.0
        return round(sum(self.beliefs.values()) / len(self.beliefs) * 100, 1)

    def to_snapshot(self) -> Dict:
        """Serialize current belief state for persistence."""
        return {
            "beliefs": self.beliefs,
            "asked_ids": list(self.asked_ids),
            "session_results": self.session_results,
        }
