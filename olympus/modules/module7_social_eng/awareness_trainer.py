"""Awareness training simulator — educational tool for identifying phishing cues."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from olympus.modules.module7_social_eng.synthetic_dataset import SyntheticEmail


@dataclass
class AwarenessChallenge:
    email: SyntheticEmail
    cues: list[str]                     # visible phishing indicators
    correct_answer: int                 # 1=phishing, 0=legit
    difficulty: str                     # "easy", "medium", "hard"


@dataclass
class AwarenessScore:
    total: int = 0
    correct: int = 0
    false_positives: int = 0            # legit flagged as phishing
    false_negatives: int = 0            # phishing missed
    per_technique: dict[str, list[bool]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)

    @property
    def level(self) -> str:
        acc = self.accuracy
        if acc >= 0.90: return "Expert"
        if acc >= 0.75: return "Advanced"
        if acc >= 0.60: return "Intermediate"
        return "Beginner"


_CUE_EXTRACTORS = [
    (lambda e: e.sender != e.reply_to,
     "Reply-To address differs from sender — possible spoofing"),
    (lambda e: any(tld in e.sender_domain for tld in [".tk", ".ml", ".xyz", ".ga"]),
     "Sender uses suspicious TLD (.tk, .ml, .xyz)"),
    (lambda e: any("http://" in u for u in e.urls),
     "Links use HTTP instead of HTTPS"),
    (lambda e: any(brand in e.sender_domain for brand in
                   ["paypa", "amaz", "app1", "micros0"]),
     "Domain is a misspelling of a well-known brand"),
    (lambda e: "urgent" in (e.subject + e.body).lower() or
               "immediately" in (e.subject + e.body).lower(),
     "Creates false urgency ('URGENT', 'immediately')"),
    (lambda e: "suspend" in (e.subject + e.body).lower() or
               "compromised" in (e.subject + e.body).lower(),
     "Uses fear tactics (account suspension, compromise)"),
    (lambda e: any(kw in e.body.lower() for kw in
                   ["click here", "click below", "verify now"]),
     "Suspicious call-to-action ('Click here', 'Verify now')"),
    (lambda e: "congratulations" in (e.subject + e.body).lower() or
               "prize" in (e.subject + e.body).lower(),
     "Unsolicited prize or reward offer"),
    (lambda e: len(e.urls) > 1,
     f"Multiple suspicious links in email"),
    (lambda e: e.subject.isupper(),
     "Subject line is all uppercase — pressure tactic"),
]


def extract_cues(email: SyntheticEmail) -> list[str]:
    return [desc for check, desc in _CUE_EXTRACTORS if check(email)]


def build_challenge(email: SyntheticEmail) -> AwarenessChallenge:
    cues = extract_cues(email) if email.label == 1 else []
    n_cues = len(cues)
    difficulty = "easy" if n_cues >= 3 else "medium" if n_cues >= 1 else "hard"
    return AwarenessChallenge(
        email=email, cues=cues,
        correct_answer=email.label,
        difficulty=difficulty,
    )


class AwarenessTrainer:
    """Interactive phishing awareness training simulator."""

    def __init__(self, emails: list[SyntheticEmail], seed: int = 42) -> None:
        self._emails = emails
        self._rng = random.Random(seed)
        self._score = AwarenessScore()

    def next_challenge(self) -> AwarenessChallenge:
        email = self._rng.choice(self._emails)
        return build_challenge(email)

    def evaluate_response(
        self, challenge: AwarenessChallenge, user_answer: int
    ) -> dict:
        correct = user_answer == challenge.correct_answer
        self._score.total += 1
        if correct:
            self._score.correct += 1
        elif challenge.correct_answer == 0 and user_answer == 1:
            self._score.false_positives += 1
        elif challenge.correct_answer == 1 and user_answer == 0:
            self._score.false_negatives += 1

        tech = challenge.email.technique
        if tech not in self._score.per_technique:
            self._score.per_technique[tech] = []
        self._score.per_technique[tech].append(correct)

        return {
            "correct": correct,
            "correct_answer": "phishing" if challenge.correct_answer == 1 else "legitimate",
            "cues_missed": challenge.cues if not correct and challenge.correct_answer == 1 else [],
            "explanation": self._explain(challenge, correct),
            "current_score": self._score.accuracy,
            "level": self._score.level,
        }

    def _explain(self, ch: AwarenessChallenge, correct: bool) -> str:
        if ch.correct_answer == 1:
            cue_str = "\n  • ".join(ch.cues) if ch.cues else "subtle manipulation"
            return (
                f"This was a {'correctly identified ' if correct else 'MISSED '}phishing email.\n"
                f"Technique: {ch.email.technique}\n"
                f"Red flags:\n  • {cue_str}"
            )
        return (
            f"This was a {'correctly identified ' if correct else 'incorrectly flagged '}legitimate email.\n"
            "No phishing indicators were present."
        )

    def get_score(self) -> AwarenessScore:
        return self._score

    def generate_report(self) -> str:
        s = self._score
        lines = [
            "OLYMPUS Phishing Awareness Training Report",
            "=" * 45,
            f"Total challenges: {s.total}",
            f"Correct:          {s.correct} ({s.accuracy:.0%})",
            f"False positives:  {s.false_positives}",
            f"False negatives:  {s.false_negatives}",
            f"Awareness level:  {s.level}",
            "",
            "Per-technique accuracy:",
        ]
        for tech, results in s.per_technique.items():
            acc = sum(results) / len(results)
            lines.append(f"  {tech:20s}: {acc:.0%} ({len(results)} challenges)")
        return "\n".join(lines)
