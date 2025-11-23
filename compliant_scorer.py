# -*- coding: utf-8 -*-
"""
CSV‑Compliant Automated Scorer (real semantic scoring)

Implements the rubric using sentence‑transformers for semantic similarity and
proportional scoring for Flow & Tone sub‑metrics.
"""

import re
from typing import Dict, Any, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

# ---------------------------------------------------------------------------
# Global model – loaded once
# ---------------------------------------------------------------------------
try:
    MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    MODEL_READY = True
except Exception as e:
    print("⚠️ Model could not be loaded:", e)
    MODEL_READY = False

# ---------------------------------------------------------------------------
# CSV‑style rubric (Generalized for all ages)
# ---------------------------------------------------------------------------
CSV_RUBRIC: Dict[str, Dict[str, Any]] = {
    "Content & Structure": {
        "weight": 40,
        "sub_criteria": {
            "Salutation Level": {
                "weight": 5,
                "keywords": ["hello", "thank you for this opportunity", "good morning"]
            },
            "Keyword Presence": {
                "weight": 30,
                "keywords": ["name", "age", "education", "profession", "family", "hobbies", "goals", "challenge", "unique point"]
            },
            "Flow & Tone (Composite)": {
                "weight": 5,
                "sub_metrics": {
                    "A. Cohesion Markers": {
                        "weight": 2.0,
                        "markers": ["first", "second", "third", "finally", "to elaborate", "however"]
                    },
                    "B. Sentiment Polarity": {
                        "weight": 2.0,
                        "positive_keywords": ["strong emphasis", "keen passion", "successfully", "long‑term goal"],
                        "negative_keywords": ["challenge", "difficulty", "foresee"]
                    },
                    "C. Emotional Intensity": {
                        "weight": 1.0,
                        "intensity_words": ["biggest", "passion", "rigorous", "aim", "master"]
                    },
                },
            },
        },
    },
    "Speech Rate": {
        "weight": 10,
        "bands": {
            (141, 999): 6,
            (111, 140): 10,
            (81, 110): 6,
            (0, 80): 2,
        },
    },
    "Language & Grammar": {
        "weight": 20,
        "sub_criteria": {
            "Grammar Error Score": {
                "weight": 10,
                "error_keywords": ["hish", "gow", "dont", "wanna"],
                "max_error_per_100_words": 10,
            },
            "Vocabulary Richness (TTR)": {
                "weight": 10,
                "bands": {
                    (0.90, 1.0): 10,
                    (0.70, 0.89): 8,
                    (0.50, 0.69): 6,
                    (0.30, 0.49): 4,
                    (0.00, 0.29): 2,
                },
            },
        },
    },
    "Clarity": {
        "weight": 10,
        "filler_words": ["um", "uh", "like", "you know", "so", "actually", "basically", "right", "i mean", "well", "kinda", "sort of", "okay", "hmm", "ah"],
        "bands": {
            (0.0, 1.0): 10,
            (1.1, 3.0): 8,
            (3.1, 5.0): 6,
            (5.1, 100.0): 2,
        },
    },
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def split_text_into_sentences(text: str) -> List[str]:
    """Very simple sentence splitter – good enough for our use‑case."""
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Return cosine similarity between two vectors (0‑1 range)."""
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b)))

def band_score(value: float, bands: Dict[Tuple[float, float], float]) -> float:
    """Select the score whose interval contains *value* (inclusive)."""
    for (low, high), score in sorted(bands.items(), key=lambda x: x[0][0], reverse=True):
        if low <= value <= high:
            return score
    return 0.0

# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_quantitative_metrics(transcript: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    wc = transcript["metadata"]["word_count"]
    # Fixed 52 s duration for demo purposes
    duration_minutes = 52 / 60
    wpm = round(wc / duration_minutes) if duration_minutes else 0
    band = CSV_RUBRIC["Speech Rate"]["bands"]
    score = band_score(wpm, band)
    return score, {"Speech Rate": {"score": score, "feedback": f"WPM={wpm} → band {score}"}}

def score_language_grammar(transcript: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    wc = transcript["metadata"]["word_count"]
    text = transcript["text"].lower()
    results = {}
    total = 0.0
    # Grammar errors (mock)
    gram = CSV_RUBRIC["Language & Grammar"]["sub_criteria"]["Grammar Error Score"]
    errors = sum(1 for kw in gram["error_keywords"] if kw in text)
    errors_per_100 = (errors / wc) * 100 if wc else 0.0
    norm_factor = 1 - min(errors_per_100 / gram["max_error_per_100_words"], 1)
    gram_score = round(norm_factor * gram["weight"], 2)
    results["Grammar Error Score"] = {"score": gram_score, "feedback": f"{errors} errors → {errors_per_100:.2f}%/100w"}
    total += gram_score
    # Vocabulary richness (TTR)
    ttr_data = CSV_RUBRIC["Language & Grammar"]["sub_criteria"]["Vocabulary Richness (TTR)"]
    tokens = re.findall(r"\b\w+\b", text)
    distinct = len(set(tokens))
    ttr = round(distinct / len(tokens), 2) if tokens else 0.0
    ttr_score = band_score(ttr, ttr_data["bands"])
    results["TTR Score"] = {"score": ttr_score, "feedback": f"TTR={ttr:.2f}"}
    total += ttr_score
    return total, results

def score_clarity(transcript: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    wc = transcript["metadata"]["word_count"]
    text = transcript["text"].lower()
    filler_words = CSV_RUBRIC["Clarity"]["filler_words"]
    filler_cnt = sum(len(re.findall(r"\b" + re.escape(f) + r"\b", text)) for f in filler_words)
    filler_rate = round((filler_cnt / wc) * 100, 2) if wc else 0.0
    score = band_score(filler_rate, CSV_RUBRIC["Clarity"]["bands"])
    return score, {"Clarity Score": {"score": score, "feedback": f"{filler_cnt} fillers → {filler_rate:.2f}%"}}

def semantic_score(query: str, sentences: List[str]) -> float:
    """Return the highest cosine similarity between *query* and any sentence.
    Uses the global `MODEL`.
    """
    if not MODEL_READY:
        return 0.0
    query_emb = MODEL.encode([query], convert_to_tensor=True)[0]
    sent_emb = MODEL.encode(sentences, convert_to_tensor=True)
    q = query_emb.cpu().numpy()
    s = sent_emb.cpu().numpy()
    q_norm = q / np.linalg.norm(q) if np.linalg.norm(q) else q
    s_norm = s / np.linalg.norm(s, axis=1, keepdims=True)
    sims = np.dot(s_norm, q_norm)
    return float(np.max(sims))

def score_content_structure(transcript: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Score Content & Structure using semantic similarity and proportional Flow/Tone."""
    sentences = split_text_into_sentences(transcript["text"]) or [transcript["text"]]
    results = {}
    total = 0.0
    sub = CSV_RUBRIC["Content & Structure"]["sub_criteria"]
    # Salutation Level
    sal_query = " ".join(sub["Salutation Level"]["keywords"])
    sal_score = semantic_score(sal_query, sentences) * sub["Salutation Level"]["weight"]
    results["Salutation Level"] = {"score": round(sal_score, 2), "feedback": f"Similarity={sal_score/sub['Salutation Level']['weight']:.2f}"}
    total += sal_score
    # Keyword Presence
    kw_query = " ".join(sub["Keyword Presence"]["keywords"])
    kw_score = semantic_score(kw_query, sentences) * sub["Keyword Presence"]["weight"]
    results["Keyword Presence"] = {"score": round(kw_score, 2), "feedback": f"Similarity={kw_score/sub['Keyword Presence']['weight']:.2f}"}
    total += kw_score
    # Flow & Tone (proportional sub‑metrics)
    flow = sub["Flow & Tone (Composite)"]["sub_metrics"]
    flow_results = {}
    flow_total = 0.0
    # A. Cohesion Markers
    markers = flow["A. Cohesion Markers"]["markers"]
    found = sum(1 for m in markers if m.lower() in transcript["text"].lower())
    coh_score = (found / len(markers)) * flow["A. Cohesion Markers"]["weight"]
    flow_results["A. Cohesion Markers"] = {"score": round(coh_score, 2), "feedback": f"{found}/{len(markers)} markers"}
    flow_total += coh_score
    # B. Sentiment Polarity
    pos = sum(1 for w in flow["B. Sentiment Polarity"]["positive_keywords"] if w.lower() in transcript["text"].lower())
    neg = sum(1 for w in flow["B. Sentiment Polarity"]["negative_keywords"] if w.lower() in transcript["text"].lower())
    polarity_raw = max(0, pos - neg)
    polarity_score = (polarity_raw / max(1, len(flow["B. Sentiment Polarity"]["positive_keywords"]))) * flow["B. Sentiment Polarity"]["weight"]
    flow_results["B. Sentiment Polarity"] = {"score": round(polarity_score, 2), "feedback": f"+{pos}/-{neg}"}
    flow_total += polarity_score
    # C. Emotional Intensity
    intensity_words = flow["C. Emotional Intensity"]["intensity_words"]
    intensity_found = sum(1 for w in intensity_words if w.lower() in transcript["text"].lower())
    intensity_score = (intensity_found / len(intensity_words)) * flow["C. Emotional Intensity"]["weight"]
    flow_results["C. Emotional Intensity"] = {"score": round(intensity_score, 2), "feedback": f"{intensity_found}/{len(intensity_words)}"}
    flow_total += intensity_score
    results["Flow & Tone (Composite)"] = {"score": round(flow_total, 2), "details": flow_results}
    total += flow_total
    return total, results

def evaluate(transcript: Dict[str, Any]) -> Tuple[float, str]:
    """Run full evaluation and return normalized score plus a report string."""
    # Quantitative parts
    rate_score, rate_res = score_quantitative_metrics(transcript)
    lang_score, lang_res = score_language_grammar(transcript)
    clarity_score, clarity_res = score_clarity(transcript)
    # Content & Structure
    content_score, content_res = score_content_structure(transcript)
    # Aggregate (raw out of 80)
    raw_total = rate_score + lang_score + clarity_score + content_score
    final_score = round((raw_total / 80) * 100, 2)
    # Report
    report = []
    report.append("--- FINAL SCORING JUSTIFICATION (CSV‑COMPLIANT) ---\n")
    report.append(f"Overall (normalised): {final_score}/100 (raw {raw_total:.2f}/80)\n")
    report.append("\n### 📈 Content & Structure (40 pts)\n")
    for k, v in content_res.items():
        if k == "Flow & Tone (Composite)":
            report.append(f"- {k}: {v['score']:.2f} / {CSV_RUBRIC['Content & Structure']['sub_criteria'][k]['weight']}\n")
            for subk, subv in v["details"].items():
                report.append(f"  * {subk}: {subv['score']:.2f} ({subv['feedback']})\n")
        else:
            report.append(f"- {k}: {v['score']:.2f} / {CSV_RUBRIC['Content & Structure']['sub_criteria'][k]['weight']} ({v['feedback']})\n")
    report.append("\n### 🗣 Speech Rate (10 pts)\n")
    report.append(f"- {rate_res['Speech Rate']['score']:.2f} / 10 ({rate_res['Speech Rate']['feedback']})\n")
    report.append("\n### 📝 Language & Grammar (20 pts)\n")
    report.append(f"- Grammar Error Score: {lang_res['Grammar Error Score']['score']:.2f} / 10 ({lang_res['Grammar Error Score']['feedback']})\n")
    report.append(f"- Vocabulary Richness (TTR): {lang_res['TTR Score']['score']:.2f} / 10 ({lang_res['TTR Score']['feedback']})\n")
    report.append("\n### 🎯 Clarity (10 pts)\n")
    report.append(f"- {clarity_res['Clarity Score']['score']:.2f} / 10 ({clarity_res['Clarity Score']['feedback']})\n")
    return final_score, "\n".join(report)

# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    example_text = """Hello everyone, and thank you for this opportunity. I am Alex Sharma, a software engineer with over 5 years of experience in full-stack development. I currently work at TechSolutions Inc., where I lead a team of developers. My background includes a Master's degree in Computer Science.

To elaborate on my interests, I have a keen passion for open-source projects and spend my weekends contributing to community repositories. My professional goals are distinct: First, I aim to architect scalable solutions for enterprise clients. Second, I plan to successfully mentor junior developers to help them grow. Third, my long-term goal is to become a CTO.

However, I foresee a challenge in balancing technical leadership with hands-on coding. Finally, I will close by saying thank you for your time."""
    transcript = {
        "text": example_text,
        "metadata": {"word_count": len(re.findall(r"\b\w+\b", example_text))},
    }
    score, report = evaluate(transcript)
    print(report)
    print("=" * 60)
    print(f"FINAL SCORE: {score}/100")
    print("=" * 60)
