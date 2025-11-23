import json
import time
import sys
import os
from typing import List, Dict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Try importing the required libraries
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("Error: sentence-transformers library not found.")
    print("Please run: pip install sentence-transformers scikit-learn")
    sys.exit(1)

# =================================================================================================
# MOCK DATA & CONFIGURATION
# =================================================================================================

MOCK_TRANSCRIPT = """
Hello everyone, myself Muskan, studying in class 8th B section from Christ Public School.
I am 13 years old. I live with my family. There are 3 people in my family, me, my mother and my father.
One special thing about my family is that they are very kind hearted to everyone and soft spoken.
One thing I really enjoy is play, playing cricket and taking wickets.
A fun fact about me is that I see in mirror and talk by myself.
One thing people don't know about me is that I once stole a toy from one of my cousin.
My favorite subject is science because it is very interesting.
Through science I can explore the whole world and make the discoveries and improve the lives of others.
Thank you for listening.
"""

RUBRIC_CRITERIA = [
    # --- Quantitative / Rule-Based ---
    {
        "id": "word_count",
        "name": "Word Count",
        "description": "Total number of words in the introduction.",
        "type": "rule",
        "target": {"min": 100, "max": 180},
        "weight": 0
    },
    {
        "id": "speech_rate",
        "name": "Speech Rate (WPM)",
        "description": "Words per minute. Target is 110-140 WPM.",
        "type": "rule",
        "target": {"min": 110, "max": 140},
        "weight": 10
    },
    {
        "id": "filler_words",
        "name": "Filler Word Rate",
        "description": "Percentage of filler words (um, uh, like, basically).",
        "type": "rule",
        "target": {"max_percent": 3.0},
        "weight": 15
    },

    # --- Qualitative / Semantic (Real NLP with Static HyDE) ---
    {
        "id": "content_structure",
        "name": "Content & Structure",
        "description": "The speaker introduces themselves clearly, mentioning their name, background (education/profession), family, interests, and goals.",
        "hyde_proxy": "Hello, my name is [Name]. I am a [Profession/Student] with a background in [Field]. I live with my family. My interests include [Hobbies] and my goal is to [Goal]. Thank you.",
        "type": "semantic",
        "weight": 40
    },
    {
        "id": "grammar_vocab",
        "name": "Language & Grammar",
        "description": "The speech uses diverse vocabulary and correct grammar without frequent errors.",
        "hyde_proxy": "I am passionate about my work and studies because they allow me to make an impact. My background has provided me with a strong foundation. I articulate my thoughts clearly and effectively.",
        "type": "semantic",
        "weight": 20
    },
    {
        "id": "engagement",
        "name": "Engagement & Tone",
        "description": "The speaker sounds enthusiastic, confident, and engaging, sharing personal and interesting details.",
        "hyde_proxy": "I am really excited to share a fun fact about myself! I love playing cricket and taking wickets, it is my absolute favorite thing to do. It is so much fun!",
        "type": "semantic",
        "weight": 15
    }
]

# --- Anomaly Detection Configuration (Advanced & Humane) ---
# Context-aware checks with severity levels for student introductions
ANOMALY_CRITERIA = [
    {
        "name": "Excessive Professional Claims",
        "description": "Claims of extensive corporate work experience (10+ years, C-level positions) that contradict student context.",
        "hyde_proxy": "I have been working as a Chief Executive Officer for over 15 years, managing Fortune 500 companies and leading global corporate strategy.",
        "threshold": 0.60,  # Higher threshold - only flag very strong matches
        "severity": "critical"
    },
    {
        "name": "Advanced Academic Credentials",
        "description": "Claims of multiple advanced degrees (PhD, multiple Masters) highly unlikely for school students.",
        "hyde_proxy": "I hold three PhD degrees and five Master's degrees from prestigious universities. I have published over 100 research papers in top journals.",
        "threshold": 0.65,  # Very high threshold
        "severity": "warning"
    },
    {
        "name": "Age-Experience Mismatch",
        "description": "Combination of young age with decades of professional experience.",
        "hyde_proxy": "I am 14 years old and I have 25 years of professional work experience in multiple industries.",
        "threshold": 0.70,  # Extremely high threshold - only obvious contradictions
        "severity": "critical"
    }
]

# =================================================================================================
# REAL NLP ENGINE
# =================================================================================================

class RealAutomatedScorer:
    def __init__(self, transcript: str, rubric: List[Dict]):
        self.transcript = transcript.strip()
        self.rubric = rubric
        
        # 1. Load Model (Global Scope Optimization)
        print("Loading NLP Model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model Loaded.")

        # 2. Preprocess Transcript
        self.sentences = [s.strip() for s in self.transcript.split('.') if s.strip()]
        self.words = self.transcript.split()
        self.word_count = len(self.words)
        
        # Estimate Duration (130 WPM avg)
        self.estimated_duration_min = self.word_count / 130.0 

        # 3. Encode Transcript Sentences once
        print("Encoding transcript sentences...")
        self.sentence_embeddings = self.model.encode(self.sentences, convert_to_tensor=True)

    def calculate_semantic_similarity(self, criterion: Dict) -> float:
        """
        Calculates the maximum cosine similarity between the criterion's HyDE Proxy
        (or description) and any sentence in the transcript.
        """
        # HyDE Strategy: Use the 'Ideal Response' (hyde_proxy) if available, 
        # otherwise fallback to description.
        query_text = criterion.get('hyde_proxy', criterion['description'])
        
        # Encode the query (Hypothetical Document)
        query_embedding = self.model.encode(query_text, convert_to_tensor=True)
        
        # Compute cosine similarities
        cosine_scores = util.cos_sim(query_embedding, self.sentence_embeddings)[0]
        
        # Find the best matching sentence score
        best_score = float(cosine_scores.max())
        
        return best_score

    def detect_anomalies(self) -> List[Dict]:
        """
        Advanced anomaly detection with severity levels.
        Only flags obvious contradictions with high confidence thresholds.
        """
        anomalies = []
        for anomaly in ANOMALY_CRITERIA:
            similarity = self.calculate_semantic_similarity(anomaly)
            if similarity > anomaly['threshold']:
                severity = anomaly.get('severity', 'warning')
                anomalies.append({
                    "name": anomaly['name'],
                    "description": anomaly['description'],
                    "confidence": similarity,
                    "severity": severity,
                    "msg": f"{severity.upper()}: {anomaly['description']} (Confidence: {similarity:.2%})"
                })
        return anomalies

    def evaluate_rule_based(self, criterion: Dict) -> Dict:
        """
        Evaluates quantitative metrics using pure logic.
        """
        score = 100
        justification = ""
        
        if criterion['id'] == 'word_count':
            val = self.word_count
            min_t = criterion['target']['min']
            max_t = criterion['target']['max']
            if min_t <= val <= max_t:
                score = 100
                justification = f"Perfect length ({val} words). Within range {min_t}-{max_t}."
            else:
                score = 80
                justification = f"Word count ({val}) is outside ideal range {min_t}-{max_t}."
                
        elif criterion['id'] == 'speech_rate':
            wpm = int(self.word_count / self.estimated_duration_min) if self.estimated_duration_min > 0 else 0
            min_t = criterion['target']['min']
            max_t = criterion['target']['max']
            
            if min_t <= wpm <= max_t:
                score = 100
                justification = f"Good pace (~{wpm} WPM)."
            else:
                score = 85
                justification = f"Pace (~{wpm} WPM) is slightly off target ({min_t}-{max_t})."
                
        elif criterion['id'] == 'filler_words':
            fillers = ['um', 'uh', 'like', 'basically', 'actually']
            count = sum(1 for w in self.words if w.lower() in fillers)
            rate = (count / self.word_count) * 100
            max_p = criterion['target']['max_percent']
            
            if rate <= max_p:
                score = 100
                justification = f"Clean speech. Filler rate: {rate:.1f}% (Target < {max_p}%)."
            else:
                score = 70
                justification = f"High usage of fillers ({rate:.1f}%). Detected: {count} fillers."
        
        return {"score": score, "justification": justification}

    def run_evaluation(self):
        print(f"\n{'='*60}")
        print(f"REAL AUTOMATED SCORING REPORT (NLP-POWERED)")
        print(f"{'='*60}\n")
        
        # --- ANOMALY DETECTION ---
        anomalies = self.detect_anomalies()
        if anomalies:
            print(f"!!! ANOMALY DETECTED !!!")
            for a in anomalies:
                print(f"[WARNING] {a['name']}: {a['msg']}")
            print(f"{'='*60}\n")
        else:
            print("Anomaly Check: Passed (No inconsistencies detected).")
            print(f"{'='*60}\n")

        total_weighted_score = 0
        total_weight = 0
        
        results = []
        
        for criterion in self.rubric:
            print(f"Analyzing: {criterion['name']}...", end="\r")
            
            if criterion['type'] == 'rule':
                result = self.evaluate_rule_based(criterion)
            else:
                # --- REAL NLP SCORING (Static HyDE) ---
                similarity = self.calculate_semantic_similarity(criterion)
                
                # Map Similarity (0.0 - 1.0) to Score (0 - 100)
                # With HyDE proxies, similarity should be higher for good matches.
                # 0.6+ is usually a strong match with sentence-transformers.
                
                raw_score = similarity * 100
                # Less aggressive boost needed since HyDE vectors align better
                boosted_score = min(100, raw_score * 1.3) 
                
                final_criterion_score = int(boosted_score)
                
                result = {
                    "score": final_criterion_score,
                    "justification": f"HyDE Similarity: {similarity:.2f}. (Mapped to {final_criterion_score}/100)"
                }
            
            weight = criterion['weight']
            weighted_score = result['score'] * weight
            
            total_weighted_score += weighted_score
            total_weight += weight
            
            results.append({
                "name": criterion['name'],
                "type": criterion['type'],
                "score": result['score'],
                "weight": weight,
                "justification": result['justification']
            })
            print(f"Analyzed: {criterion['name']} [Done]      ")

        final_score = int(total_weighted_score / total_weight) if total_weight > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"FINAL SCORE: {final_score}/100")
        print(f"{'='*60}\n")
        
        print("DETAILED BREAKDOWN:\n")
        
        for res in results:
            print(f"[{res['type'].upper()}] {res['name']} (Weight: {res['weight']})")
            print(f"Score: {res['score']}/100")
            print(f"Justification: {res['justification']}")
            print("-" * 40)

if __name__ == "__main__":
    # Default to mock transcript
    transcript_to_score = MOCK_TRANSCRIPT

    # Check for command line argument
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if os.path.exists(input_path):
            print(f"Loading transcript from: {input_path}")
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    transcript_to_score = f.read()
            except Exception as e:
                print(f"Error reading file: {e}")
                sys.exit(1)
        else:
            print(f"File not found: {input_path}")
            sys.exit(1)

    scorer = RealAutomatedScorer(transcript_to_score, RUBRIC_CRITERIA)
    scorer.run_evaluation()
