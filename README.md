# AI Introduction Scorer

An automated tool to analyze and score student introductions based on the Nirmaan AI Case Study rubrics.
This project uses **RAG (Retrieval-Augmented Generation)** and **HyDE (Hypothetical Document Embeddings)** to evaluate semantic quality, alongside rule-based logic for quantitative metrics.

## Features

- **Hybrid Scoring Engine**:
  - **Rule-Based**: Checks Word Count, Speech Rate (WPM), and Filler Word usage.
  - **Semantic (AI)**: Uses `sentence-transformers` (all-MiniLM-L6-v2) to evaluate Content, Grammar, and Engagement.
- **HyDE Strategy**: Uses "Ideal Student Responses" as search queries to improve semantic matching accuracy.
- **Anomaly Detection**: Automatically flags statements that are inconsistent with a student persona (e.g., claiming to be a professional engineer).
- **Dual Interface**:
  - **CLI**: Run scripts directly for quick analysis.
  - **Web UI**: A user-friendly Streamlit app for easy interaction.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Web UI (Recommended)
Launch the interactive dashboard:
```bash
streamlit run app.py
```
- Paste the student transcript.
- Click **Analyze Transcript**.
- View detailed scores, justifications, and anomaly warnings.

### 2. CLI (Command Line)
Run the scorer on a text file:
```bash
python real_automated_scorer.py path/to/transcript.txt
```

## Scoring Logic

| Criterion | Type | Method |
| :--- | :--- | :--- |
| **Word Count** | Rule | Checks if count is within 100-180 words. |
| **Speech Rate** | Rule | Estimates WPM based on word count (Target: 110-140). |
| **Filler Words** | Rule | Calculates percentage of fillers (um, uh, like). |
| **Content** | AI (HyDE) | Compares against an ideal introduction structure. |
| **Grammar** | AI (HyDE) | Compares against a grammatically perfect sample. |
| **Engagement** | AI (HyDE) | Compares against an enthusiastic, high-energy sample. |

## Anomaly Detection
The tool checks for:
- **Professional Experience**: Claims of corporate work history.
- **Unrealistic Qualifications**: Claims of PhDs or advanced degrees.

## Files
- `app.py`: Streamlit Web Application.
- `real_automated_scorer.py`: Core scoring engine.
- `requirements.txt`: Project dependencies.
