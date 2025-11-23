import streamlit as st
from real_automated_scorer import RealAutomatedScorer, RUBRIC_CRITERIA

# Page Config
st.set_page_config(
    page_title="AI Communication Scorer",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Minimalist Design
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background-color: #262730;
        border: 1px solid #464B5C;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        color: #979797;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 24px;
        font-weight: 700;
    }
    .highlight-good { color: #00CC96; }
    .highlight-avg { color: #FFA15A; }
    .highlight-bad { color: #EF553B; }
    
    /* Clean up Streamlit defaults */
    .stTextArea textarea {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #464B5C;
    }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("## 🎤 AI Communication Scorer")
st.markdown("Minimalist analysis of student introductions using **RAG + HyDE**.")
st.divider()

# Main Layout
col_input, col_summary = st.columns([2, 1], gap="large")

with col_input:
    st.markdown("### Transcript Input")
    transcript_input = st.text_area(
        "Paste text here...",
        height=250,
        placeholder="Hello everyone, myself Muskan...",
        label_visibility="collapsed"
    )
    analyze_btn = st.button("Analyze Transcript", type="primary")

with col_summary:
    st.markdown("### System Status")
    st.info("🟢 **AI Model**: all-MiniLM-L6-v2 (Loaded)")
    st.info("🟢 **HyDE**: Active (Static Proxies)")
    st.info("🟢 **Anomaly Detection**: Active")

# Analysis Logic
if analyze_btn and transcript_input:
    scorer = RealAutomatedScorer(transcript_input, RUBRIC_CRITERIA)
    
    # --- 1. Advanced Anomaly Check ---
    anomalies = scorer.detect_anomalies()
    if anomalies:
        # Separate by severity
        critical = [a for a in anomalies if a.get('severity') == 'critical']
        warnings = [a for a in anomalies if a.get('severity') == 'warning']
        
        if critical:
            st.error(f"🚨 **{len(critical)} Critical Issue(s) Detected**")
            for a in critical:
                st.markdown(f"""
                <div style="padding: 10px; background-color: #3d1818; border-left: 4px solid #ff4b4b; margin-bottom: 10px;">
                    <strong>{a['name']}</strong><br>
                    <span style="font-size: 12px; color: #ffbaba;">{a['description']}</span><br>
                    <span style="font-size: 11px; color: #ff8888;">Confidence: {a['confidence']:.1%}</span>
                </div>
                """, unsafe_allow_html=True)
        
        if warnings:
            st.warning(f"⚠️ **{len(warnings)} Warning(s)**")
            for a in warnings:
                st.markdown(f"""
                <div style="padding: 10px; background-color: #3d2e18; border-left: 4px solid #ffa500; margin-bottom: 10px;">
                    <strong>{a['name']}</strong><br>
                    <span style="font-size: 12px; color: #ffd699;">{a['description']}</span><br>
                    <span style="font-size: 11px; color: #ffcc88;">Confidence: {a['confidence']:.1%}</span>
                </div>
                """, unsafe_allow_html=True)
    
    
    # --- 2. Scoring Loop ---
    total_weighted_score = 0
    total_weight = 0
    results = []
    
    for criterion in scorer.rubric:
        if criterion['type'] == 'rule':
            res = scorer.evaluate_rule_based(criterion)
        else:
            similarity = scorer.calculate_semantic_similarity(criterion)
            raw_score = similarity * 100
            boosted_score = min(100, raw_score * 1.3)
            res = {"score": int(boosted_score), "justification": f"Semantic Match: {similarity:.2f}"}
        
        w = criterion['weight']
        total_weighted_score += res['score'] * w
        total_weight += w
        results.append({**res, "name": criterion['name'], "weight": w})
        
    final_score = int(total_weighted_score / total_weight) if total_weight > 0 else 0
    
    # --- 3. Results Grid ---
    st.divider()
    st.markdown("### Analysis Results")
    
    # Overall Score Card
    score_color = "highlight-good" if final_score >= 80 else "highlight-avg" if final_score >= 60 else "highlight-bad"
    st.markdown(f"""
    <div class="metric-card" style="text-align: center;">
        <div class="metric-title">OVERALL SCORE</div>
        <div class="metric-value {score_color}" style="font-size: 48px;">{final_score}/100</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed Cards Grid
    row1 = st.columns(3)
    row2 = st.columns(3)
    grid_cols = row1 + row2
    
    for i, res in enumerate(results):
        col = grid_cols[i] if i < len(grid_cols) else None
        if col:
            with col:
                s_color = "highlight-good" if res['score'] >= 80 else "highlight-avg" if res['score'] >= 60 else "highlight-bad"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{res['name'].upper()}</div>
                    <div class="metric-value {s_color}">{res['score']}</div>
                    <div style="font-size: 12px; color: #979797; margin-top: 8px;">{res['justification']}</div>
                </div>
                """, unsafe_allow_html=True)

elif analyze_btn and not transcript_input:
    st.warning("Please paste a transcript first.")
