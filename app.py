import streamlit as st
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CineScore · Sentiment Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS  — Film-noir dark theme with amber accents
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>

/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --ink:        #0A0B0E;
    --surface:    #111318;
    --card:       #181C24;
    --border:     #252A35;
    --border-hi:  #343B4A;
    --amber:      #E8A830;
    --amber-dim:  #A87420;
    --amber-glow: rgba(232,168,48,0.12);
    --red:        #E8503A;
    --green:      #38C96A;
    --muted:      #606880;
    --text:       #CDD3E0;
    --text-hi:    #EEF0F5;
}

/* ── Base reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background: var(--ink); }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1200px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .sidebar-content { padding: 0; }

/* ── Sidebar title ── */
.sb-logo {
    padding: 28px 24px 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
}
.sb-logo .mark {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: var(--amber);
    letter-spacing: 0.02em;
    line-height: 1;
}
.sb-logo .sub {
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Nav links ── */
div[data-testid="stRadio"] label {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 20px;
    margin: 2px 0;
    border-radius: 8px;
    color: var(--muted) !important;
    font-size: 14px;
    font-weight: 400;
    cursor: pointer;
    transition: all .15s;
}
div[data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,0.04);
    color: var(--text) !important;
}
div[data-testid="stRadio"] [aria-checked="true"] + label,
div[data-testid="stRadio"] input:checked + div {
    background: var(--amber-glow) !important;
    color: var(--amber) !important;
}

/* ── Hide radio buttons ── */
div[data-testid="stRadio"] input[type="radio"] { display: none; }
div[data-testid="stRadio"] > div { gap: 0; }

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    color: var(--text-hi);
}
h1 { font-size: 36px; line-height: 1.15; }
h2 { font-size: 24px; }
h3 { font-size: 18px; }

/* ── Hero strip ── */
.hero-strip {
    position: relative;
    padding: 40px 44px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 2rem;
}
.hero-strip::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: var(--amber);
    border-radius: 4px 0 0 4px;
}
.hero-strip .eyebrow {
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 10px;
}
.hero-strip h1 { margin: 0 0 12px; }
.hero-strip p {
    color: var(--muted);
    font-size: 15px;
    margin: 0;
    max-width: 560px;
    line-height: 1.7;
}

/* ── Stat cards ── */
.stat-row { display: flex; gap: 16px; margin-bottom: 2rem; flex-wrap: wrap; }
.stat-card {
    flex: 1;
    min-width: 140px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 22px;
}
.stat-card .label {
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}
.stat-card .value {
    font-family: 'DM Serif Display', serif;
    font-size: 30px;
    color: var(--amber);
    line-height: 1;
}
.stat-card .delta {
    font-size: 12px;
    color: var(--muted);
    margin-top: 6px;
}

/* ── Feature list ── */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-bottom: 2rem;
}
.feature-item {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
}
.feature-item .icon {
    font-size: 22px;
    margin-bottom: 10px;
}
.feature-item .title {
    font-weight: 600;
    color: var(--text-hi);
    font-size: 14px;
    margin-bottom: 4px;
}
.feature-item .desc {
    font-size: 13px;
    color: var(--muted);
    line-height: 1.6;
}

/* ── Section header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 2rem 0 1.2rem;
}
.section-header .line {
    flex: 1;
    height: 1px;
    background: var(--border);
}
.section-header .label {
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
}

/* ── Textarea ── */
textarea {
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    resize: vertical;
}
textarea:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 3px var(--amber-glow) !important;
}

/* ── Button ── */
div[data-testid="stButton"] button {
    background: var(--amber) !important;
    color: #0A0B0E !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 28px !important;
    letter-spacing: 0.02em;
    transition: all .15s;
}
div[data-testid="stButton"] button:hover {
    background: #f0b840 !important;
    transform: translateY(-1px);
}

/* ── Result cards ── */
.result-positive {
    background: rgba(56,201,106,0.08);
    border: 1px solid rgba(56,201,106,0.30);
    border-radius: 12px;
    padding: 24px 28px;
    margin: 1.2rem 0;
}
.result-negative {
    background: rgba(232,80,58,0.08);
    border: 1px solid rgba(232,80,58,0.30);
    border-radius: 12px;
    padding: 24px 28px;
    margin: 1.2rem 0;
}
.result-label {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
    opacity: 0.7;
}
.result-verdict {
    font-family: 'DM Serif Display', serif;
    font-size: 32px;
    line-height: 1;
}
.result-positive .result-verdict { color: var(--green); }
.result-negative .result-verdict { color: var(--red); }
.result-confidence {
    font-size: 14px;
    color: var(--muted);
    margin-top: 8px;
}

/* ── Confidence bar ── */
.conf-bar-wrap {
    background: var(--border);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
    margin: 6px 0 2px;
}
.conf-bar-fill-pos {
    height: 100%;
    border-radius: 100px;
    background: var(--green);
    transition: width .5s cubic-bezier(.4,0,.2,1);
}
.conf-bar-fill-neg {
    height: 100%;
    border-radius: 100px;
    background: var(--red);
    transition: width .5s cubic-bezier(.4,0,.2,1);
}

/* ── Prob row ── */
.prob-row {
    display: flex;
    gap: 16px;
    margin: 1.4rem 0 1.8rem;
}
.prob-pill {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
}
.prob-pill .plabel {
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 4px;
}
.prob-pill .pval { font-size: 22px; font-weight: 600; }
.prob-pill.pos .pval { color: var(--green); }
.prob-pill.neg .pval { color: var(--red); }

/* ── Table styling ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* ── Plotly chart bg override ── */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* ── Info box ── */
.info-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 28px;
    margin: 1rem 0;
}
.info-box h3 { margin: 0 0 14px; color: var(--text-hi); }
.info-box p, .info-box li {
    font-size: 14px;
    color: var(--muted);
    line-height: 1.8;
}
.info-box ul { margin: 0; padding-left: 18px; }

/* ── Architecture flow ── */
.arch-flow {
    display: flex;
    align-items: center;
    gap: 0;
    flex-wrap: wrap;
    margin: 1.2rem 0;
}
.arch-step {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 13px;
    color: var(--text);
    white-space: nowrap;
}
.arch-arrow {
    color: var(--amber);
    font-size: 18px;
    padding: 0 6px;
}

/* ── Tag/badge ── */
.badge {
    display: inline-block;
    background: var(--amber-glow);
    color: var(--amber);
    border: 1px solid var(--amber-dim);
    border-radius: 100px;
    padding: 3px 12px;
    font-size: 12px;
    font-weight: 500;
}

/* ── Comparison winner ── */
.winner-card {
    border-color: var(--amber) !important;
    position: relative;
}
.winner-badge {
    position: absolute;
    top: -12px; right: 16px;
    background: var(--amber);
    color: var(--ink);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 100px;
}

/* ── Footer ── */
.footer {
    border-top: 1px solid var(--border);
    padding-top: 1.5rem;
    margin-top: 3rem;
    font-size: 12px;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
}

/* Streamlit overrides */
[data-testid="metric-container"] { display: none; }
.stProgress > div > div > div { background: var(--amber) !important; }
div[data-testid="stAlert"] { border-radius: 10px !important; }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    model_path = "DemonKing112/imdb-distilbert-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────

def predict_sentiment(text):
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, padding=True, max_length=256
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pos = probs[0][1].item()
    neg = probs[0][0].item()
    label = "Positive" if pos > neg else "Negative"
    confidence = max(pos, neg)
    return label, confidence, pos, neg


# ─────────────────────────────────────────────────────────────
# PLOTLY THEME HELPER
# ─────────────────────────────────────────────────────────────

def dark_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#181C24",
        font=dict(family="DM Sans", color="#CDD3E0"),
        margin=dict(t=40, b=20, l=20, r=20),
        title_font=dict(family="DM Serif Display", size=18, color="#EEF0F5"),
    )
    fig.update_xaxes(gridcolor="#252A35", linecolor="#252A35", tickcolor="#606880")
    fig.update_yaxes(gridcolor="#252A35", linecolor="#252A35", tickcolor="#606880")
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="mark">CineScore</div>
        <div class="sub">Sentiment Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Analyze Review",
            "📊  Performance",
            "⚖️  Model Comparison",
            "🧠  DistilBERT",
            "📋  Project Info",
        ],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div style="padding:0 20px;margin-top:24px;">
        <div style="font-size:11px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px;">
            Model Health
        </div>
        <div style="background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin-bottom:20px;">
            <div style="font-size:12px;color:var(--muted);margin-bottom:2px;">Accuracy</div>
            <div style="font-size:20px;color:var(--amber);font-family:'DM Serif Display',serif;">92.16%</div>
            <div style="height:4px;background:var(--border);border-radius:99px;margin-top:8px;">
                <div style="width:92.16%;height:100%;background:var(--amber);border-radius:99px;"></div>
            </div>
        </div>
        <div style="font-size:11px;color:var(--muted);text-align:center;padding-bottom:8px;">
            Built by <span style="color:var(--amber);">Aditya</span> &middot; DistilBERT fine-tuned
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────

if "Overview" in page:

    st.markdown("""
    <div class="hero-strip">
        <div class="eyebrow">&#127916; Transformer &middot; NLP &middot; Classification</div>
        <h1>IMDB Sentiment<br>Intelligence</h1>
        <p>
            State-of-the-art sentiment analysis powered by a fine-tuned DistilBERT
            transformer trained on 50,000 real IMDB movie reviews. Identify emotional
            tone with 92%+ accuracy in milliseconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-row">
        <div class="stat-card">
            <div class="label">Accuracy</div>
            <div class="value">92.16%</div>
            <div class="delta">&#8593; 3.37% vs baseline</div>
        </div>
        <div class="stat-card">
            <div class="label">Precision</div>
            <div class="value">91.83%</div>
            <div class="delta">Positive class</div>
        </div>
        <div class="stat-card">
            <div class="label">Recall</div>
            <div class="value">92.56%</div>
            <div class="delta">Positive class</div>
        </div>
        <div class="stat-card">
            <div class="label">F1 Score</div>
            <div class="value">92.19%</div>
            <div class="delta">Harmonic mean</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header">
        <div class="line"></div>
        <div class="label">Key Capabilities</div>
        <div class="line"></div>
    </div>

    <div class="feature-grid">
        <div class="feature-item">
            <div class="icon">⚡</div>
            <div class="title">Real-time Inference</div>
            <div class="desc">Sub-second predictions powered by DistilBERT's optimised 66M parameter architecture.</div>
        </div>
        <div class="feature-item">
            <div class="icon">📐</div>
            <div class="title">Calibrated Confidence</div>
            <div class="desc">Softmax probabilities give precise confidence scores for both sentiment classes.</div>
        </div>
        <div class="feature-item">
            <div class="icon">🗃️</div>
            <div class="title">50K Training Reviews</div>
            <div class="desc">Fine-tuned on the full IMDB benchmark dataset with an 80/20 train-val split.</div>
        </div>
        <div class="feature-item">
            <div class="icon">📈</div>
            <div class="title">Baseline Comparison</div>
            <div class="desc">Outperforms TF-IDF + SGDClassifier by 3.37 percentage points on the same data.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: ANALYZE REVIEW
# ─────────────────────────────────────────────────────────────

elif "Analyze" in page:

    st.markdown("<h2>Analyze a Review</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:var(--muted);font-size:14px;margin-bottom:1.5rem;'>Paste any movie review below. The model will classify its sentiment and return calibrated probabilities.</p>",
        unsafe_allow_html=True,
    )

    review = st.text_area(
        "Movie review",
        placeholder="e.g. 'This film was a masterpiece — the cinematography was breathtaking and the performances were flawless...'",
        height=200,
        label_visibility="collapsed",
    )

    col_btn, col_space = st.columns([1, 4])
    with col_btn:
        run = st.button("Run Analysis →")

    if run:
        if not review.strip():
            st.warning("Please paste a review before running analysis.")
        else:
            with st.spinner("Inferring sentiment…"):
                label, conf, pos, neg = predict_sentiment(review)

            css_class = "result-positive" if label == "Positive" else "result-negative"
            emoji = "😊" if label == "Positive" else "😞"
            color_word = "Positive" if label == "Positive" else "Negative"
            bar_class = "conf-bar-fill-pos" if label == "Positive" else "conf-bar-fill-neg"

            st.markdown(f"""
            <div class="{css_class}">
                <div class="result-label">Verdict</div>
                <div class="result-verdict">{emoji} {color_word}</div>
                <div class="result-confidence">Confidence · {conf*100:.2f}%</div>
                <div class="conf-bar-wrap" style="margin-top:12px;">
                    <div class="{bar_class}" style="width:{conf*100:.1f}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-pill pos">
                    <div class="plabel">Positive probability</div>
                    <div class="pval">{pos*100:.2f}%</div>
                </div>
                <div class="prob-pill neg">
                    <div class="plabel">Negative probability</div>
                    <div class="pval">{neg*100:.2f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            df = pd.DataFrame({
                "Sentiment": ["Positive", "Negative"],
                "Probability": [round(pos * 100, 2), round(neg * 100, 2)],
            })

            fig = px.bar(
                df, x="Sentiment", y="Probability",
                color="Sentiment",
                color_discrete_map={"Positive": "#38C96A", "Negative": "#E8503A"},
                title="Prediction Confidence Distribution",
                labels={"Probability": "Probability (%)"},
            )
            fig.update_traces(marker_line_width=0, width=0.4)
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE: PERFORMANCE ANALYTICS
# ─────────────────────────────────────────────────────────────

elif "Performance" in page:

    st.markdown("<h2>Model Performance</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:var(--muted);font-size:14px;margin-bottom:1.5rem;'>Evaluation results on 10,000 held-out validation samples.</p>",
        unsafe_allow_html=True,
    )

    # Metrics bar
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Score": [92.16, 91.83, 92.56, 92.19],
    })
    fig_m = px.bar(
        metrics_df, x="Metric", y="Score",
        title="Classification Metrics (%)",
        color="Score",
        color_continuous_scale=[[0, "#343B4A"], [0.5, "#A87420"], [1, "#E8A830"]],
        text="Score",
    )
    fig_m.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        marker_line_width=0,
        width=0.45,
    )
    fig_m.update_layout(coloraxis_showscale=False, yaxis_range=[88, 94])
    dark_fig(fig_m)
    st.plotly_chart(fig_m, use_container_width=True)

    # Confusion matrix
    cm = [[4588, 412], [372, 4628]]
    cm_fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Pred: Negative", "Pred: Positive"],
            y=["True: Negative", "True: Positive"],
            colorscale=[[0, "#181C24"], [0.5, "#A87420"], [1, "#E8A830"]],
            showscale=True,
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            textfont={"size": 16, "color": "white"},
        )
    )
    cm_fig.update_layout(title="Confusion Matrix — Validation Set (10,000 samples)")
    dark_fig(cm_fig)
    st.plotly_chart(cm_fig, use_container_width=True)

    # Class report
    st.markdown("""
    <div class="info-box">
        <h3>Classification Report</h3>
        <table style="width:100%;font-size:13px;border-collapse:collapse;">
            <thead>
                <tr>
                    <th style="text-align:left;padding:8px 0;color:var(--muted);font-weight:500;border-bottom:1px solid var(--border);">Class</th>
                    <th style="text-align:right;padding:8px 0;color:var(--muted);font-weight:500;border-bottom:1px solid var(--border);">Precision</th>
                    <th style="text-align:right;padding:8px 0;color:var(--muted);font-weight:500;border-bottom:1px solid var(--border);">Recall</th>
                    <th style="text-align:right;padding:8px 0;color:var(--muted);font-weight:500;border-bottom:1px solid var(--border);">F1</th>
                    <th style="text-align:right;padding:8px 0;color:var(--muted);font-weight:500;border-bottom:1px solid var(--border);">Support</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding:8px 0;color:var(--text);">Negative</td>
                    <td style="text-align:right;padding:8px 0;color:var(--text);">92.55%</td>
                    <td style="text-align:right;padding:8px 0;color:var(--text);">91.76%</td>
                    <td style="text-align:right;padding:8px 0;color:var(--text);">92.15%</td>
                    <td style="text-align:right;padding:8px 0;color:var(--muted);">5000</td>
                </tr>
                <tr>
                    <td style="padding:8px 0;color:var(--text);">Positive</td>
                    <td style="text-align:right;padding:8px 0;color:var(--text);">91.83%</td>
                    <td style="text-align:right;padding:8px 0;color:var(--text);">92.56%</td>
                    <td style="text-align:right;padding:8px 0;color:var(--text);">92.19%</td>
                    <td style="text-align:right;padding:8px 0;color:var(--muted);">5000</td>
                </tr>
                <tr style="border-top:1px solid var(--border);">
                    <td style="padding:8px 0;color:var(--amber);font-weight:600;">Weighted Avg</td>
                    <td style="text-align:right;padding:8px 0;color:var(--amber);">92.19%</td>
                    <td style="text-align:right;padding:8px 0;color:var(--amber);">92.16%</td>
                    <td style="text-align:right;padding:8px 0;color:var(--amber);">92.17%</td>
                    <td style="text-align:right;padding:8px 0;color:var(--muted);">10000</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: MODEL COMPARISON
# ─────────────────────────────────────────────────────────────

elif "Comparison" in page:

    st.markdown("<h2>TF-IDF vs DistilBERT</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:var(--muted);font-size:14px;margin-bottom:1.5rem;'>Head-to-head benchmark on the same 10,000 validation reviews.</p>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="info-box" style="position:relative;height:100%;">
            <div style="font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:12px;">Baseline</div>
            <div style="font-family:'DM Serif Display',serif;font-size:20px;color:var(--text-hi);margin-bottom:16px;">TF-IDF + SGD Classifier</div>
            <div style="font-size:13px;color:var(--muted);line-height:2;">
                <div>&#8226; Bag-of-words vectorisation</div>
                <div>&#8226; No contextual understanding</div>
                <div>&#8226; Fast, low memory footprint</div>
                <div>&#8226; Struggles with negation and sarcasm</div>
            </div>
            <div style="margin-top:16px;">
                <div style="font-size:12px;color:var(--muted);margin-bottom:4px;">Accuracy</div>
                <div style="font-size:28px;color:var(--text-hi);font-family:'DM Serif Display',serif;">88.79%</div>
                <div style="height:6px;background:var(--border);border-radius:99px;margin-top:8px;">
                    <div style="width:88.79%;height:100%;background:var(--muted);border-radius:99px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="info-box winner-card" style="border:1px solid var(--amber);position:relative;height:100%;">
            <div class="winner-badge">&#10022; Best Model</div>
            <div style="font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--amber);margin-bottom:12px;">Production</div>
            <div style="font-family:'DM Serif Display',serif;font-size:20px;color:var(--text-hi);margin-bottom:16px;">DistilBERT Fine-Tuned</div>
            <div style="font-size:13px;color:var(--muted);line-height:2;">
                <div>&#8226; Deep contextual representations</div>
                <div>&#8226; Handles negation and nuance</div>
                <div>&#8226; 6-layer transformer, 66M params</div>
                <div>&#8226; Fine-tuned on domain data</div>
            </div>
            <div style="margin-top:16px;">
                <div style="font-size:12px;color:var(--muted);margin-bottom:4px;">Accuracy</div>
                <div style="font-size:28px;color:var(--amber);font-family:'DM Serif Display',serif;">92.16%</div>
                <div style="height:6px;background:var(--border);border-radius:99px;margin-top:8px;">
                    <div style="width:92.16%;height:100%;background:var(--amber);border-radius:99px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    comp_df = pd.DataFrame({
        "Model": ["TF-IDF + SGD", "DistilBERT"],
        "Accuracy": [88.79, 92.16],
    })
    fig_c = px.bar(
        comp_df, x="Model", y="Accuracy",
        title="Accuracy Comparison (%)",
        color="Model",
        color_discrete_map={"TF-IDF + SGD": "#606880", "DistilBERT": "#E8A830"},
        text="Accuracy",
    )
    fig_c.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        marker_line_width=0,
        width=0.35,
    )
    fig_c.update_layout(yaxis_range=[85, 95], showlegend=False)
    dark_fig(fig_c)
    st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("""
    <div style="background:rgba(232,168,48,0.08);border:1px solid rgba(232,168,48,0.3);border-radius:10px;padding:16px 20px;font-size:14px;color:var(--amber);">
        &#10022; DistilBERT outperforms the TF-IDF baseline by <strong>+3.37 percentage points</strong> &mdash; a 30% reduction in error rate.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: ABOUT DISTILBERT
# ─────────────────────────────────────────────────────────────

elif "DistilBERT" in page:

    st.markdown("<h2>About DistilBERT</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:var(--muted);font-size:14px;margin-bottom:1.5rem;'>A distilled version of BERT — smaller, faster, and still 97% as capable.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:2rem;">
        <div class="stat-card">
            <div class="label">Parameters</div>
            <div class="value" style="font-size:24px;">66M</div>
            <div class="delta">vs 110M in BERT-base</div>
        </div>
        <div class="stat-card">
            <div class="label">Layers</div>
            <div class="value" style="font-size:24px;">6</div>
            <div class="delta">vs 12 in BERT-base</div>
        </div>
        <div class="stat-card">
            <div class="label">Speed</div>
            <div class="value" style="font-size:24px;">60%</div>
            <div class="delta">Faster than BERT</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-bottom:1.2rem;">
        <h3>Architecture Overview</h3>
        <p>DistilBERT is trained via knowledge distillation from BERT-base. It retains the general language understanding while being significantly more efficient. The base model used is <code>distilbert-base-uncased</code>.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h3>Inference Pipeline</h3>
        <div class="arch-flow">
            <div class="arch-step">Raw Review Text</div>
            <div class="arch-arrow">&#8594;</div>
            <div class="arch-step">WordPiece Tokeniser</div>
            <div class="arch-arrow">&#8594;</div>
            <div class="arch-step">DistilBERT Encoder</div>
            <div class="arch-arrow">&#8594;</div>
            <div class="arch-step">CLS Pooling</div>
            <div class="arch-arrow">&#8594;</div>
            <div class="arch-step">Linear Head</div>
            <div class="arch-arrow">&#8594;</div>
            <div class="arch-step">Softmax Output</div>
        </div>
        <ul style="margin-top:14px;">
            <li>Max token length: 256 (truncation applied for longer reviews)</li>
            <li>Padding applied dynamically per batch</li>
            <li>Inference runs on CPU with <code>torch.no_grad()</code></li>
            <li>Classification head: single linear layer, 2 logits</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: PROJECT INFO
# ─────────────────────────────────────────────────────────────

elif "Project" in page:

    st.markdown("<h2>Project Information</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>Dataset</h3>
            <ul>
                <li><strong style="color:var(--text-hi);">Source:</strong> IMDB Movie Review Dataset</li>
                <li><strong style="color:var(--text-hi);">Total samples:</strong> 50,000 labelled reviews</li>
                <li><strong style="color:var(--text-hi);">Training split:</strong> 40,000 reviews (80%)</li>
                <li><strong style="color:var(--text-hi);">Validation split:</strong> 10,000 reviews (20%)</li>
                <li><strong style="color:var(--text-hi);">Classes:</strong> Binary — Positive / Negative</li>
                <li><strong style="color:var(--text-hi);">Balance:</strong> Perfectly balanced (25k each)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <h3>Final Results</h3>
            <ul>
                <li>Accuracy · <span style="color:var(--amber);">92.16%</span></li>
                <li>Precision · <span style="color:var(--amber);">91.83%</span></li>
                <li>Recall · <span style="color:var(--amber);">92.56%</span></li>
                <li>F1 Score · <span style="color:var(--amber);">92.19%</span></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>Training Configuration</h3>
            <ul>
                <li><strong style="color:var(--text-hi);">Base model:</strong> distilbert-base-uncased</li>
                <li><strong style="color:var(--text-hi);">Epochs:</strong> 3</li>
                <li><strong style="color:var(--text-hi);">Learning rate:</strong> 2e-5 (AdamW)</li>
                <li><strong style="color:var(--text-hi);">Batch size:</strong> 32</li>
                <li><strong style="color:var(--text-hi);">Max token length:</strong> 256</li>
                <li><strong style="color:var(--text-hi);">Hardware:</strong> Tesla T4 (Google Colab)</li>
                <li><strong style="color:var(--text-hi);">Framework:</strong> HuggingFace Transformers + PyTorch</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <h3>Tech Stack</h3>
            <ul>
                <li>🤗 HuggingFace Transformers</li>
                <li>🔥 PyTorch</li>
                <li>🎈 Streamlit</li>
                <li>📊 Plotly Express</li>
                <li>🐍 Python 3.10+</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        <span>CineScore &middot; IMDB Sentiment Intelligence</span>
        <span>Author: <span style="color:var(--amber);">Aditya</span> &middot; DistilBERT &middot; PyTorch &middot; Streamlit</span>
    </div>
    """, unsafe_allow_html=True)
