# app.py - CyberGuard Improved Dark UI

import streamlit as st
import pickle
import numpy as np
import re
import emoji
import nltk
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ── PREPROCESSOR CLASS (must match Colab exactly) ──────────
class MultiPlatformPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.slang_map = {
            'u': 'you', 'ur': 'your', 'r': 'are', 'y': 'why',
            'bc': 'because', 'b4': 'before', 'gr8': 'great',
            'l8r': 'later', 'msg': 'message', 'pls': 'please',
            'plz': 'please', 'thx': 'thanks', 'tho': 'though',
            'rn': 'right now', 'ngl': 'not gonna lie',
            'tbh': 'to be honest', 'imo': 'in my opinion',
            'smh': 'shaking my head', 'fr': 'for real',
            'af': 'as fuck', 'lmao': 'laughing my ass off',
            'lol': 'laughing out loud', 'omg': 'oh my god',
            'wtf': 'what the fuck', 'stfu': 'shut the fuck up',
            'idk': 'i do not know', 'ik': 'i know',
            'gonna': 'going to', 'wanna': 'want to',
            'gotta': 'got to', 'kinda': 'kind of'
        }

    def clean_text(self, text):
        if pd.isna(text) or text == '':
            return ''
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = emoji.demojize(text, delimiters=(" ", " "))
        text = re.sub(r'\brt\b', '', text)
        text = self.expand_slang(text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s\']', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if len(word) > 1]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    def expand_slang(self, text):
        words = text.split()
        return ' '.join([self.slang_map.get(word, word) for word in words])

    def platform_specific_preprocessing(self, text, platform='twitter'):
        text = self.clean_text(text)
        if platform == 'instagram':
            text = re.sub(r'([!?.]){2,}', r'\1', text)
        elif platform == 'youtube':
            text = re.sub(r'\d+:\d+', '', text)
        return text


# ── TOXIC KEYWORDS for word highlighting ──────────────────
TOXIC_WORDS = {
    'high': ['kill', 'die', 'hate', 'stupid', 'ugly', 'worthless', 'pathetic',
             'loser', 'idiot', 'moron', 'disgusting', 'horrible', 'terrible',
             'fat', 'dumb', 'freak', 'trash', 'garbage', 'useless', 'failure'],
    'medium': ['nobody', 'leave', 'go away', 'shut up', 'annoying', 'weird',
               'lame', 'gross', 'awful', 'bad', 'wrong', 'fake', 'lie', 'liar'],
    'low': ['sad', 'cry', 'alone', 'never', 'always', 'stop', 'dont']
}

def highlight_words(text):
    """Return HTML with toxic words highlighted"""
    words = text.split()
    highlighted = []
    for word in words:
        clean_word = word.lower().strip("'\".,!?")
        if clean_word in TOXIC_WORDS['high']:
            highlighted.append(
                f'<span style="background:#ff4444;color:white;padding:2px 6px;'
                f'border-radius:4px;font-weight:bold;">{word}</span>'
            )
        elif clean_word in TOXIC_WORDS['medium']:
            highlighted.append(
                f'<span style="background:#ff8c00;color:white;padding:2px 6px;'
                f'border-radius:4px;">{word}</span>'
            )
        elif clean_word in TOXIC_WORDS['low']:
            highlighted.append(
                f'<span style="background:#ffd700;color:#1a1a2e;padding:2px 6px;'
                f'border-radius:4px;">{word}</span>'
            )
        else:
            highlighted.append(word)
    return ' '.join(highlighted)


# ── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="CyberGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DARK THEME CSS ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Rajdhani:wght@400;500;600;700&display=swap');

    /* Global dark theme */
    .stApp {
        background-color: #0a0e1a;
        color: #e0e0e0;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Header */
    .cyber-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #00d4ff, #7b2fff, #ff006e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 4px;
        margin-bottom: 0;
        text-transform: uppercase;
    }

    .cyber-subheader {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        text-align: center;
        color: #00d4ff;
        letter-spacing: 6px;
        text-transform: uppercase;
        margin-bottom: 2rem;
        opacity: 0.8;
    }

    /* Glowing divider */
    .glow-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #00d4ff, #7b2fff, transparent);
        margin: 1.5rem 0;
        box-shadow: 0 0 10px #00d4ff50;
    }

    /* Platform selector */
    .stRadio > div {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }

    /* Card style */
    .cyber-card {
        background: linear-gradient(135deg, #0d1226, #111827);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.05);
    }

    /* Result cards */
    .result-danger {
        background: linear-gradient(135deg, #1a0505, #2d0a0a);
        border: 1px solid #ff4444;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 0 20px rgba(255, 68, 68, 0.2);
        animation: pulse-danger 2s infinite;
    }

    .result-safe {
        background: linear-gradient(135deg, #051a0a, #0a2d12);
        border: 1px solid #00cc66;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 0 20px rgba(0, 204, 102, 0.2);
    }

    @keyframes pulse-danger {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 68, 68, 0.2); }
        50% { box-shadow: 0 0 30px rgba(255, 68, 68, 0.4); }
    }

    /* Result text */
    .result-title-danger {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #ff4444;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .result-title-safe {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #00cc66;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #0d1226, #0a1628);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }

    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d4ff;
    }

    .metric-label {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Platform info bar */
    .platform-bar {
        background: linear-gradient(135deg, #0d1226, #111827);
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        color: #8892b0;
        margin: 0.5rem 0;
    }

    /* Word highlight legend */
    .legend-high {
        background:#ff4444;
        color:white;
        padding:2px 8px;
        border-radius:4px;
        font-size:0.8rem;
    }
    .legend-medium {
        background:#ff8c00;
        color:white;
        padding:2px 8px;
        border-radius:4px;
        font-size:0.8rem;
    }
    .legend-low {
        background:#ffd700;
        color:#1a1a2e;
        padding:2px 8px;
        border-radius:4px;
        font-size:0.8rem;
    }

    /* Highlighted text box */
    .highlight-box {
        background: #0d1226;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-size: 1.05rem;
        line-height: 2;
        margin: 0.5rem 0;
    }

    /* Stats cards */
    .stat-number {
        font-family: 'Space Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1;
    }

    .stat-label {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #8892b0;
        margin-top: 0.25rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080c18, #0a0e1a);
        border-right: 1px solid #1e3a5f;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff20, #7b2fff20);
        border: 1px solid #00d4ff;
        color: #00d4ff;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #00d4ff40, #7b2fff40);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        transform: translateY(-1px);
    }

    /* Text area */
    .stTextArea textarea {
        background: #0d1226 !important;
        border: 1px solid #1e3a5f !important;
        color: #e0e0e0 !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.9rem !important;
        border-radius: 8px !important;
    }

    .stTextArea textarea:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.2) !important;
    }

    /* Select box */
    .stSelectbox > div > div {
        background: #0d1226 !important;
        border-color: #1e3a5f !important;
        color: #e0e0e0 !important;
    }

    /* Radio */
    .stRadio label {
        color: #8892b0 !important;
        font-family: 'Rajdhani', sans-serif !important;
    }

    /* Section headers */
    .section-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: #00d4ff;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 1rem;
    }

    /* Safety tips */
    .safety-tip {
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        color: #8892b0;
        padding: 0.5rem 0;
        border-bottom: 1px solid #1e3a5f;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #0d1226;
        border-bottom: 1px solid #1e3a5f;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Rajdhani', sans-serif;
        color: #8892b0;
        letter-spacing: 2px;
    }

    .stTabs [aria-selected="true"] {
        color: #00d4ff !important;
        border-bottom: 2px solid #00d4ff !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #0d1226 !important;
        color: #8892b0 !important;
        font-family: 'Rajdhani', sans-serif !important;
    }

    /* History items */
    .history-item {
        background: #0d1226;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        font-family: 'Space Mono', monospace;
        font-size: 0.78rem;
    }

    .history-bully {
        border-left: 3px solid #ff4444;
    }

    .history-safe {
        border-left: 3px solid #00cc66;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #00d4ff !important;
    }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE ──────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_analyzed' not in st.session_state:
    st.session_state.total_analyzed = 0
if 'total_bullying' not in st.session_state:
    st.session_state.total_bullying = 0
if 'platform_counts' not in st.session_state:
    st.session_state.platform_counts = {'Twitter': 0, 'Instagram': 0, 'YouTube': 0, 'TikTok': 0}


# ── LOAD MODEL ─────────────────────────────────────────────
@st.cache_resource
def load_components():
    try:
        model = load_model('cyberbullying_lstm_model.h5', compile=False)
        with open('tokenizer_dl.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        with open('model_config.pkl', 'rb') as f:
            config = pickle.load(f)
        return model, tokenizer, preprocessor, config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

model, tokenizer, preprocessor, config = load_components()


# ── PLATFORM CONFIG ────────────────────────────────────────
platform_config = {
    'Twitter':   {'icon': '🐦', 'color': '#1DA1F2', 'char_limit': 280,   'disclaimer': None},
    'Instagram': {'icon': '📸', 'color': '#E1306C', 'char_limit': 2200,  'disclaimer': 'Results generalized from Twitter training data.'},
    'YouTube':   {'icon': '▶️', 'color': '#FF0000', 'char_limit': 10000, 'disclaimer': 'Results generalized from Twitter training data.'},
    'TikTok':    {'icon': '🎵', 'color': '#69C9D0', 'char_limit': 2200,  'disclaimer': 'Results generalized from Twitter training data.'},
}


# ── PREDICTION ─────────────────────────────────────────────
def predict(text, platform):
    cleaned = preprocessor.platform_specific_preprocessing(text, platform.lower())
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=config['max_len'], padding='post', truncating='post')
    prob = model.predict(padded, verbose=0)[0][0]
    pred = 1 if prob > config['optimal_threshold'] else 0
    confidence = prob * 100 if pred == 1 else (1 - prob) * 100
    return pred, prob, confidence, cleaned


# ── CONFIDENCE GAUGE ───────────────────────────────────────
def make_gauge(probability):
    color = "#ff4444" if probability > config['optimal_threshold'] else "#00cc66"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={'suffix': '%', 'font': {'size': 28, 'color': color, 'family': 'Space Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#8892b0',
                     'tickfont': {'color': '#8892b0', 'size': 10}},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': '#0d1226',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': '#051a0a'},
                {'range': [30, 70], 'color': '#1a1505'},
                {'range': [70, 100], 'color': '#1a0505'},
            ],
            'threshold': {
                'line': {'color': '#00d4ff', 'width': 2},
                'thickness': 0.75,
                'value': config['optimal_threshold'] * 100
            }
        },
        title={'text': "Bullying Probability", 
               'font': {'color': '#8892b0', 'size': 12, 'family': 'Rajdhani'}}
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#8892b0'},
        margin=dict(t=60, b=10, l=30, r=30),
        height=220
    )
    return fig


# ── PLATFORM STATS CHART ───────────────────────────────────
def make_platform_chart():
    platforms = list(st.session_state.platform_counts.keys())
    counts = list(st.session_state.platform_counts.values())
    colors = ['#1DA1F2', '#E1306C', '#FF0000', '#69C9D0']

    fig = go.Figure(go.Bar(
        x=platforms, y=counts,
        marker_color=colors,
        marker_line_width=0,
        text=counts,
        textposition='outside',
        textfont={'color': '#e0e0e0', 'family': 'Space Mono', 'size': 11}
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#8892b0', 'family': 'Rajdhani'},
        xaxis={'gridcolor': '#1e3a5f', 'tickfont': {'color': '#8892b0'}},
        yaxis={'gridcolor': '#1e3a5f', 'tickfont': {'color': '#8892b0'}},
        margin=dict(t=20, b=20, l=20, r=20),
        height=220
    )
    return fig


# ── HISTORY DONUT ──────────────────────────────────────────
def make_donut():
    safe = st.session_state.total_analyzed - st.session_state.total_bullying
    bully = st.session_state.total_bullying

    fig = go.Figure(go.Pie(
        labels=['Safe', 'Bullying'],
        values=[max(safe, 0), max(bully, 0)],
        hole=0.65,
        marker_colors=['#00cc66', '#ff4444'],
        textfont={'color': 'white', 'size': 12},
        hoverinfo='label+percent'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend={'font': {'color': '#8892b0', 'size': 11}},
        margin=dict(t=10, b=10, l=10, r=10),
        height=220,
        annotations=[dict(
            text=f"{bully}/{max(st.session_state.total_analyzed,1)}",
            font=dict(size=16, color='#00d4ff', family='Space Mono'),
            showarrow=False
        )]
    )
    return fig


# ═══════════════════════════════════════════════════════════
# ── MAIN LAYOUT ────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════

# Header
st.markdown('<div class="cyber-header">⟨ CyberGuard ⟩</div>', unsafe_allow_html=True)
st.markdown('<div class="cyber-subheader">// AI-Powered Cyberbullying Detection System //</div>', unsafe_allow_html=True)
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍  ANALYZE", "📊  DASHBOARD"])

# ══════════════════════════════════════════
# TAB 1: ANALYZE
# ══════════════════════════════════════════
with tab1:

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        # Platform selector
        st.markdown('<div class="section-header">// Select Platform</div>', unsafe_allow_html=True)
        selected_platform = st.radio(
            "Platform",
            options=list(platform_config.keys()),
            format_func=lambda x: f"{platform_config[x]['icon']}  {x}",
            horizontal=True,
            label_visibility="collapsed"
        )

        p = platform_config[selected_platform]
        st.markdown(f"""
        <div class="platform-bar">
            {p['icon']} <strong style="color:#00d4ff">{selected_platform}</strong>
            &nbsp;|&nbsp; Char limit: <strong>{p['char_limit']:,}</strong>
            &nbsp;|&nbsp; {"✅ High confidence" if not p['disclaimer'] else "⚠️ Generalized prediction"}
        </div>
        """, unsafe_allow_html=True)

        # Text input
        st.markdown('<div class="section-header" style="margin-top:1.5rem">// Enter Text</div>', unsafe_allow_html=True)
        user_text = st.text_area(
            "Text input",
            height=160,
            placeholder=f"Type or paste a {selected_platform} post, comment, or message...",
            label_visibility="collapsed",
            max_chars=p['char_limit']
        )

        if user_text:
            char_pct = len(user_text) / p['char_limit']
            color = "#ff4444" if char_pct > 0.9 else "#00d4ff"
            st.markdown(
                f'<p style="font-family:Space Mono;font-size:0.75rem;color:{color};'
                f'text-align:right">{len(user_text):,} / {p["char_limit"]:,}</p>',
                unsafe_allow_html=True
            )

        # Analyze button
        st.markdown("")
        analyze_btn = st.button("⟶  ANALYZE TEXT", use_container_width=True)

    with col_right:
        # Stats overview panel
        st.markdown('<div class="section-header">// Session Stats</div>', unsafe_allow_html=True)

        s1, s2 = st.columns(2)
        with s1:
            safe_count = st.session_state.total_analyzed - st.session_state.total_bullying
            st.markdown(f"""
            <div class="cyber-card" style="text-align:center">
                <div class="stat-number" style="color:#00cc66">{safe_count}</div>
                <div class="stat-label">Safe</div>
            </div>
            """, unsafe_allow_html=True)
        with s2:
            st.markdown(f"""
            <div class="cyber-card" style="text-align:center">
                <div class="stat-number" style="color:#ff4444">{st.session_state.total_bullying}</div>
                <div class="stat-label">Flagged</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="cyber-card" style="text-align:center;margin-top:0.5rem">
            <div class="stat-number" style="color:#00d4ff">{st.session_state.total_analyzed}</div>
            <div class="stat-label">Total Analyzed</div>
        </div>
        """, unsafe_allow_html=True)

        # Recent history
        st.markdown('<div class="section-header" style="margin-top:1.5rem">// Recent History</div>', unsafe_allow_html=True)

        if not st.session_state.history:
            st.markdown(
                '<p style="font-family:Space Mono;font-size:0.75rem;color:#8892b0;">'
                'No analyses yet...</p>',
                unsafe_allow_html=True
            )
        else:
            for item in reversed(st.session_state.history[-5:]):
                css_class = "history-bully" if item['result'] == 1 else "history-safe"
                icon = "🔴" if item['result'] == 1 else "🟢"
                label = "BULLYING" if item['result'] == 1 else "SAFE"
                st.markdown(f"""
                <div class="history-item {css_class}">
                    {icon} <strong style="color:{'#ff4444' if item['result']==1 else '#00cc66'}">{label}</strong>
                    &nbsp;·&nbsp; {item['platform']} &nbsp;·&nbsp; {item['conf']:.0f}%<br>
                    <span style="color:#8892b0">{item['text'][:55]}{'...' if len(item['text'])>55 else ''}</span>
                </div>
                """, unsafe_allow_html=True)


    # ── RESULTS ────────────────────────────────────────────
    if analyze_btn:
        if not user_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        elif model is None:
            st.error("❌ Model failed to load. Check your model files.")
        else:
            with st.spinner("Scanning for cyberbullying patterns..."):
                prediction, probability, confidence, cleaned = predict(
                    user_text, selected_platform
                )

            # Update session stats
            st.session_state.total_analyzed += 1
            if prediction == 1:
                st.session_state.total_bullying += 1
            st.session_state.platform_counts[selected_platform] += 1
            st.session_state.history.append({
                'text': user_text,
                'platform': selected_platform,
                'result': prediction,
                'prob': float(probability),
                'conf': confidence,
                'time': datetime.now().strftime("%H:%M:%S")
            })

            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">// Analysis Results</div>', unsafe_allow_html=True)

            # Result + gauge side by side
            res_col1, res_col2 = st.columns([2, 1], gap="large")

            with res_col1:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-danger">
                        <div class="result-title-danger">⚠ Cyberbullying Detected</div>
                        <p style="color:#ff8888;font-family:Space Mono;font-size:0.85rem;margin-top:0.5rem">
                        This content has been flagged as potential cyberbullying.
                        </p>
                        <p style="color:#8892b0;font-family:Space Mono;font-size:0.75rem">
                        Platform: {selected_platform} &nbsp;|&nbsp; Probability: {probability:.3f} &nbsp;|&nbsp; Threshold: {config['optimal_threshold']:.2f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-safe">
                        <div class="result-title-safe">✓ No Cyberbullying Detected</div>
                        <p style="color:#88ff88;font-family:Space Mono;font-size:0.85rem;margin-top:0.5rem">
                        This content appears to be safe and non-bullying.
                        </p>
                        <p style="color:#8892b0;font-family:Space Mono;font-size:0.75rem">
                        Platform: {selected_platform} &nbsp;|&nbsp; Probability: {probability:.3f} &nbsp;|&nbsp; Threshold: {config['optimal_threshold']:.2f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Disclaimer
                if p['disclaimer']:
                    st.markdown(
                        f'<p style="font-family:Space Mono;font-size:0.72rem;'
                        f'color:#8892b0;margin-top:0.5rem">⚠️ {p["disclaimer"]}</p>',
                        unsafe_allow_html=True
                    )

            with res_col2:
                st.plotly_chart(make_gauge(probability), use_container_width=True)

            # ── WORD HIGHLIGHTING ──────────────────────────
            st.markdown('<div class="section-header" style="margin-top:1rem">// Word Analysis</div>', unsafe_allow_html=True)

            highlighted_html = highlight_words(user_text)
            toxic_found = [w for w in user_text.lower().split()
                          if w.strip("'\".,!?") in TOXIC_WORDS['high'] +
                          TOXIC_WORDS['medium'] + TOXIC_WORDS['low']]

            col_h1, col_h2 = st.columns([3, 1])
            with col_h1:
                st.markdown(f'<div class="highlight-box">{highlighted_html}</div>',
                           unsafe_allow_html=True)

                # Legend
                st.markdown("""
                <div style="display:flex;gap:1rem;margin-top:0.5rem;align-items:center;flex-wrap:wrap">
                    <span style="font-family:Rajdhani;font-size:0.8rem;color:#8892b0">Legend:</span>
                    <span class="legend-high">High risk word</span>
                    <span class="legend-medium">Medium risk word</span>
                    <span class="legend-low">Low risk word</span>
                </div>
                """, unsafe_allow_html=True)

            with col_h2:
                st.markdown(f"""
                <div class="cyber-card" style="text-align:center">
                    <div class="stat-number" style="color:#ff4444;font-size:2rem">
                        {len([w for w in user_text.lower().split() if w.strip("'\".,!?") in TOXIC_WORDS['high']])}
                    </div>
                    <div class="stat-label">High Risk</div>
                </div>
                <div class="cyber-card" style="text-align:center;margin-top:0.5rem">
                    <div class="stat-number" style="color:#ff8c00;font-size:2rem">
                        {len([w for w in user_text.lower().split() if w.strip("'\".,!?") in TOXIC_WORDS['medium']])}
                    </div>
                    <div class="stat-label">Medium Risk</div>
                </div>
                """, unsafe_allow_html=True)

            # Safety tips (if bullying detected)
            if prediction == 1:
                st.markdown('<div class="section-header" style="margin-top:1rem">// Recommended Actions</div>', unsafe_allow_html=True)
                tips_col1, tips_col2 = st.columns(2)
                with tips_col1:
                    for tip in ["🚫 Do not respond to the bully", "📸 Screenshot as evidence", "🔒 Block the person"]:
                        st.markdown(f'<div class="safety-tip">{tip}</div>', unsafe_allow_html=True)
                with tips_col2:
                    for tip in ["🚨 Report to the platform", "💬 Talk to a trusted adult", "📞 Contact a helpline if needed"]:
                        st.markdown(f'<div class="safety-tip">{tip}</div>', unsafe_allow_html=True)

            # Preprocessed text
            with st.expander("🔧 View Preprocessed Text"):
                st.code(cleaned, language=None)


# ══════════════════════════════════════════
# TAB 2: DASHBOARD
# ══════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">// Statistics Dashboard</div>', unsafe_allow_html=True)

    if st.session_state.total_analyzed == 0:
        st.markdown("""
        <div class="cyber-card" style="text-align:center;padding:3rem">
            <div style="font-family:Space Mono;font-size:1rem;color:#8892b0">
                No data yet. Run some analyses in the ANALYZE tab to see statistics.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Top stats row
        d1, d2, d3, d4 = st.columns(4)
        safe_pct = ((st.session_state.total_analyzed - st.session_state.total_bullying)
                    / max(st.session_state.total_analyzed, 1) * 100)
        bully_pct = st.session_state.total_bullying / max(st.session_state.total_analyzed, 1) * 100

        with d1:
            st.markdown(f"""
            <div class="cyber-card" style="text-align:center">
                <div class="stat-number" style="color:#00d4ff">{st.session_state.total_analyzed}</div>
                <div class="stat-label">Total Scanned</div>
            </div>""", unsafe_allow_html=True)
        with d2:
            st.markdown(f"""
            <div class="cyber-card" style="text-align:center">
                <div class="stat-number" style="color:#ff4444">{st.session_state.total_bullying}</div>
                <div class="stat-label">Flagged</div>
            </div>""", unsafe_allow_html=True)
        with d3:
            st.markdown(f"""
            <div class="cyber-card" style="text-align:center">
                <div class="stat-number" style="color:#ff8c00">{bully_pct:.1f}%</div>
                <div class="stat-label">Detection Rate</div>
            </div>""", unsafe_allow_html=True)
        with d4:
            most_used = max(st.session_state.platform_counts,
                           key=st.session_state.platform_counts.get)
            st.markdown(f"""
            <div class="cyber-card" style="text-align:center">
                <div class="stat-number" style="color:#7b2fff;font-size:1.5rem">{platform_config[most_used]['icon']}</div>
                <div class="stat-label">Top Platform</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Charts row
        chart1, chart2 = st.columns(2)

        with chart1:
            st.markdown('<div class="section-header">// Safe vs Flagged</div>', unsafe_allow_html=True)
            st.plotly_chart(make_donut(), use_container_width=True)

        with chart2:
            st.markdown('<div class="section-header">// Analyses by Platform</div>', unsafe_allow_html=True)
            st.plotly_chart(make_platform_chart(), use_container_width=True)

        # Full history table
        if st.session_state.history:
            st.markdown('<div class="section-header" style="margin-top:1rem">// Full Analysis History</div>', unsafe_allow_html=True)

            history_df = pd.DataFrame(st.session_state.history)
            history_df['Result'] = history_df['result'].map({1: '🔴 BULLYING', 0: '🟢 SAFE'})
            history_df['Confidence'] = history_df['conf'].apply(lambda x: f"{x:.1f}%")
            history_df['Text Preview'] = history_df['text'].apply(
                lambda x: x[:60] + '...' if len(x) > 60 else x)

            display_df = history_df[['time', 'platform', 'Result', 'Confidence', 'Text Preview']].copy()
            display_df.columns = ['Time', 'Platform', 'Result', 'Confidence', 'Text']
            display_df = display_df.iloc[::-1].reset_index(drop=True)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=300
            )

            # Clear history button
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.history = []
                st.session_state.total_analyzed = 0
                st.session_state.total_bullying = 0
                st.session_state.platform_counts = {
                    'Twitter': 0, 'Instagram': 0, 'YouTube': 0, 'TikTok': 0
                }
                st.rerun()


# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:Rajdhani;font-size:1.5rem;font-weight:700;
    color:#00d4ff;letter-spacing:3px;text-align:center;margin-bottom:0.5rem">
    ⟨ CYBERGUARD ⟩
    </div>
    <div style="font-family:Space Mono;font-size:0.65rem;color:#8892b0;
    text-align:center;margin-bottom:1.5rem">AI SAFETY SYSTEM v1.0</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">// Model Info</div>', unsafe_allow_html=True)
    if config:
        info_items = [
            ("Model", "Bidirectional LSTM"),
            ("F1-Score", f"{config['f1_score']*100:.1f}%"),
            ("Threshold", f"{config['optimal_threshold']:.2f}"),
            ("Vocab Size", f"{config['vocabulary_size']:,}"),
            ("Training Data", "Twitter Dataset"),
        ]
        for label, value in info_items:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
            padding:0.4rem 0;border-bottom:1px solid #1e3a5f;
            font-family:Space Mono;font-size:0.75rem">
                <span style="color:#8892b0">{label}</span>
                <span style="color:#00d4ff">{value}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-header">// Platforms</div>', unsafe_allow_html=True)
    for pname, pconf in platform_config.items():
        trained = "✅" if not pconf['disclaimer'] else "⚠️"
        st.markdown(f"""
        <div style="font-family:Space Mono;font-size:0.75rem;
        padding:0.3rem 0;color:#8892b0">
            {pconf['icon']} {pname} {trained}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:Space Mono;font-size:0.65rem;color:#8892b0;margin-top:0.5rem">
    ✅ High confidence &nbsp; ⚠️ Generalized
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:Space Mono;font-size:0.65rem;color:#8892b0;
    text-align:center;line-height:1.8">
    This tool uses AI for detection.<br>
    Always apply human judgment.<br><br>
    <span style="color:#1e3a5f">────────────────</span><br>
    BSc Project<br>Cyberbullying Detection System
    </div>
    """, unsafe_allow_html=True)