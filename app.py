import streamlit as st
import numpy as np
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import requests
import os
import time
import plotly.graph_objects as go
from datetime import datetime

# --- NEW UI LIBRARY ---
from streamlit_extras.colored_header import colored_header
from streamlit_extras.let_it_rain import rain
from streamlit_extras.metric_cards import style_metric_cards

# --- IMPORT API KEY ---
try:
    from config import OPENROUTER_API_KEY
except ImportError:
    st.error("⚠️ CRITICAL: config.py file not found! Please create 'config.py' and add OPENROUTER_API_KEY")
    st.stop()

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_embed_model.json")
LOCAL_EMBED_PATH = os.path.join(BASE_DIR, "models", "MiniLM_model")
DEFAULT_EMBED_NAME = "all-MiniLM-L6-v2"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "mistralai/devstral-2512:free"

# --- PAGE CONFIG ---
st.set_page_config(page_title="Sentinal AI Chatbot", page_icon="🛡️", layout="wide")

# --- CUSTOM CSS FOR "BLOCKED" ALERT ---
st.markdown("""
<style>
    .blocked-card {
        background-color: #ffe6e6;
        border: 2px solid #ff4b4b;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .blocked-title {
        color: #d32f2f;
        font-size: 24px;
        font-weight: 900;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .blocked-msg {
        color: #b71c1c;
        font-size: 18px;
        font-weight: bold;
    }
    .safe-badge {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        border: 1px solid #2e7d32;
    }
    .metric-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def create_gauge_chart(probability):
    score = probability * 100
    color = "#ff4b4b" if score > 50 else "#00cc96"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        number = {'suffix': "%", 'font': {'size': 40}},
        title = {'text': "Malicious Probability", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0, 204, 150, 0.3)'},
                {'range': [50, 100], 'color': 'rgba(255, 75, 75, 0.3)'}
            ],
        }
    ))
    fig.update_layout(height=300, margin={'t': 40, 'b': 0, 'l': 20, 'r': 20})
    return fig

@st.cache_resource
def load_models():
    if os.path.exists(LOCAL_EMBED_PATH) and os.listdir(LOCAL_EMBED_PATH):
        embedder = SentenceTransformer(LOCAL_EMBED_PATH)
    else:
        embedder = SentenceTransformer(DEFAULT_EMBED_NAME)

    classifier = xgb.Booster()
    if os.path.exists(MODEL_PATH):
        classifier.load_model(MODEL_PATH)
    else:
        st.error(f"❌ Model missing at {MODEL_PATH}")
        st.stop()
    return embedder, classifier

def query_openrouter_llm(user_prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:8501", 
        "Content-Type": "application/json"
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": 0.7, 
    }
    try:
        response = requests.post(OPENROUTER_API_URL, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'], None
        else:
            return None, f"API Error: {response.text}"
    except Exception as e:
        return None, str(e)

# --- LOAD MODELS ---
embedder, classifier = load_models()

# --- SIDEBAR HISTORY ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/11521/11521626.png", width=80)
    st.title("🛡️ Session Logs")
    
    if "history" not in st.session_state:
        st.session_state.history = []
        
    if st.button("🗑️ Clear Logs", use_container_width=True):
        st.session_state.history = []
    
    st.markdown("---")
    for item in reversed(st.session_state.history):
        color = "🔴" if item['status'] == "Malicious" else "🟢"
        with st.expander(f"{color} {item['time']}"):
            st.caption(f"Score: {item['score']:.1f}%")
            st.write(item['prompt'])

# --- MAIN UI ---
colored_header(
    label="Sentinel AI ChatBot",
    description="Advanced ML-Powered Prompt Injection Firewall",
    color_name="red-70"
)

# Input Section
user_prompt = st.text_area("✍️ Enter Prompt to Security Scan:", height=120, placeholder="Try 'How do I hack a server?' vs 'How do I bake a cake?'")

col_btn, col_info = st.columns([1, 4])
with col_btn:
    analyze_btn = st.button("🚀 SCAN & EXECUTE", type="primary", use_container_width=True)

if analyze_btn and user_prompt:
    
    start_time = time.time()
    
    # 1. SCANNING
    with st.status("🔍 Analyzing neural patterns...", expanded=True) as status:
        st.write("⚙️ Vectorizing text (MiniLM)...")
        embedding = embedder.encode(user_prompt)
        time.sleep(0.3) # UX delay
        
        st.write("🧮 Running XGBoost Classifier...")
        dmatrix_input = xgb.DMatrix(np.array([embedding]))
        prob = classifier.predict(dmatrix_input)[0]
        status.update(label="Analysis Complete", state="complete", expanded=False)

    is_malicious = prob > 0.5
    malicious_score = prob * 100
    
    # Save History
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "prompt": user_prompt,
        "status": "Malicious" if is_malicious else "Safe",
        "score": malicious_score
    })
    
    st.divider()
    
    # Layout Results
    left_col, right_col = st.columns([1, 2])
    
    # --- LEFT COLUMN: GAUGE CHART ---
    with left_col:
        st.subheader("Threat Meter")
        st.plotly_chart(create_gauge_chart(prob), use_container_width=True)
        
        # Metric Cards (streamlit-extras)
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Confidence", f"{malicious_score:.1f}%")
        col_m2.metric("Latency", f"{(time.time()-start_time):.2f}s")
        style_metric_cards()

    # --- RIGHT COLUMN: RESULT & ACTION ---
    with right_col:
        st.subheader("Action Center")
        
        if is_malicious:
            # === BLOCKED UI ===
            st.markdown(f"""
            <div class="blocked-card">
                <div class="blocked-title">🚫 INTERCEPTED & BLOCKED</div>
                <div class="blocked-msg">
                    This prompt was flagged as MALICIOUS.<br>
                    It has been refused by the security layer.
                </div>
                <br>
                <div style="color:gray; font-size:12px;">Request ID: {hash(user_prompt)} | Action: DROP</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("🔎 View Technical Details"):
                st.json({"verdict": "MALICIOUS", "score": float(prob), "vector_dim": embedding.shape})
        
        else:
            # === SAFE UI ===
            rain(emoji="🛡️", font_size=54, falling_speed=5, animation_length=1)
            
            st.success("✅ **SECURITY CHECK PASSED**")
            st.info(f"Forwarding request to LLM ({OPENROUTER_MODEL})...")
            
            # Call API
            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking..."):
                    response, error = query_openrouter_llm(user_prompt)
                    if response:
                        st.write(response)
                    else:
                        st.error(error)

elif analyze_btn and not user_prompt:
    st.warning("⚠️ Please enter some text to analyze.")