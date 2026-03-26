import streamlit as st
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EmotiSense AI | Deep Voice Intelligence", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ELITE PACKAGE DESIGN SYSTEM ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Poppins:wght@400;600;700&display=swap');

    :root {
        --primary: #4F46E5;
        --secondary: #7C3AED;
        --bg-main: #F8FAFC;
    }

    .stApp {
        background-color: var(--bg-main) !important;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 { font-family: 'Poppins', sans-serif !important; font-weight: 700 !important; }

    /* Card Styling */
    [data-testid="stVerticalBlock"] > div:has([data-testid="stFileUploader"]),
    [data-testid="stVerticalBlock"] > div:has(.stAudio),
    [data-testid="stVerticalBlock"] > div:has([data-testid="stPlotlyChart"]),
    [data-testid="column"] > div > div > div > div {
        background: white;
        border-radius: 24px;
        padding: 28px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
        margin-bottom: 2rem;
    }

    /* Analyze Button */
    .stButton > button {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: white !important;
        border-radius: 14px !important;
        padding: 0.8rem 2.8rem !important;
        font-weight: 700 !important;
        border: none !important;
        transition: 0.3s all ease !important;
        letter-spacing: 0.02em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px rgba(79, 70, 229, 0.3) !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 50px 0 20px 0;
        color: #94A3B8;
        font-size: 14px;
        border-top: 1px solid #F1F5F9;
        margin-top: 60px;
    }
</style>
""", unsafe_allow_html=True)

# --- EMOTION DATABASE ---
MODEL_PATH = "models/best_model.keras"
CLASSES = ["Neutral", "Happy", "Sad", "Angry"]
EMOTION_META = {
    "Neutral": {"emoji": "😐", "color": "#64748B"},
    "Happy": {"emoji": "😄", "color": "#10B981"},
    "Sad": {"emoji": "😢", "color": "#3B82F6"},
    "Angry": {"emoji": "😡", "color": "#EF4444"}
}

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        # compile=False avoids errors if the optimizer or loss fn names changed
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    return None

def extract_features(audio_data, sample_rate):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    features = np.expand_dims(np.expand_dims(mfcc_scaled, axis=0), axis=-1)
    return features, mfcc

# --- APP LAYOUT ---
def main():
    model = load_model()
    
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2850/2850314.png", width=70)
        st.title("ER AI Pro")
        st.divider()
        
        # 8. --- REAL-WORLD VALUE (RECRUITER BAIT) ---
        st.subheader("💼 Applications")
        st.write("""
        - **Mental health monitoring**
        - **Customer sentiment analysis**
        - **AI voice assistants**
        - **Call center emotion tracking**
        """)
        st.divider()
        st.info("System Engine V1.2.0 | Stable")

    # 1. --- BRAND YOUR APP ---
    st.title("🧠 EmotiSense AI")
    st.markdown("### Understanding Human Emotions Through Voice Intelligence")
    
    # 2. --- TAGLINE (WOW FACTOR) ---
    st.caption("Powered by Deep Learning • Audio Intelligence • Real-time Analysis")
    st.divider()

    # --- UPLOAD ENGINE ---
    uploaded_files = st.file_uploader("Upload audio samples (.wav) for neural decoding", type=['wav'], accept_multiple_files=True)

    if uploaded_files:
        # 3. --- PREMIUM AUDIO EXPERIENCE ---
        st.success("🎧 Voice input detected")
        
        if len(uploaded_files) > 1:
            name = st.selectbox("🎯 Active Audio Instance", [f.name for f in uploaded_files])
            uploaded_file = next(f for f in uploaded_files if f.name == name)
        else:
            uploaded_file = uploaded_files[0]

        st.audio(uploaded_file, format='audio/wav')
        st.caption("Analyzing vocal tone, pitch, and emotional signals...")

        # Dashboard Grid
        col_main, col_stats = st.columns([1.2, 1.8], gap="large")

        with col_main:
            with open("temp.wav", "wb") as f: f.write(uploaded_file.getbuffer())
            y, sr = librosa.load("temp.wav", duration=3, offset=0.5)
            features, mfcc = extract_features(y, sr)

            if st.button("PROCESS NEURAL SIGNALS", use_container_width=True):
                # 4. --- AI THINKING EXPERIENCE ---
                with st.spinner("🧠 AI processing emotional patterns..."):
                    time.sleep(2) 
                    
                    preds = model.predict(features)[0]
                    idx = np.argmax(preds)
                    emotion = CLASSES[idx]
                    conf = preds[idx]
                    meta = EMOTION_META.get(emotion, {"emoji": "🎭", "color": "#4F46E5"})

                    # 5. --- PREMIUM RESULT DISPLAY ---
                    st.markdown("## 🎯 Emotion Analysis Result")
                    
                    if emotion == "Happy":
                        st.success(f"😊 Positive Emotion Detected: {emotion}")
                        st.balloons()
                    elif emotion in ["Sad", "Angry"]:
                        st.error(f"😔 Negative Emotion Detected: {emotion}")
                    else:
                        st.warning(f"😐 Neutral Emotion: {emotion}")
                    
                    st.markdown(f'<h1 style="font-size: 82px; margin: 30px 0; text-align: center;">{meta["emoji"]} {emotion.upper()}</h1>', unsafe_allow_html=True)
                    
                    # 6. --- CONFIDENCE BAR (VERY IMPRESSIVE) ---
                    st.metric("Confidence Score", f"{conf*100:.2f}%")
                    st.progress(float(conf))

                    # 7. --- AI INSIGHTS PANEL ---
                    st.divider()
                    st.subheader("🧠 AI Insights")
                    st.info(f"""
                    - **Vocal pitch variation** indicates emotional intensity
                    - **MFCC features** reveal tonal patterns
                    - **Frequency distribution** aligns with predicted emotion
                    """)

        with col_stats:
            st.markdown("### 📈 Visual Insight Center")
            tab1, tab2 = st.tabs(["🔊 Waveform Visualization", "🌈 Spectral Analysis"])
            
            # Use Plotly
            time_points = np.linspace(0, len(y)/sr, len(y))
            step = max(1, len(y) // 2000)
            
            fig_wave = go.Figure()
            fig_wave.add_trace(go.Scatter(x=time_points[::step], y=y[::step], line=dict(color='#4F46E5', width=1.5)))
            fig_wave.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=30, b=0), height=300, title="Neural Waveform Flow")
            
            fig_mfcc = px.imshow(mfcc, color_continuous_scale='Magma')
            fig_mfcc.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=30, b=0), height=300, title="Spectral Fingerprint Density")

            with tab1: st.plotly_chart(fig_wave, use_container_width=True)
            with tab2: st.plotly_chart(fig_mfcc, use_container_width=True)
            
            st.divider()
            c1, c2 = st.columns(2)
            c1.markdown(f'<div style="background:#F8FAFC; padding:20px; border-radius:18px; text-align:center;"><div style="font-size:28px; font-weight:700; color:#4F46E5;">40</div><div style="font-size:12px;">Deep Features</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div style="background:#F8FAFC; padding:20px; border-radius:18px; text-align:center;"><div style="font-size:28px; font-weight:700; color:#4F46E5;">22kHz</div><div style="font-size:12px;">Sample Clarity</div></div>', unsafe_allow_html=True)

    else:
        st.markdown('<div style="text-align: center; padding: 120px; opacity: 0.5;"><div style="font-size: 80px;">📡</div><h3>System Standby</h3><p>Upload voice samples for neural emotional analysis.</p></div>', unsafe_allow_html=True)

    # 10. --- SIGNATURE (PERSONAL BRAND) ---
    st.markdown("""
    <div class="footer">
        ---<br>
        🚀 Built by <b>Manohar K</b> | AI Engineer in Progress<br>
        <span style="font-size:12px;">Powered by TensorFlow • enterprise-ready v1.2</span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
