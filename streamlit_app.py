import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Smart MCQ Solver", page_icon="🤖", layout="centered")

st.title("🤖 Smart MCQ Solver — Live Model Inference")
st.markdown("### Deep Learning & GenAI Project (BSDA2001P) • Roll No: 23f2004827")
st.markdown("Ranks multiple-choice candidate answers (A–E) using vector representations and cosine similarity scoring.")

option_cols = ['A', 'B', 'C', 'D', 'E']

prompt = st.text_area("Question Prompt", value="What is the SI base unit of time?", height=100)
col1, col2 = st.columns(2)
with col1:
    opt_a = st.text_input("Option A", value="Meter")
    opt_b = st.text_input("Option B", value="Second")
    opt_c = st.text_input("Option C", value="Kilogram")
with col2:
    opt_d = st.text_input("Option D", value="Ampere")
    opt_e = st.text_input("Option E", value="Kelvin")

if st.button("🚀 Predict Top-3 Answers (MAP@3)", type="primary"):
    if not prompt.strip() or not opt_a.strip():
        st.warning("Please provide a question prompt and candidate options.")
    else:
        opts = [opt_a, opt_b, opt_c, opt_d, opt_e]
        all_text = [prompt] + opts
        
        vectorizer = TfidfVectorizer(stop_words='english')
        vectorizer.fit(all_text)
        
        prompt_vec = vectorizer.transform([prompt])
        opt_vecs = vectorizer.transform(opts)
        
        sims = cosine_similarity(prompt_vec, opt_vecs)[0]
        
        top3_idx = np.argsort(-sims)[:3]
        top3_letters = [option_cols[i] for i in top3_idx]
        prediction = " ".join(top3_letters)
        
        st.success(f"### 🎯 Top-3 Ranked Predictions: **`{prediction}`**")
        
        st.markdown("#### 📊 Candidate Similarity Scores:")
        for idx in np.argsort(-sims):
            letter = option_cols[idx]
            score = sims[idx]
            st.write(f"- **Option {letter}** ({opts[idx]}): `{score:.4f}`")
