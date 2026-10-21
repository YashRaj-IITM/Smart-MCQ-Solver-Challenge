import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load real pretrained SentenceTransformer model
model_st = SentenceTransformer("all-MiniLM-L6-v2")

option_cols = ['A', 'B', 'C', 'D', 'E']

def predict_mcq(prompt, opt_a, opt_b, opt_c, opt_d, opt_e):
    if not prompt or not opt_a:
        return "Please provide a valid question prompt and options.", ""
    
    opts = [opt_a, opt_b, opt_c, opt_d, opt_e]
    
    # Compute real embeddings using all-MiniLM-L6-v2
    prompt_embed = model_st.encode([prompt], show_progress_bar=False)
    opt_embeds = model_st.encode(opts, show_progress_bar=False)
    
    # Compute cosine similarity scores
    sims = cosine_similarity(prompt_embed, opt_embeds)[0]
    
    # Rank options by similarity
    top3_idx = np.argsort(-sims)[:3]
    top3_letters = [option_cols[i] for i in top3_idx]
    prediction = " ".join(top3_letters)
    
    # Format detailed score breakdown
    breakdown = "### 📊 Option Similarity Scores (all-MiniLM-L6-v2):\n"
    for idx in np.argsort(-sims):
        letter = option_cols[idx]
        score = sims[idx]
        breakdown += f"- **Option {letter}** ({opts[idx]}): `{score:.4f}`\n"
        
    return f"**Top 3 Predictions (MAP@3 Ranking)**: `{prediction}`", breakdown

demo = gr.Interface(
    fn=predict_mcq,
    inputs=[
        gr.Textbox(label="Question Prompt", value="What is the SI base unit of time?", lines=2),
        gr.Textbox(label="Option A", value="Meter"),
        gr.Textbox(label="Option B", value="Second"),
        gr.Textbox(label="Option C", value="Kilogram"),
        gr.Textbox(label="Option D", value="Ampere"),
        gr.Textbox(label="Option E", value="Kelvin"),
    ],
    outputs=[
        gr.Markdown(label="Top-3 Candidate Predictions"),
        gr.Markdown(label="Detailed Model Scores"),
    ],
    title="🤖 Smart MCQ Solver — Live Model Inference",
    description="Powered by `sentence-transformers/all-MiniLM-L6-v2`. Ranks candidate answers using 384-dimensional cosine similarity embeddings.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
