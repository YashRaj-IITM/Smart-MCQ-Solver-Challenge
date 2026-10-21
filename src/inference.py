import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from utils import map_at_3, ensemble_predict

option_cols = ['A', 'B', 'C', 'D', 'E']

def tfidf_predict(df):
    all_text = df['prompt'].tolist()
    for col in option_cols:
        all_text += df[col].tolist()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    vectorizer.fit(all_text)
    
    predictions = []
    for _, row in df.iterrows():
        prompt_vec = vectorizer.transform([str(row['prompt'])])
        opt_vecs = vectorizer.transform([str(row[col]) for col in option_cols])
        sims = cosine_similarity(prompt_vec, opt_vecs)[0]
        top3_idx = np.argsort(-sims)[:3]
        predictions.append(" ".join([option_cols[i] for i in top3_idx]))
    return predictions

def sentence_transformer_predict(df, model_st, batch_size=32):
    prompts = df['prompt'].tolist()
    prompt_embeds = model_st.encode(prompts, batch_size=batch_size, show_progress_bar=False)
    
    predictions = []
    for i, row in df.iterrows():
        opts = [str(row[col]) for col in option_cols]
        opt_embeds = model_st.encode(opts, show_progress_bar=False)
        sims = cosine_similarity([prompt_embeds[i]], opt_embeds)[0]
        top3_idx = np.argsort(-sims)[:3]
        predictions.append(" ".join([option_cols[k] for k in top3_idx]))
    return predictions

def run_full_inference(test, test_preds_scratch, test_preds_m2, test_preds_m3, weights=[0.6, 0.2, 0.2]):
    test_preds_ensemble = ensemble_predict(
        [test_preds_scratch, test_preds_m2, test_preds_m3],
        weights=weights
    )

    submission = pd.DataFrame({"ID": test["id"], "Prediction": test_preds_ensemble})
    submission.to_csv("submission.csv", index=False)
    print(submission.head())
    print("Submission saved. Rows:", len(submission))
    return submission
