# Smart MCQ Solver Challenge

Project repository for **Deep Learning and Generative AI (BSDA2001P)**, Diploma Level in Data Science & Applications.

- **Student Name**: Yash Raj
- **Student ID**: 23f2004827
- **Repository Name**: DL-23f2004827-notebook-t22026
- **GitHub Link**: [https://github.com/23f2004827/dl-23f2004827-notebook-t22026](https://github.com/23f2004827/dl-23f2004827-notebook-t22026)
- **Kaggle Competition**: Smart MCQ Solver Challenge
- **Evaluation Metric**: Mean Average Precision at 3 (MAP@3)
- **Kaggle Public Score**: **0.73192** (Version 11 submission)
- **Kaggle Private Score**: **0.76262** (Version 11 submission)
- **Best Validation Ensemble MAP@3**: **0.9283**
- **W&B Project**: `23f2004827-t22026` ([https://wandb.ai/23f2004827-t22026](https://wandb.ai/23f2004827-t22026))
- **Live Deployment Link (Hugging Face)**: [https://huggingface.co/spaces/Yash-Raj-1812/DL-23f2004827-mcq-solver](https://huggingface.co/spaces/Yash-Raj-1812/DL-23f2004827-mcq-solver)
- **Alternative Deployment Link (Streamlit Cloud)**: [https://appappppy-6ezt6h6twvolnaxei5yjef.streamlit.app](https://appappppy-6ezt6h6twvolnaxei5yjef.streamlit.app)
- **Project Report (PDF)**: [reports/report.pdf](reports/report.pdf)
- **Project Report (Markdown)**: [reports/final-report.md](reports/final-report.md)

---

## Model Experiments & Comparative Results

All model experiments were tracked using Weights & Biases under project `23f2004827-t22026`.

| Model | Architecture / Method | Validation MAP@3 | Accuracy | Macro F1 | Status / W&B Logged |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Baseline** | TF-IDF (5,000 features) + Cosine Similarity | 0.2762 (train) | — | — | Baseline reference |
| **Model 1 (Scratch)** | PyTorch ScratchMCQNet (TF-IDF features) | 0.9122 | 0.8400 | 0.8450 | `map@3` (Validation) |
| **Model 2 (Pretrained)** | `sentence-transformers/all-MiniLM-L6-v2` | 0.4050 | 0.2600 | 0.2578 | `map@3` (Validation) |
| **Model 3 (Fine-tuned)** | `bert-base-uncased` MultipleChoice Head | 0.8900 | 0.8200 | 0.8135 | `map@3` (Validation) |
| **Ensemble (Rank Fusion)** | Scratch (0.5) + MiniLM (0.1) + BERT (0.4) | **0.9283** | — | — | Best Validation Ensemble |
| **Kaggle Submission** | Blend: Scratch (0.6) + MiniLM (0.2) + BERT (0.2) | — | — | — | **Public: 0.73192 \| Private: 0.76262** |

---

## Directory Structure

```text
dl-23f2004827-notebook-t22026/
├── DL-23f2004827-notebook-t22026.ipynb   # Main end-to-end notebook (57 cells with all models & Qwen LLM)
├── README.md                             # Project overview and documentation
├── requirements.txt                      # Project dependencies
├── notebooks/                            # Milestone notebooks
│   ├── final_notebook.ipynb              # Complete final notebook
│   ├── milestone-1.ipynb                 # Milestone 1: NLP Foundation & TF-IDF baseline
│   ├── milestone-2.ipynb                 # Milestone 2: Pretrained Transformers & Embeddings
│   ├── milestone-3.ipynb                 # Milestone 3: RAG pipeline concepts
│   ├── milestone-4.ipynb                 # Milestone 4: MCQ formatting & BERT Fine-Tuning
│   └── milestone-5.ipynb                 # Milestone 5: Ensembling & Final Predictions
├── src/                                  # Modular source code
│   ├── model.py                          # ScratchMCQNet PyTorch architecture
│   ├── train.py                          # Training pipeline
│   ├── inference.py                      # Batch inference and ensembling
│   └── utils.py                          # MAP@3 evaluation metric and helpers
├── models/                               # Saved model weights
├── reports/                              # Project reports
│   ├── report.pdf                        # Official 6-page comprehensive PDF report
│   ├── Yash_Raj-23F2004827-IITM_BS.pdf   # Form submission PDF report
│   └── final-report.md                   # Full 14-section markdown technical report
└── deployment/                           # Live deployment configurations
    ├── app.py                            # Gradio web app for Hugging Face Spaces
    ├── streamlit_app.py                  # Streamlit web app
    ├── requirements.txt                  # Deployment dependencies
    └── README.md                         # Hugging Face Spaces metadata
```

---

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Training & Evaluation:**
   ```bash
   python src/train.py
   ```

3. **Run Inference:**
   ```bash
   python src/inference.py
   ```

4. **Launch Local Web App:**
   ```bash
   streamlit run deployment/streamlit_app.py
   # or
   python deployment/app.py
   ```
