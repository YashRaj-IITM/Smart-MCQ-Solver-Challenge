---
title: Smart MCQ Solver Challenge
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: mit
---

# Smart MCQ Solver Challenge

Deep Learning & GenAI Course Project (BS in Data Science & Applications). Student ID: 23f2004827

## Note on Deployed Model Architecture

This Hugging Face Space uses Model 2 (SentenceTransformer, `all-MiniLM-L6-v2`) for the live interactive web demo, selected for its compact memory footprint (~80MB) and low-latency CPU inference suited to free cloud hosting hardware.

The project's full GPU-accelerated ensemble pipeline (PyTorch Scratch MLP + SentenceTransformer + Fine-Tuned BERT, MAP@3 = 0.9189 on validation) is documented in the main project notebooks and source scripts.
