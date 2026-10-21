#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class ScratchMCQNet(nn.Module):
    def __init__(self, dim=2000, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(dim * 2, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, prompt_vec, option_vec):
        combined = torch.cat([prompt_vec, option_vec], dim=-1)
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)

def train_scratch_mlp(train_df, val_df, max_features=2000, epochs=30, lr=1e-3):
    option_cols = ['A', 'B', 'C', 'D', 'E']
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    vec = TfidfVectorizer(max_features=max_features, stop_words='english')
    all_text = train_df['prompt'].tolist()
    for col in option_cols:
        all_text += train_df[col].tolist()
    vec.fit(all_text)
    
    def make_feats(df):
        p = torch.tensor(vec.transform(df['prompt']).toarray(), dtype=torch.float32)
        opts = torch.stack([torch.tensor(vec.transform(df[c]).toarray(), dtype=torch.float32) for c in option_cols], dim=1)
        return p, opts

    p_train, o_train = make_feats(train_df)
    p_val, o_val = make_feats(val_df)
    y_train = torch.tensor([label_map[a] for a in train_df['answer']])
    
    model = ScratchMCQNet(dim=max_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    p_train_exp = p_train.unsqueeze(1).expand(-1, 5, -1)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(p_train_exp, o_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        
    return model, vec

if __name__ == "__main__":
    print("PyTorch Scratch MCQ Neural Network module loaded.")
