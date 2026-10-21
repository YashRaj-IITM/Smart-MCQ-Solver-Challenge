import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
option_cols = ['A', 'B', 'C', 'D', 'E']

def preprocess(df):
    first_sentences = [[p]*5 for p in df['prompt']]
    second_sentences = [[df.iloc[i][col] for col in option_cols] for i in range(len(df))]
    
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    
    tokenized = tokenizer(first_sentences, second_sentences, truncation=True, max_length=256, padding="max_length")
    return {k: [v[i:i+5] for i in range(0, len(v), 5)] for k, v in tokenized.items()}

class MCQDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features['input_ids'])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.features.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

def train_bert_model(train_dataset, val_dataset, output_dir="./model3_output"):
    model3 = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)
    if torch.cuda.is_available():
        model3 = model3.to("cuda")

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=1,
        max_steps=150,
        logging_steps=25,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model3,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    return model3, trainer
