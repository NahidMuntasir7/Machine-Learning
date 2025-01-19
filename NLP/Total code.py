import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CSV file
data = pd.read_csv("data.csv")  # Replace with your CSV file
sentences = data["sentence"]  # Sentence column
labels = data["label"]  # Label column

# Split the dataset
train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    sentences, labels, test_size=0.2, random_state=42
)

# Pre-trained model and tokenizer
MODEL_NAME = "bert-base-multilingual-cased"  # Change if needed (e.g., xlm-roberta-base)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME)

# Dataset class
class SentenceDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        # Tokenize sentence
        encoding = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Create datasets and dataloaders
train_dataset = SentenceDataset(train_sentences.tolist(), train_labels.tolist(), tokenizer)
test_dataset = SentenceDataset(test_sentences.tolist(), test_labels.tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# === LSTM Model === #
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last hidden state
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out

# Generate sentence embeddings using BERT
def generate_embeddings(dataloader, model):
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings
            embeddings.append(cls_embeddings)
            labels.append(label)
    return torch.cat(embeddings), torch.cat(labels)

# Generate embeddings for LSTM
train_embeddings, train_labels = generate_embeddings(train_loader, bert_model)
test_embeddings, test_labels = generate_embeddings(test_loader, bert_model)

# Train LSTM
def train_lstm(train_embeddings, train_labels, test_embeddings, test_labels):
    lstm_model = LSTMClassifier(embedding_dim=train_embeddings.size(1), hidden_dim=128, output_dim=len(labels.unique())).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    epochs = 10

    for epoch in range(epochs):
        lstm_model.train()
        optimizer.zero_grad()
        outputs = lstm_model(train_embeddings.to(device))
        loss = criterion(outputs, train_labels.to(device))
        loss.backward()
        optimizer.step()

    # Evaluate LSTM
    lstm_model.eval()
    with torch.no_grad():
        test_outputs = lstm_model(test_embeddings.to(device))
        _, predicted = torch.max(test_outputs, 1)
        accuracy = accuracy_score(test_labels.cpu(), predicted.cpu())
    return accuracy

# === Train and Evaluate BERT === #
def train_bert(train_loader, test_loader):
    bert_classifier = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(labels.unique())).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(bert_classifier.parameters(), lr=2e-5)
    epochs = 3

    for epoch in range(epochs):
        bert_classifier.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = bert_classifier(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Evaluate BERT
    bert_classifier.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = bert_classifier(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

# Train and Evaluate Models
lstm_accuracy = train_lstm(train_embeddings, train_labels, test_embeddings, test_labels)
bert_accuracy = train_bert(train_loader, test_loader)

print(f"LSTM Accuracy: {lstm_accuracy * 100:.2f}%")
print(f"BERT Accuracy: {bert_accuracy * 100:.2f}%")
