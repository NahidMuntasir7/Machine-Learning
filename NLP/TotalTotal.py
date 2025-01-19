Step-by-Step Implementation for Embedding, Model, and Training
1. Load the Dataset
First, load the dataset from your CSV file. The dataset should have two columns: sentence and label.


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('path_to_your_csv_file.csv')

# Split the dataset into train and test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['sentence'], df['label'], test_size=0.2, random_state=42
)
2. Tokenization and Embedding Preparation
Here, you will tokenize the sentences and convert them into embeddings using a pre-trained model like BERT.


from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT tokenizer and BERT model (multilingual or Bangla-specific)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')  # Change to a model that supports Bangla
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Tokenize the sentences
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512, return_tensors="pt")
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=512, return_tensors="pt")

# Labels must be converted to tensors
train_labels = torch.tensor(train_labels.values)
test_labels = torch.tensor(test_labels.values)

3. Data Loader
Next, create data loaders to feed your batches into the model.


from torch.utils.data import TensorDataset, DataLoader, random_split

# Create Tensor datasets from BERT encodings and labels
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
4. LSTM Model Definition
Now, define your LSTM model that will use the BERT embeddings.


import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)  # Using BERT's hidden states
        lstm_out = lstm_out[:, -1, :]  # Use only the last hidden state for classification
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out

# Define model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(input_dim=768, hidden_dim=128, output_dim=len(df['label'].unique())).to(device)


5. Training the LSTM Model
Now, define the training loop for your LSTM model.


from sklearn.metrics import accuracy_score
import torch.optim as optim

# Define the criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_lstm(model, train_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = tuple(b.to(device) for b in batch)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

# Train the model
train_lstm(model, train_loader)


6. Evaluation
Finally, you’ll evaluate the LSTM model using accuracy on the test set.


def evaluate_lstm(model, test_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = tuple(b.to(device) for b in batch)
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

# Evaluate the LSTM model
lstm_accuracy = evaluate_lstm(model, test_loader)
print(f"LSTM Model Accuracy: {lstm_accuracy * 100:.2f}%")


7. BERT Fine-Tuning Model
Next, you’ll fine-tune BERT using the same data and tokenized inputs.


from transformers import BertForSequenceClassification

# BERT fine-tuning model
class BertClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

# Initialize the BERT model
model_name = 'bert-base-multilingual-cased'  # Change to Bangla-specific model if available
bert_model = BertClassifier(model_name=model_name, num_classes=len(df['label'].unique())).to(device)


8. BERT Training Loop
Fine-tuning the BERT model:


from transformers import AdamW
from transformers import get_scheduler

# BERT fine-tuning training function
def train_bert(model, train_loader, optimizer, scheduler, epochs=3):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = tuple(b.to(device) for b in batch)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

# Optimizer and scheduler
optimizer = AdamW(bert_model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 3
)

# Train BERT model
train_bert(bert_model, train_loader, optimizer, lr_scheduler)


9. BERT Model Evaluation
Finally, evaluate the BERT model.

  
def evaluate_bert(model, test_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = tuple(b.to(device) for b in batch)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

# Evaluate BERT model
bert_accuracy = evaluate_bert(bert_model, test_loader)
print(f"BERT Model Accuracy: {bert_accuracy * 100:.2f}%")



Summary:
Data Preparation: Tokenize sentences and generate embeddings using BERT.
Model 1: LSTM uses BERT embeddings and LSTM layers.
Model 2: BERT fine-tunes directly using the encoded sentences.
Training: Use Adam optimizer, Cross-Entropy Loss, and epochs to optimize both models.
Evaluation: Accuracy calculation on test data to measure the models’ performance.
