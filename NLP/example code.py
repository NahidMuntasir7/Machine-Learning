import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Hyperparameters
PRETRAINED_MODEL = "sagorsarker/bangla-bert-base"
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy dataset (replace this with your labeled Bangla dataset)
data = [
    {"text": "তুমি কি বইটি পড়েছ?", "label": 1},  # Example for Assertive
    {"text": "দয়া করে বইটি দাও।", "label": 2},  # Example for Imperative
    {"text": "তুমি কি জানো?", "label": 3},      # Example for Interrogative
    {"text": "কি সুন্দর দিন!", "label": 4},    # Example for Exclamatory
    {"text": "তুমি তা করছ কেন?", "label": 5},  # Example for Others
]

# Split dataset
texts = [item["text"] for item in data]
labels = [item["label"] for item in data]
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Dataset class
class BanglaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label - 1, dtype=torch.long)  # Labels start from 0
        }
# Initialize tokenizer and datasets
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
train_dataset = BanglaDataset(train_texts, train_labels, tokenizer)
val_dataset = BanglaDataset(val_texts, val_labels, tokenizer)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
class BanglaClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(BanglaClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        return self.fc(dropout_output)

model = BanglaClassifier(pretrained_model=PRETRAINED_MODEL, num_classes=5).to(DEVICE)

# Optimizer and Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Training Function
def train_model(model, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

        train_acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # Validation
        validate_model(model, val_loader)

# Validation Function
def validate_model(model, val_loader):
    model.eval()
    correct, total_loss = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = correct / len(val_loader.dataset)
    print(f"Validation Loss: {total_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    print(classification_report(all_labels, all_preds))

# Train the model
train_model(model, train_loader, val_loader, EPOCHS)
