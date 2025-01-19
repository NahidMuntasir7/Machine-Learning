import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Load CSV
data = pd.read_csv('data.csv')  # Replace 'data.csv' with your file name
sentences = data['sentence']  # Column name for sentences
labels = data['label']  # Column name for labels

# Pre-trained model and tokenizer
MODEL_NAME = "bert-base-multilingual-cased"  # You can use "xlm-roberta-base" or another Bangla-supported model
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME)

# Custom Dataset Class
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
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Split dataset into train and test
train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    sentences, labels, test_size=0.2, random_state=42
)

# Create datasets and data loaders
train_dataset = SentenceDataset(train_sentences.tolist(), train_labels.tolist(), tokenizer)
test_dataset = SentenceDataset(test_sentences.tolist(), test_labels.tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Generate embeddings using BERT
def generate_embeddings(dataloader, model):
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            label = batch['label']
            
            # Pass through BERT model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Use the CLS token representation
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embeddings)
            labels.append(label)

    # Concatenate all embeddings and labels
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    return embeddings, labels

# Generate embeddings for train and test sets
train_embeddings, train_labels = generate_embeddings(train_loader, bert_model)
test_embeddings, test_labels = generate_embeddings(test_loader, bert_model)

# Now you can use train_embeddings and test_embeddings for classification
print("Train Embeddings Shape:", train_embeddings.shape)
print("Test Embeddings Shape:", test_embeddings.shape)
