1. Word2Vec
First, let's generate Word2Vec embeddings using a pre-trained model and fit it into your LSTM model.

Step-by-Step Implementation:
1.1. Loading the Dataset
python
Copy
Edit
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from CSV
df = pd.read_csv('path_to_your_csv_file.csv')

# Split the dataset into train and test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['sentence'], df['label'], test_size=0.2, random_state=42
)
1.2. Word2Vec Embedding Preparation
First, load the Word2Vec model:

python
Copy
Edit
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import preprocessing
import nltk
nltk.download('punkt')

# Tokenize the sentences
tokenized_texts = [preprocessing.tokenize(doc.lower()) for doc in train_texts]

# Train a Word2Vec model on your tokenized sentences
word2vec_model = Word2Vec(tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
Now, let's prepare the Word2Vec embeddings for the input data.

python
Copy
Edit
# Create embeddings matrix
def get_embeddings(sentence):
    tokens = preprocessing.tokenize(sentence.lower())
    vec = []
    for token in tokens:
        if token in word2vec_model.wv:
            vec.append(word2vec_model.wv[token])
    if len(vec) > 0:
        return torch.tensor(np.mean(vec, axis=0))
    else:
        return torch.zeros(100)  # Using zero vector for unknown words

# Tokenize sentences into embeddings
train_embeddings = torch.stack([get_embeddings(s) for s in train_texts])
test_embeddings = torch.stack([get_embeddings(s) for s in test_texts])
1.3. LSTM Model with Word2Vec Embeddings
python
Copy
Edit
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, embeddings):
        lstm_out, _ = self.lstm(embeddings.unsqueeze(1))  # Make it [batch_size, 1, input_dim]
        lstm_out = lstm_out[:, -1, :]  # Get the last hidden state
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out

# Define the model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_w2v = LSTMClassifier(input_dim=100, hidden_dim=128, output_dim=len(df['label'].unique())).to(device)
1.4. Training the LSTM Model with Word2Vec Embeddings
python
Copy
Edit
from sklearn.metrics import accuracy_score
import torch.optim as optim

# Define the criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_w2v.parameters(), lr=0.001)

# Training function
def train_lstm(model, train_data, train_labels, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_data, batch_labels in zip(train_data, train_labels):
            batch_data = batch_data.to(device).float()
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_data):.4f}")

# Train the LSTM model with Word2Vec embeddings
train_lstm(model_w2v, train_embeddings, train_labels)
1.5. Evaluation of Word2Vec Model
python
Copy
Edit
def evaluate_lstm(model, test_data, test_labels):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch_data, batch_labels in zip(test_data, test_labels):
            batch_data = batch_data.to(device).float()
            outputs = model(batch_data)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch_labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

# Evaluate the LSTM model with Word2Vec embeddings
w2v_accuracy = evaluate_lstm(model_w2v, test_embeddings, test_labels)
print(f"Word2Vec LSTM Model Accuracy: {w2v_accuracy * 100:.2f}%")
2. FastText
Now, let's go through the FastText embedding process.

2.1. Loading the Dataset
python
Copy
Edit
# Load the dataset from CSV
df = pd.read_csv('path_to_your_csv_file.csv')

# Split the dataset into train and test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['sentence'], df['label'], test_size=0.2, random_state=42
)
2.2. FastText Embedding Preparation
First, train a FastText model on your tokenized sentences:

python
Copy
Edit
from fasttext import FastText
from sklearn.feature_extraction.text import preprocessing
import nltk
nltk.download('punkt')

# Tokenize the sentences
tokenized_texts = [preprocessing.tokenize(doc.lower()) for doc in train_texts]

# Train a FastText model on your tokenized sentences
fasttext_model = FastText(vector_size=100, window=5, min_count=1, workers=4)
fasttext_model.build_vocab(sentences=tokenized_texts)
fasttext_model.train(sentences=tokenized_texts, total_examples=len(tokenized_texts), epochs=10)
Now, prepare FastText embeddings for the input data:

python
Copy
Edit
def get_fasttext_embeddings(sentence):
    tokens = preprocessing.tokenize(sentence.lower())
    vec = []
    for token in tokens:
        if token in fasttext_model.wv:
            vec.append(fasttext_model.wv[token])
    if len(vec) > 0:
        return torch.tensor(np.mean(vec, axis=0))
    else:
        return torch.zeros(100)  # Using zero vector for unknown words

# Convert sentences into FastText embeddings
train_fasttext_embeddings = torch.stack([get_fasttext_embeddings(s) for s in train_texts])
test_fasttext_embeddings = torch.stack([get_fasttext_embeddings(s) for s in test_texts])
2.3. LSTM Model with FastText Embeddings
python
Copy
Edit
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, embeddings):
        lstm_out, _ = self.lstm(embeddings.unsqueeze(1))  # Make it [batch_size, 1, input_dim]
        lstm_out = lstm_out[:, -1, :]  # Get the last hidden state
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out

# Define the model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft = LSTMClassifier(input_dim=100, hidden_dim=128, output_dim=len(df['label'].unique())).to(device)
2.4. Training the LSTM Model with FastText Embeddings
python
Copy
Edit
from sklearn.metrics import accuracy_score
import torch.optim as optim

# Define the criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.parameters(), lr=0.001)

# Training function
def train_lstm(model, train_data, train_labels, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_data, batch_labels in zip(train_data, train_labels):
            batch_data = batch_data.to(device).float()
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_data):.4f}")

# Train the LSTM model with FastText embeddings
train_lstm(model_ft, train_fasttext_embeddings, train_labels)
2.5. Evaluation of FastText Model
python
Copy
Edit
def evaluate_lstm(model, test_data, test_labels):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch_data, batch_labels in zip(test_data, test_labels):
            batch_data = batch_data.to(device).float()
            outputs = model(batch_data)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch_labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

# Evaluate the LSTM model with FastText embeddings
ft_accuracy = evaluate_lstm(model_ft, test_fasttext_embeddings, test_labels)
print(f"FastText LSTM Model Accuracy: {ft_accuracy * 100:.2f}%")
Summary:
Word2Vec: Used Word2Vec embeddings to represent the sentences and trained the LSTM model accordingly.
FastText: Used FastText embeddings for sentence representations and trained the same LSTM model as above.
