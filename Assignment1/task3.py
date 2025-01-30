import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from task1 import WordPieceTokenizer
from task2 import Word2VecModel
import numpy as np

# Dataset for Neural LM
class NeuralLMDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, word2vec_model, context_size=3):
        self.context_size = context_size
        self.tokenizer = tokenizer
        self.word2vec_model = word2vec_model
        self.unk = "<UNK>"
        self.pad = "<PAD>"
        
        # Load and tokenize corpus
        with open(corpus_path, 'r', encoding='utf-8') as file:
            self.sentences = [line.strip() for line in file.readlines()]
        self.tokenized_sentences = [self.tokenizer.tokenize(sentence) for sentence in self.sentences]
        
        self.token_to_index = self.tokenizer.vocab
        self.idx_to_token = {idx: token for token, idx in self.token_to_index.items()}
        self.data = self.create_training_data()

    def create_training_data(self):
        training_data = []
        for sentence in self.tokenized_sentences:
            for i in range(len(sentence) - self.context_size):
                context = sentence[i:i + self.context_size]
                target = sentence[i + self.context_size]
                training_data.append((context, target))
        return training_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_indices = torch.tensor([self.token_to_index.get(word, self.token_to_index[self.unk]) for word in context], dtype=torch.long)
        target_index = torch.tensor(self.token_to_index.get(target, self.token_to_index[self.unk]), dtype=torch.long)
        return context_indices, target_index


# Neural LM Architectures
class NeuralLM1(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(NeuralLM1, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 3, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embeds = self.embedding(x).view(x.shape[0], -1)
        hidden = self.relu(self.fc1(embeds))
        out = self.fc2(hidden)
        return out

class NeuralLM2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(NeuralLM2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embeds = self.embedding(x).view(x.shape[0], -1)
        hidden = self.relu(self.fc1(embeds))
        hidden = self.relu(self.fc2(hidden))
        out = self.fc3(hidden)
        return out

class NeuralLM3(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(NeuralLM3, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embeds = self.embedding(x).view(x.shape[0], -1)
        hidden = self.tanh(self.fc1(embeds))
        hidden = self.dropout(self.tanh(self.fc2(hidden)))
        out = self.fc3(hidden)
        return out

# Training Function
def train(model, dataset, epochs, learning_rate, batch_size=32):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# Load dataset and models
tokenizer = WordPieceTokenizer(5000, "./corpus.txt")
word2vec = Word2VecModel(5002, 10)
dataset = NeuralLMDataset("./corpus.txt", tokenizer, word2vec, context_size=3)
# Accuracy and Perplexity Functions
def compute_accuracy(model, data_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for context, target in data_loader:
            output = model(context)
            pred = torch.argmax(output, dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total

def compute_perplexity(model, data_loader, criterion):
    model.eval()
    total_loss, total_words = 0, 0
    with torch.no_grad():
        for context, target in data_loader:
            output = model(context)
            loss = criterion(output, target)
            total_loss += loss.item()
            total_words += target.size(0)
    return np.exp(total_loss / total_words)

# Initialize models
vocab_size = len(dataset.token_to_index)
embedding_dim = 10
hidden_dim = 128

model1 = NeuralLM1(vocab_size, embedding_dim, hidden_dim)
model2 = NeuralLM2(vocab_size, embedding_dim, hidden_dim)
model3 = NeuralLM3(vocab_size, embedding_dim, hidden_dim)

# Train models
train(model1, dataset, epochs=10, learning_rate=0.001)
train(model2, dataset, epochs=10, learning_rate=0.001)
train(model3, dataset, epochs=10, learning_rate=0.001)
