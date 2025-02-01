import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from task1 import WordPieceTokenizer
from task2 import Word2VecModel, Word2VecDataset
import numpy as np
import matplotlib.pyplot as plt

class NeuralLMDataset(Dataset):
    def __init__(self, corpus_path,vocab_size, context_size=3):
        self.context_size = context_size
        self.word2vec_model = Word2VecModel(vocab_size, 10,2)  # Ensure correct model parameters
        self.word2vec_model.load_state_dict(torch.load("word2vec_model.pth"))
        self.word2vec_model.eval()  # Set to evaluation mode
        self.tokenizer = WordPieceTokenizer(vocab_size, corpus_path)
        self.tokenizer.fit()
        self.unk = "<UNK>"
        self.pad = "<PAD>"
        self.sentences = self.tokenizer.liness
        self.embeddings = self.word2vec_model.embeddings.weight.detach()
        self.token_to_index = {self.unk: 0, self.pad: 1}
        self.index_to_token = {0: self.unk, 1: self.pad}
        self.preprocess_data()

    def preprocess_data(self):
        self.tokenized_sentences = [self.tokenizer.tokenize(sentence) for sentence in self.sentences]
        self.create_training_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_indices = torch.tensor(
            [self.token_to_index.get(word, self.token_to_index[self.unk]) for word in context], dtype=torch.long
        )
        target_index = torch.tensor(
            self.token_to_index.get(target, self.token_to_index[self.unk]), dtype=torch.long
        )
        return context_indices, target_index

    def create_training_data(self):
        self.data = []
        for sentence in self.tokenized_sentences:
            if len(sentence) < self.context_size + 1:
                continue
            for i in range(len(sentence) - self.context_size):
                context = sentence[i:i + self.context_size]
                target = sentence[i + self.context_size]
                self.data.append((context, target))

class NeuralLM1(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, context_size, pretrained_embeddings):
        super(NeuralLM1, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        self.fc1 = nn.Linear(embedding_dim * context_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x).view(x.shape[0], -1)
        hidden = self.relu(self.fc1(embeds))
        out = self.fc2(hidden)
        return out

def train1(model, train_data, val_data, epochs, learning_rate, batch_size=32):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    
    for epoch in range(epochs):
        model.train()
        training_loss, correct_train, total_train = 0, 0, 0
        
        for context, target in train_loader:
            optimizer.zero_grad()
            predictions = model(context)
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            correct_train += (torch.argmax(predictions, dim=1) == target).sum().item()
            total_train += target.size(0)
        
        train_losses.append(training_loss / len(train_loader))
        train_accuracies.append(correct_train / total_train)
        
        model.eval()
        val_loss, correct_val, total_val = 0, 0, 0
        with torch.no_grad():
            for context, target in val_loader:
                predictions = model(context)
                loss = criterion(predictions, target)
                val_loss += loss.item()
                correct_val += (torch.argmax(predictions, dim=1) == target).sum().item()
                total_val += target.size(0)
        
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(correct_val / total_val)
        
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_losses[-1]:.8f}, Validation Loss: {val_losses[-1]:.8f}, Training Accuracy: {train_accuracies[-1]:.8f}, Validation Accuracy: {val_accuracies[-1]:.8f}")
    
    plt.plot(range(epochs), train_losses, label="Train Accuracy")
    plt.plot(range(epochs), val_losses, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def predict_next_tokens(model, dataset, test_file, num_predictions=3):
    model.eval()
    with open(test_file, 'r') as f:
        test_sentences = [line.strip() for line in f.readlines()]
    
    for sentence in test_sentences:
        tokens = dataset.tokenizer.tokenize(sentence) 
        context = tokens[-3:]
        context_indices = torch.tensor([
            dataset.token_to_index.get(word, dataset.token_to_index[dataset.unk]) for word in context
        ], dtype=torch.long).unsqueeze(0)
        
        predictions = model(context_indices)
        top_tokens = torch.topk(predictions, num_predictions, dim=1).indices.squeeze(0).tolist()
        predicted_words = [dataset.index_to_token[idx] for idx in top_tokens]
        print(f"Input: {sentence}\nPredicted next words: {predicted_words}\n")
def compute_accuracy(model, data_loader):
    correct, total = 0, 0
    with torch.no_grad():
        for context, target in data_loader:
            predictions = model(context)
            correct += (torch.argmax(predictions, dim=1) == target).sum().item()
            total += target.size(0)
    return correct / total

def compute_perplexity(model, data_loader):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for context, target in data_loader:
            predictions = model(context)
            loss = criterion(predictions, target)
            total_loss += loss.item() * target.size(0)
            total_samples += target.size(0)
    return np.exp(total_loss / total_samples)
# Load dataset and train model
dataset = NeuralLMDataset("./corpus.txt",5004, context_size=3)
vocab_size = len(dataset.token_to_index)
split = int(0.8 * len(dataset))
train_data, val_data = torch.utils.data.random_split(dataset, [split, len(dataset) - split])

hidden_dim = 128
model = NeuralLM1(5004,10, hidden_dim,3, dataset.embeddings)
train1(model, train_data, val_data, epochs=5, learning_rate=0.01)
print("Train accuracy is", compute_accuracy(model, train_data))
print("Validation accuracy is", compute_accuracy(model, val_data))
print("Train perplexity score is", compute_perplexity(model, train_data))
print("Val perplexity score is", compute_perplexity(model, val_data))
# Predict next tokens
predict_next_tokens(model, dataset, "sample_test.txt")
