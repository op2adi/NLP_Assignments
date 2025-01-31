import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from task1 import WordPieceTokenizer
from task2 import Word2VecModel, Word2VecDataset
import numpy as np

class NeuralLMDataset(Dataset):
    def __init__(self, corpus_path, embedding_dim, context_size=3):
        self.context_size = context_size
        self.word2vec_model = Word2VecModel(vocab_size, embedding_dim, 2)
        self.word2vec_model.load_state_dict(torch.load("word2vec_model.pth"))#now the saved word2vec model is used for generating the embeddings
        self.tokenizer= WordPieceTokenizer(vocab_size, corpus_path)
        tokenizer.fit()
        self.unk = "<UNK>"
        self.pad = "<PAD>"
        self.sentences= self.tokenizer.liness
        self.embeddings= self.word2vec_model.embeddings
        self.token_to_index = {}
        self.index_to_token={}
        
    def preprocess_data(self):
        self.tokenized_sentences = [self.tokenizer.tokenize(sentence) for sentence in self.sentences]
        self.token_to_index[self.unk]=0
        self.token_to_index[self.pad]=1
        self.index_to_token[0]=self.unk
        self.index_to_token[1]=self.pad
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
        training_data = []
        
        for sentence in self.tokenized_sentences:
            if len(sentence) < self.context_size + 1:
                continue  # Skip short sentences
            
            for i in range(len(sentence) - self.context_size):
                context = sentence[i:i+self.context_size]
                target = sentence[i+self.context_size]
                training_data.append((context, target))
        
        return training_data


# First neural architecture 
class NeuralLM1(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, context_size):
        super(NeuralLM1, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * context_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embeds = self.embedding(x).view(x.shape[0], -1)
        hidden = self.relu(self.fc1(embeds))
        out = self.fc2(hidden)
        return out

class NeuralLM2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, context_size):
        super(NeuralLM2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * context_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embeds = self.embedding(x).view(x.shape[0], -1)
        hidden = self.relu(self.fc1(embeds))
        hidden = self.relu(self.fc2(hidden))
        out = self.fc3(hidden)
        return out
#This architecture implements dropout 
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
def train(model, train_data,val_data,  epochs, learning_rate, batch_size=32):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader= DataLoader(val_data, batch_size=batch_size, shuffle=True)
    train_losses=[]
    val_losses=[]
    train_accuracy= []
    val_accuracy=[]
    for epoch in range(epochs):
        model.train()
        training_loss = 0

        for lst, target in train_loader:
            optimizer.zero_grad()
            predictions = model(lst)
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        avg_training_loss = training_loss / len(train_loader)
        train_losses.append(avg_training_loss)

        # Validation
        model.eval()
        val_current_loss = 0
        with torch.no_grad():
            for lst, target in val_loader:
                predictions = model(lst)
                loss = criterion(predictions, target)
                val_current_loss += loss.item()

        avg_val_loss = val_current_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_training_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")


tokenizer = WordPieceTokenizer(5000, "./corpus.txt")
word2vec = Word2VecModel(5002, 10)
dataset = NeuralLMDataset("./corpus.txt", tokenizer, word2vec, context_size=3)
test_data= NeuralLMDataset("./sample_test.txt",tokenizer, word2vec, context_size=3)
split = int(0.8 * len(dataset))
train_data, val_data = torch.utils.data.random_split(dataset, [split, len(dataset) - split])
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

# intializing the models 
vocab_size = len(dataset.token_to_index)
embedding_dim = 10
hidden_dim = 128

model1 = NeuralLM1(vocab_size, embedding_dim, hidden_dim)
model2 = NeuralLM2(vocab_size, embedding_dim, hidden_dim)
model3 = NeuralLM3(vocab_size, embedding_dim, hidden_dim)

# training the model 
train(model1, dataset, epochs=10, learning_rate=0.01)
train(model2, dataset, epochs=10, learning_rate=0.01)
train(model3, dataset, epochs=10, learning_rate=0.01)
print("Accuracy of model 1 on test dataset is ", compute_accuracy(model1, test_data))
print("Accuracy of model 1 on train dataset is", compute_accuracy(model1, dataset))
print("Accuracy of model2 on test dataset is", compute_perplexity(model2, test_data))
print("Accuracy of model2 on train dataset is",compute_perplexity(model2,dataset) )
print("Accuracy of model3 on test dataset is", compute_accuracy(model3, test_data))
print("Accuracy of model3 on train dataset is", compute_accuracy(model3, dataset))