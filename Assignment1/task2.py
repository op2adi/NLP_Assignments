from task1 import WordPieceTokenizer
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

class Word2VecDataset(Dataset):
    def __init__(self, window_size, file_path_corpus, vocab_size):
        self.window_size = window_size
        self.tokenizer = WordPieceTokenizer(vocab_size, file_path_corpus)
        self.training_data = []
        self.unk = "<UNK>"
        self.pad = "<PAD>"
        self.sentences = []
        self.token_to_index = {}  # stores token to index mapping
        self.idx_to_token = {}    # stores index to token mapping
        self.preprocess_data()

    def preprocess_data(self):
        self.tokenizer.fit()  # prepares the vocab
        self.vocab = self.tokenizer.vocab
        self.sentences = self.tokenizer.liness

        self.token_to_index[self.unk] = 0
        self.token_to_index[self.pad] = 1
        self.idx_to_token[0] = self.unk
        self.idx_to_token[1] = self.pad

        current_count = 2
        for ele in self.vocab:
            if ele not in self.token_to_index:
                self.token_to_index[ele] = current_count
                self.idx_to_token[current_count] = ele
                current_count += 1

        # make the training data 
        for sentence in self.sentences:
            words = self.tokenizer.tokenize(sentence)
            for i in range(len(words)):
                context = []

                # create the left context
                for j in range(i - self.window_size, i):
                    if(j>=0 and (words[j] not in self.vocab)):
                        context.append(self.unk)
                        continue
                    context.append(words[j] if j >= 0 else self.pad)

                # create the right context 
                for j in range(i + 1, i + 1 + self.window_size):
                    if(j<len(words) and (words[j] not in self.vocab)):
                        context.append(self.unk)
                        continue
                    context.append(words[j] if j < len(words) else self.pad)

                self.training_data.append((context, words[i]))

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        context, target = self.training_data[idx]
        context_indices = torch.tensor(
            [self.token_to_index.get(word, self.token_to_index[self.unk]) for word in context],
            dtype=torch.long
        )
        target_index = torch.tensor(
            self.token_to_index.get(target, self.token_to_index[self.unk]),
            dtype=torch.long
        )
        return context_indices, target_index


class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, window_size=2):
        super(Word2VecModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim*window_size*2, vocab_size, bias=False) #input will be 2*window_size*(tensors)

    def forward(self, context):
        embeds = self.embeddings(context)  # shape will be -  (batch_size,2*window_size, embedding_dim)
        concat_embeds = embeds.view(embeds.shape[0], -1)  # shape to the (batch_size, 2*window_size*embedding_dim)
        logits = self.linear(concat_embeds)
        return torch.log_softmax(logits, dim=1)  

def train(model, epochs, training_data, learning_rate, batch_size=32):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []

    #split the training and the validation dataset into 80 : 20 ration 
    split = int(0.8 * len(training_data))
    train_data, val_data = torch.utils.data.random_split(training_data, [split, len(training_data) - split])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

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

    # Plot losses
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.plot(range(epochs), val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.show()
    #saving the model 
    torch.save(model.state_dict(), "word2vec_model.pth")


def cosine_similarities(model, dataset, word1, word2, word3):
    embeddings = model.embeddings.weight.detach().numpy()
    vec1 = embeddings[dataset.token_to_index.get(word1, 0)]
    vec2 = embeddings[dataset.token_to_index.get(word2, 0)]
    vec3 = embeddings[dataset.token_to_index.get(word3, 0)]

    sim1 = cosine_similarity([vec1], [vec2])[0, 0]
    sim2 = cosine_similarity([vec1], [vec3])[0, 0]
    sim3 = cosine_similarity([vec2], [vec3])[0, 0]

    print(f"Similarity between '{word1}' and '{word2}': {sim1:.4f}")
    print(f"Similarity between '{word1}' and '{word3}': {sim2:.4f}")
    print(f"Similarity between '{word2}' and '{word3}': {sim3:.4f}")


# testing code is here 
#dataset1 = Word2VecDataset(2, './corpus.txt', 5002) # vocab size would be 5002 = 5000  +  2, for two special tokens 
#model1 = Word2VecModel(len(dataset1.token_to_index), 10,2)  
#train(model1, 10, dataset1, 0.01, 32)
#cosine_similarities(model1, dataset1, "i", "feel", "so")


              


              
              
              
            
     

