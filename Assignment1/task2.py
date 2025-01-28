from task1 import WordPieceTokenizer
import torch.nn as nn
import torch.utils 
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
class Word2VecDataset(Dataset): # inheriting from the Dataset class present in Pytorch module 
        def __init__(self, window_size, file_path_corpus):
            self.window_size= window_size
            self.tokenizer= WordPieceTokenizer(1000, file_path_corpus)
            self.training_data = []
            self.unk = "<UNK>"
            self.pad= "<PAD>"
            self.sentences=[]
            self.token_to_index={} # dictionary to map the the token to index 
            self.idx_to_token={}
            self.preprocess_data()
        
        def preprocess_data(self):
              self.tokenizer.fit() # this would prepare the vocabulary
              self.vocab = self.tokenizer.vocab
              self.sentences= self.tokenizer.liness
              self.token_to_index["<UNK>"]=0
              self.token_to_index["<PAD>"]=1
              self.idx_to_token[0]="<UNK>"
              self.idx_to_token[1]="<PAD>"
              current_count=2
              for ele in self.vocab:
                    self.token_to_index[ele]=current_count 
                    self.idx_to_token[current_count]=ele
                    current_count= current_count + 1
               
              print("self.sentences is", self.sentences)
              for sentence in self.sentences: 
                    words= sentence.split()
                    print("words is", words)
                    for i in range(len(words)):
                            context=[]
                            #adding the left context of words of length = window
                            for j in range (i -self.window_size, i):
                                    if(j < 0):
                                        context.append(self.pad)
                                    else:
                                        context.append(words[j])
                            #adding the right context of words of length=window
                            for j in range (i + 1, i + 1 + self.window_size):
                                    if(j>=len(words)):
                                        context.append(self.pad)
                                    else:
                                        context.append(words[j])
                                
                            self.training_data.append((context,words[i]))
        def __len__(self):
            return len(self.training_data)
        
        def __getitem__(self, idx):
            context, target = self.training_data[idx]
            context_indices = torch.tensor(
                [self.token_to_index.get(word, self.token_to_index["<UNK>"]) for word in context],
                dtype=torch.long,
            )
            target_index = torch.tensor(
                self.token_to_index.get(target, self.token_to_index["<UNK>"]),
                dtype=torch.long,
            )
            return context_indices, target_index

            
        
              
class Word2VecModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim):
            super(Word2VecModel, self).__init__()
            self.vocab_size= vocab_size
            self.embedding_dim = embedding_dim
            self.embeddings = nn.Embedding(vocab_size, embedding_dim) # this would create a look up table for each token in the vocabulary 
            self.linear=nn.Linear(embedding_dim , vocab_size, bias= False) # a single dense layer of dimension embedding_dim * vocab_size 
        def  forward(self, context):
              embeds = self.embeddings(context)
              avg_embeds= embeds.mean(dim = 1)
              logits= self.linear(avg_embeds)
              return logits
        

#function for training the word2vec model using CBOW approach 
def train(model, epochs, training_data, learning_rate, batch_size=32):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []

    #split the dataset into two parts 
    split = int(0.8 * len(training_data))
    train_data, val_data = torch.utils.data.random_split(training_data, [split, len(training_data) - split])

    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        # Training loop
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

        # Validation loop
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

    # plotting the losses now 
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.plot(range(epochs), val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.show()

    # saving the model now
    torch.save(model.state_dict(), "word2vec_model.pth")

  
def cosine_similarities(model, dataset, word1, word2, word3):
    embeddings = model.embeddings.weight.detach().numpy()
    vec1 = embeddings[dataset.token_to_index[word1]]
    vec2 = embeddings[dataset.token_to_index[word2]]
    vec3 = embeddings[dataset.token_to_index[word3]]

    sim1 = cosine_similarity([vec1], [vec2])[0, 0]
    sim2 = cosine_similarity([vec1], [vec3])[0, 0]
    sim3 = cosine_similarity([vec2], [vec3])[0, 0]
    print(f"Similarity between '{word1}' and '{word2}': {sim1:.4f}")
    print(f"Similarity between '{word1}' and '{word3}': {sim2:.4f}")
    print(f"Similarity between '{word2}' and '{word3}': {sim3:.4f}")

      
               

#the testing part is here

dataset1= Word2VecDataset(2, './corpus.txt')
#now the dataset1 is created 
model1= Word2VecModel(1000, 200)
train(model1, 100, dataset1, 0.1, 32)



              
      
        
                                               
                                



              


              
              
              
            
     

