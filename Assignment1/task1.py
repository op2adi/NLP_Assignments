# Jai Mata Di
import numpy as np
import pandas as pd

class WordPieceTokenizer:
    def __init__(self, vocab_size, corpus_file_ka_path):
        self.vocab_size = vocab_size
        self.corpus_file_ka_path = corpus_file_ka_path
        self.vocab = {}
        self.corpus = []
        self.word_freq = {}

    def read_karo_corpus(self):
        with open(self.corpus_file_ka_path) as f:
            for line in f:
                self.corpus.append(line.strip())
        return self.corpus

    def preprocess_data(self):
        # Process each line in corpus
        for line in self.corpus:
            # Convert to lowercase and remove extra whitespace
            line = line.lower().strip()
            # Remove punctuation and special characters, keeping only letters and spaces
            line = ''.join(char for char in line if char.isalnum() or char.isspace())
            # Split into words
            words = line.split()
            # Count word frequencies
            for word in words:
                if word in self.word_freq:
                    self.word_freq[word] += 1
                else:
                    self.word_freq[word] = 1

    def construct_vocabulary(self):
        # Initialize subword frequencies
        subword_freq = {}
        for word, freq in self.word_freq.items():
            for i in range(len(word)):
                if i == 0:
                    subword = word[i]
                else:
                    subword = "##" + word[i]
                if subword in subword_freq:
                    subword_freq[subword] += freq
                else:
                    subword_freq[subword] = freq

        # Start with a base vocabulary of single characters and subwords
        self.vocab = {char: freq for char, freq in subword_freq.items() if len(char) == 1 or char.startswith("##")}

        while len(self.vocab) < self.vocab_size:
            # Merge the most frequent subwords to form new tokens
            # print(len(self.vocab))
            candidates = {}
            for word, freq in self.word_freq.items():
                subwords = []
                i = 0
                while i < len(word):
                    for j in range(len(word), i, -1):
                        candidate = word[i:j]
                        if i > 0:
                            candidate = "##" + candidate
                        if candidate in self.vocab:
                            subwords.append(candidate)
                            i = j - 1
                            break
                    i += 1
                for k in range(len(subwords) - 1):
                    merge = subwords[k] + subwords[k+1]
                    # print(merge in self.vocab)
                    if merge in candidates:
                        candidates[merge] += freq
                    else:
                        candidates[merge] = freq

            # Find the best candidate to add to the vocabulary
            best_candidate = max(candidates, key=candidates.get, default=None)
            if best_candidate is None:
                break
            self.vocab[best_candidate] = candidates[best_candidate]
            #now remove the subwords from the vocab which are part of the best_candidate and update the word_freq and also merge the subwords in the word_freq

            for word in self.word_freq:
                if best_candidate in word:
                    new_word = word.replace(best_candidate, best_candidate[2:])
                    self.word_freq[new_word] = self.word_freq[word]
                    del self.word_freq[word]
            print(best_candidate)

        return self.vocab

def test():
    a = WordPieceTokenizer(1000,'Assignment1\corpus.txt')
    # a.___init__()
    print(a.read_karo_corpus())
    print(a.preprocess_data())
    print(a.construct_vocabulary())

test()

#jb bhi run karna ho toh test() ko uncomment kar dena
