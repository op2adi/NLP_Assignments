# Jai Mata Di
import numpy as np
import pandas as pd

class WordPieceTokenizer:
    def __init__(self, vocab_size, corpus_file_ka_path):
        self.vocab_size = vocab_size
        self.corpus_file_ka_path = corpus_file_ka_path
        self.vocab = {}
        self.liness = []
        self.corpus = {}

    def read_karo_corpus(self):
        with open(self.corpus_file_ka_path) as f:
            for line in f:
                self.liness.append(line.strip())
        return self.liness

    def preprocess_data(self):
        # Process each line in corpus
        for line in self.liness:
            # Convert to lowercase and remove extra whitespace
            line = line.lower().strip()
            # Remove punctuation and special characters, keeping only letters and spaces
            line = ''.join(char for char in line if char.isalnum() or char.isspace())
            # Split into words
            words = line.split()
            # Count word frequencies
            for word in words:
                if word in self.corpus:
                    self.corpus[word] += 1
                else:
                    self.corpus[word] = 1

    def construct_vocabulary(self):
        # Initialize subword frequencies
        subword_freq = {}
        for word, freq in self.corpus.items():
            for i in range(len(word)):
                if i == 0:
                    subword = word[i]
                else:
                    subword = "##" + word[i]

                if subword in subword_freq:
                    subword_freq[subword] += freq
                else:
                    subword_freq[subword] = freq

        for char, freq in subword_freq.items():
            if len(char) == 1 or char.startswith("##"):
                self.vocab[char] = freq
            else:
                print("maro merko")
                print(char+":"+ freq)

        while len(self.vocab) < self.vocab_size:
            # Merge the most frequent subwords to form new tokens
            # print(len(self.vocab))
            candidates = {}
            for word, freq in self.corpus.items():
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
                    pair = (subwords[k], subwords[k + 1])
                    if pair in candidates:
                        candidates[pair] += freq
                    else:
                        candidates[pair] = freq

            # Calculate pair scores and choose the best candidate
            pair_scores = {}
            for pair, freq in candidates.items():
                first, second = pair
                score = freq / (self.vocab.get(first, 1) * self.vocab.get(second, 1))
                pair_scores[pair] = score

            best_candidate = max(pair_scores, key=pair_scores.get, default=None)
            print("best_candidate", best_candidate)

            if best_candidate is None:
                print("maro merko 2")
                break

            # Add the new merged token to the vocabulary
            merged_token = best_candidate[0] + best_candidate[1][2:]  # Merge without "##" in the second subword
            self.vocab[merged_token] = candidates[best_candidate]

            # Update the corpus with the merged token
            new_corpus = {}
            for word, freq in self.corpus.items():
                new_word = word.replace(best_candidate[0] + best_candidate[1][2:], merged_token)
                new_corpus[new_word] = freq
            self.corpus = new_corpus

            print("best_candidate", merged_token)

        return self.vocab

def test():
    a = WordPieceTokenizer(1000, 'Assignment1\corpus.txt')
    print(a.read_karo_corpus())
    print(a.preprocess_data())
    print(a.construct_vocabulary())

test()
