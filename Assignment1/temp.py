# Jai Mata Di
import json
import numpy as np

class WordPieceTokenizer:
    def __init__(self, vocab_size, corpus_file_path):
        self.vocab_size = vocab_size
        self.corpus_file_path = corpus_file_path
        self.vocab = {}
        self.corpus = []
        self.word_freq = {}

    def read_corpus(self):
        """Reads the corpus from the specified file."""
        with open(self.corpus_file_path, 'r') as f:
            self.corpus = [line.strip() for line in f if line.strip()]
        return self.corpus

    def preprocess_data(self):
        """Processes the corpus: lowercase, remove special chars, and calculate word frequencies."""
        for line in self.corpus:
            line = line.lower().strip()
            # Keep only alphanumeric characters and spaces
            line = ''.join(char for char in line if char.isalnum() or char.isspace())
            # Split into words
            words = line.split()
            # Count word frequencies
            for word in words:
                self.word_freq[word] = self.word_freq.get(word, 0) + 1

    def construct_vocabulary(self):
        """Constructs the vocabulary using the WordPiece algorithm."""
        # Initialize vocabulary with single characters
        subword_freq = {}
        for word, freq in self.word_freq.items():
            for i in range(len(word)):
                if i == 0:
                    subword = word[i]
                else:
                    subword = "##" + word[i]
                subword_freq[subword] = subword_freq.get(subword, 0) + freq

        for char, freq in subword_freq.items():
            if len(char) == 1 or char.startswith("##"):
                self.vocab[char] = freq
            else:
                print("maro merko")
                print(char+":"+ freq)

        while len(self.vocab) < self.vocab_size:
            # Find the most frequent subword pairs
            pairs = {}
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
                    pair = subwords[k] + subwords[k + 1]
                    pairs[pair] = pairs.get(pair, 0) + freq

            # Add the best pair to the vocabulary
            best_pair = max(pairs, key=pairs.get, default=None)
            if best_pair is None:
                break
            self.vocab[best_pair] = pairs[best_pair]

            # Update word frequencies
            updated_word_freq = {}
            for word, freq in self.word_freq.items():
                merged_word = word.replace(best_pair.replace("##", ""), best_pair)
                updated_word_freq[merged_word] = updated_word_freq.get(merged_word, 0) + freq
            self.word_freq = updated_word_freq

    def tokenize(self, sentence):
        """Tokenizes a sentence using the constructed vocabulary."""
        sentence = ''.join(char for char in sentence.lower().strip() if char.isalnum() or char.isspace())
        tokens = []
        while sentence:
            match = None
            for i in range(len(sentence), 0, -1):
                sub = sentence[:i]
                if sub in self.vocab:
                    match = sub
                    break
            if match:
                tokens.append(match)
                sentence = sentence[len(match):]
            else:
                tokens.append("[UNK]")
                break
        return tokens

    def write_vocabulary(self, group_no):
        """Writes the vocabulary to a file."""
        with open(f"vocabulary_{group_no}.txt", "w") as f:
            for token in sorted(self.vocab.keys()):
                f.write(token + "\n")

    def generate_tokenized_output(self, test_file, group_no):
        """Generates tokenized output for test sentences."""
        with open(test_file, 'r') as f:
            test_data = json.load(f)

        tokenized_data = {}
        for item in test_data:
            tokenized_data[item["id"]] = self.tokenize(item["sentence"])

        with open(f"tokenized_{group_no}.json", "w") as f:
            json.dump(tokenized_data, f, indent=4)


# Test the implementation
def test():
    group_no = 123  # Replace with your group number
    tokenizer = WordPieceTokenizer(vocab_size=1000, corpus_file_path="corpus.txt")
    tokenizer.read_corpus()
    tokenizer.preprocess_data()
    tokenizer.construct_vocabulary()
    tokenizer.write_vocabulary(group_no)
    tokenizer.generate_tokenized_output("sample_test.json", group_no)

test()
