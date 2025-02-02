# Jai Mata Di
from collections import defaultdict
import json
from collections import defaultdict
import re

class WordPieceTokenizer:
    def __init__(self, vocab_size, corpus_file_ka_path, vocab_file_path=None):
        self.vocab_size = vocab_size
        self.corpus_file_ka_path = corpus_file_ka_path
        self.vocab_file_path = vocab_file_path
        self.vocab = {}
        self.liness = []
        self.corpus = defaultdict(int)
        self.vocab["[PAD]"] = 0  # PAD as spl
        self.vocab["[UNK]"] = 0  # UNK as spl

    def read_karo_corpus(self):
        # self.corpus = defaultdict(int)
        self.liness = []
        with open(self.corpus_file_ka_path, encoding="utf-8") as f:
            for line in f:
                self.liness.append(line.strip())
        self.corpus.clear()
        for i in self.liness:
            for j in i.split():
                j = re.sub(r'[^a-zA-Z0-9]', '', j)
                if not(j.isalnum):
                    continue
                self.corpus[j] += 1
        return self.liness

    def preprocess_data(self):
        self.corpus = defaultdict(int)
        self.corpus.clear()
        #print(self.corpus)
        lines = self.read_karo_corpus()
        #print(self.corpus)
        # removed lower case krne ki glti 
        # hugging face wale ne bhi nhi kiya tha
        self.corpus2 = defaultdict(int)
        for line in lines:
            line = re.sub(r'\s+', ' ', line).strip()
            line = re.sub(r'([,.])', r' \1 ', line)
            tokens = line.split()
            for token in tokens:
                # Agar token punctuation hai ya single character, use as tuple
                if len(token) == 1 or re.fullmatch(r'[,.]', token):
                    token_tuple = (token,)
                else:
                    # First character as is, rest with ## pel do 
                    # yhi glti thi 
                    # commenting ki yaaad rhe 
                    token_tuple = (token[0],) + tuple("##" + ch for ch in token[1:])
                self.corpus2[token_tuple] += 1

    def get_score(self, pair, pair_count,lopa):
        # 1 lagaya for non zero division 
        first, second = pair
        score = pair_count / (lopa.get(first)*lopa.get(second))
        return score

    def write_vocabulary(self, group_no):
        with open(f"vocabulary_{group_no}.txt", "w", encoding="utf-8") as f:
            for token in sorted(self.vocab.keys()):
                f.write(token + "\n")

    def merge_pair_in_word(self, word, pair):
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                if i == 0:
                    merged = word[i] + (word[i+1][2:] if word[i+1].startswith("##") else word[i+1])
                else:
                    merged = "##" + (word[i][2:] if word[i].startswith("##") else word[i]) + (word[i+1][2:] if word[i+1].startswith("##") else word[i+1])
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)
    def merge_rish(self,pair,addd,f1,f2,spl_das):
        # print(f1,f2)
        # print(pair,addd)
        # print("##"*10)
        # print(spl_das
        for i,j in self.corpus.items():
            timo = spl_das[i].copy()
            if len(timo) == 1:
                continue
            k = 0
            while k < len(timo)-1:
                if timo[k] == pair[0]  and timo[k+1] == pair[1]:
                    gait = (pair[0]+pair[1][2::] if pair[1].startswith("##") else pair[0]+pair[1])
                    timo = timo[:k]+ [gait] + timo[k+2:]
                else:
                    k += 1
            spl_das[i] = timo
        # print(spl_das)
        return spl_das



    # def scr_finder(self,)
    def construct_vocabulary(self):
        # self.preprocess_data()
        # Initialize vocabulary with individual tokens from corpus
        token_freq = defaultdict(int)
        #print(self.corpus)
        for word, freq in self.corpus.items():
            for j in range(len(word)):
                if j == 0:
                    token_freq[word[j]] += freq
                    self.vocab[word[j]] = 1
                else:
                    token_freq["##"+word[j]] += freq
                    self.vocab["##"+word[j]] = 1  
        spl_das = defaultdict(list)
        for i in self.corpus:
            for j in range(len(i)):
                if j == 0:
                    spl_das[i].append(i[j])
                else:
                    spl_das[i].append("##"+i[j])
        f1,f2 = defaultdict(int),defaultdict(int)
        for i,j in self.corpus.items():
           # print(i,j)
            stp = spl_das[i]
            if len(stp) == 1:
                f1[stp[0]] += j
            else:
                # print(stp)
                for k in range(len(stp)-1):
                    peakock = (stp[k],stp[k+1])
                    f1[stp[k]] += j
                    f2[peakock] += j
                f1[stp[-1]] += j
                # last char ko miss kiya tha to daalana padega nhi to back 
        # print(f1)
        ans_comp = defaultdict(int)
        for i,j in f2.items():
            score = self.get_score(i,j,f1)
            # print(i,j,score)
            ans_comp[i] = score
        # self.vocab = {}
        self.vocab["[PAD]"] = 0
        self.vocab["[UNK]"] = 0
        # for 
        # print(len(self.vocab))
        # exit(1)
        while len(self.vocab) < self.vocab_size:
            # print(len(self.vocab))
            # pairs = defaultdict(int)
            f1,f2 = defaultdict(int),defaultdict(int)
            for i,j in self.corpus.items():
                stp = spl_das[i]
                if len(stp) == 1:
                    f1[stp[0]] += j
                else:
                    for k in range(len(stp)-1):
                        peakock = (stp[k],stp[k+1])
                        f1[stp[k]] += j
                        f2[peakock] += j
                    f1[stp[-1]] += j
                    # last char ko miss kiya tha to daalana padega nhi to back 
            ans_comp = defaultdict(int)
            for i,j in f2.items():
                score = self.get_score(i,j,f1)
                # print(i,j,score)
                ans_comp[i] = score
            best_pair = max(ans_comp.keys(), key=lambda p: ans_comp[p])
            addd = ans_comp[best_pair]
            # print(best_pair,addd)
            # exit(0)
            spl_das = self.merge_rish(best_pair,addd,f1,f2,spl_das)
            tmp_add = ""
            if best_pair[1].startswith("##"):
                tmp_add = best_pair[0]+best_pair[1][2::]
            else:
                tmp_add = best_pair[0]+best_pair[1]
            self.vocab[tmp_add]=1
    def load_vocab(self):
        with open(self.vocab_file_path, 'r', encoding="utf-8") as f:
            for line in f:
                token = line.strip()
                self.vocab[token] = 1  # We set frequency to 1 initially as we are only loading the vocabulary

    def tokenize(self, s):
        # test krne ke liye
        s = s.strip()
        s = re.sub(r'([,.])', r' \1 ', s)
        s = re.sub(r'\s+', ' ', s)
        # help from Tutorial jo thi
        tokens = s.split()
        ans_dp = []
        for i in tokens:
            start = []
            # print(i)
            fck = 0
            while len(i) > 0:
                ris = len(i)
                while ris > 0 and i[:ris] not in self.vocab: # longest match krunga 
                    # print(i[:ris],i[:ris] in self.vocab)
                    ris -= 1
                # print(ris)
                #print(i[:ris] in self.vocab,i[:ris],self.vocab)
                if ris == 0:
                    ans_dp.append("[UNK]")
                    fck = 1
                    break
                start.append(i[:ris])
                i = i[ris::]
                if len(i) > 0:
                    i = "##"+i
            else:
                ans_dp.extend(start)
            # if fck == 1:
            #     if len(start) > 0:
            #         ans_dp.extend(start)
        return ans_dp


    def json_formatter(self, data, group_no:int):
        final_output = {}

        for item in data:
            tokens = self.tokenize(item['sentence'])
            final_output[item['id']] = tokens

        with open(f'tokenized_{group_no}.json', 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)

        print(f"Tokenized data saved to tokenized_{group_no}.json")


    def fit(self):
        self.read_karo_corpus()
        self.preprocess_data()
        self.construct_vocabulary()
# def test():
#     a = WordPieceTokenizer(1000,r'corpus.txt')
#     (a.read_karo_corpus())
#     # print(a.corpus)
#     (a.preprocess_data())
#     # print(a.corpus)
#     # print("{}{}{}")
#     q = (a.construct_vocabulary())
#     print(a.tokenize("i mA x7 boy"))
#     # print(sorted(q,key=len))



# test()

# def test2():
#     a = WordPieceTokenizer(2, 'Assignment1\corpus.txt', 'vocabulary_5.txt')
#     # a.load_vocab()  # Load vocabulary from file
#     a.read_karo_corpus()
#     a.construct_vocabulary()
#     q = (a.tokenize("This is an example sentence for tokenization!"))
#     print(q)
#     # for i in a.vocab:
#     #     print(i,a.vocab[i])

# test2()