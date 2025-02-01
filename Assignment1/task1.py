# Jai Mata Di
import numpy as np
import pandas as pd
from collections import Counter
import os
from collections import defaultdict
import time
class WordPieceTokenizer:
    def __init__(self, vocab_size, corpus_file_ka_path, vocab_file_path = None):
        self.vocab_size = vocab_size
        self.corpus_file_ka_path = corpus_file_ka_path
        self.vocab_file_path = vocab_file_path
        self.vocab = {}
        self.liness = []
        self.corpus = defaultdict(int)
        self.unk="<UNK>" #used as a token for something which is not there currently 
        self.pad="<PAD>"

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
            
            words = line.split()
            for word in words:
                self.corpus[word] += 1
    def get_score(self,pair,val):
        # 1 lagaya for non zero division 
        first, second = pair
        score = val/(self.vocab.get(first, 1) * self.vocab.get(second, 1))
        return score
    def write_vocabulary(self, group_no):
        with open(f"vocabulary_{group_no}.txt", "w") as f:
            for token in sorted(self.vocab.keys()):
                f.write(token + "\n")

    def update_corpus_spl(self,abhi_tk_best):
        new = defaultdict(int)
        # print(self.corpus)
        # time.sleep(5)
        for i,j in self.corpus.items():
            if type(i) == tuple: 
                continue
            np_i = i.replace(abhi_tk_best[0]+abhi_tk_best[1][2::],abhi_tk_best[0]+abhi_tk_best[1][2::])
            new[np_i] = j
        # print(new)
        return new


    def construct_vocabulary(self):
        # Initialize subword frequencies
        ans_tmp = defaultdict(int)
        for i,j in self.corpus.items():
            # print(i,j)
            if type(i) == tuple:
                continue
            for k in range(len(i)):
                if k == 0:
                    ans_tmp[i[k]] += j
                else:
                    ans_tmp["##" + i[k]] += j
        for i,j in ans_tmp.items():
            # if len(i) == 3:
            #     print(i)
            #     exit(0)
            # print(i,j)
            # tmp print check ke liye 
            self.vocab[i] = j
        # print(self.vocab)
        while len(self.vocab) < self.vocab_size:
            # print(len(self.vocab)) # len same aa rhi issue hai
            # hr baar ek new token add krna hai 
            tmp = defaultdict(int)
            for i,j in self.corpus.items():
                # print(type(i))
                out_words =[]
                start_old = 0
                while start_old < len(i):
                    for new_tmp in range(len(i),start_old,-1):
                        # print(i[start_old:new_tmp])
                        tmp2 = i[start_old:new_tmp]
                        # original initializa krke rkha hai 
                        if start_old != 0:
                            tmp2 = "##"+i[start_old:new_tmp]
                        if tmp2 in self.vocab:
                            out_words.append(tmp2)
                            start_old = new_tmp-1
                            break
                    start_old += 1  
                for k in range(len(out_words)-1):
                    tmp[(out_words[k],out_words[k+1])] += j
                # if 
            # print(tmp)
            final_dp = {} # to hold the aakhri score on which  comparisons krenge 
            for utk,arp in tmp.items():
                score = self.get_score(utk,arp)
                # print(score)
                final_dp[utk] = score
            for i,j in self.corpus.items():
                if type(i) == tuple:
                    continue
            abhi_tk_best = max(final_dp,key = lambda x:final_dp[x],default = None)
            # print(abhi_tk_best,"KLKLL")
            if abhi_tk_best is None:
                # not possible iska mtlb corpus se bda vocab size manga hai 
                break
            if "##" not in abhi_tk_best[1]:
                abhi_tk_best[1] = "##" + abhi_tk_best[1]
            self.vocab[abhi_tk_best[0]+abhi_tk_best[1][2::]] = tmp[abhi_tk_best]

            # ab corpus ko update krna hai
            qp = self.update_corpus_spl(abhi_tk_best)
            # print(qp)
            self.corpus = qp.copy()
        self.write_vocabulary(5)
        return self.vocab

    def load_vocab(self):
        with open(self.vocab_file_path, 'r') as f:
            for line in f:
                token = line.strip()
                self.vocab[token] = 1  # We set frequency to 1 initially as we are only loading the vocabulary

    def tokenize(self,s):
        # test krne ke liye 
        s = s.lower().strip()
        s = ''.join(char for char in s if char.isalnum() or char.isspace()) # Assumption thi vocab me sirf letters,digits,spaces honge
        ans_dp = []
        i = 0
        while i < len(s):
            cur_mx = 0
            cur_best = None
            for j in range(i+1,len(s)+1):
                u = s[i:j]
                # wo segment test krna hai
                if u in self.vocab:
                    if len(u) > cur_mx:
                        cur_mx = len(u)
                        cur_best = u
                    # hm try kete hai jo longest hai wo le 
            if cur_best:
                ans_dp.append(cur_best)
                i += cur_mx
            else:
                ans_dp.append(self.unk) # appending the unknown token if no matching token found 
                i += 1
        return ans_dp
    def json_formatter(self,data):
        # data maine json se load kr liya using ek module jispr code run hoga 
        # print(data)
        # print(type(data))
        # data = data.split(" ")
        final = {}
        ans = []
        for i in data:
            q = self.tokenize(i['sentence'])
            final['id'] = i['id']
            final['tokens'] = q
            ans.append(q)
            print(final)
        for i in ans:
            print(i)
            print()
    def fit(self):
        self.read_karo_corpus()
        self.preprocess_data()
        self.construct_vocabulary()
# def test():
#     a = WordPieceTokenizer(100, 'Assignment1\corpus.txt')
#     (a.read_karo_corpus())
#     print(a.preprocess_data())
#     # print("{}{}{}")
#     print(a.construct_vocabulary())


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