while len(self.vocab) < self.vocab_size:
        #     print(len(self.vocab))
        #     # Merge the most frequent subwords to form new tokens
        #     candidates = {}
        #     for word, freq in self.corpus.items():
        #         subwords = []
        #         i = 0
        #         while i < len(word):
        #             for j in range(len(word), i, -1):
        #                 candidate = word[i:j]
        #                 if i > 0:
        #                     candidate = "##" + candidate
        #                 if candidate in self.vocab:
        #                     subwords.append(candidate)
        #                     i = j - 1
        #                     break
        #             i += 1

        #         for k in range(len(subwords) - 1):
        #             pair = (subwords[k], subwords[k + 1])
        #             if pair in candidates:
        #                 candidates[pair] += freq
        #             else:
        #                 candidates[pair] = freq

        #     # Calculate pair scores and choose the best candidate
        #     pair_scores = {}
        #     for pair, freq in candidates.items():
        #         first, second = pair
        #         score = freq / (self.vocab.get(first, 1) * self.vocab.get(second, 1))
        #         pair_scores[pair] = score

        #     best_candidate = max(pair_scores, key=pair_scores.get, default=None)
        #     print("best_candidate", best_candidate)

        #     if best_candidate is None:
        #         print("maro merko 2")
        #         break

        #     # Add the new merged token to the vocabulary
        #     merged_token = best_candidate[0] + best_candidate[1][2:]  # Merge without "##" in the second subword
        #     self.vocab[merged_token] = candidates[best_candidate]

        #     # Update the corpus with the merged token
        #     new_corpus = {}
        #     for word, freq in self.corpus.items():
        #         new_word = word.replace(best_candidate[0] + best_candidate[1][2:], merged_token)
        #         new_corpus[new_word] = freq
        #     self.corpus = new_corpus

        #     print("best_candidate", merged_token)
        
        # self.write_vocabulary(5)

        # return self.vocab