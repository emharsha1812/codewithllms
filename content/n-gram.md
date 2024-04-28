---
title: Creating an n-gram language model
---

#### N-Gram Models

N-gram models are a type of probabilistic language model used in natural language processing and computational linguistics. These models are based on the idea that the probability of a word depends on the previous

−
For example, consider the following sentence: I love language models.

Unigram (1-gram): “I,” “love,” “language,” “models”

Bigram (2-gram): “I love,” “love language,” “language models”

Trigram (3-gram): “I love language,” “love language models”

4-gram: “I love language models”

N-gram models are simple and computationally efficient, making them suitable for various natural language processing tasks. However, their limitations include the inability to capture long-range dependencies in language and the sparsity problem when dealing with higher-order n-grams.

> Note: More advanced language models, such as recurrent neural networks (RNNs), have been replaced by large language models (LLMs).

Its algorithm is as follows:

Tokenization: Split the input text into individual words or tokens.

N-gram generation: Create n-grams by forming sequences of n-consecutive words from the tokenized text.

Frequency counting: Count the occurrences of each n-gram in the training corpus.

Probability estimation: Calculate the conditional probability of each word given its previous n−1
words using the frequency counts.

Smoothing (optional): Apply smoothing techniques to handle unseen n-grams and avoid zero probabilities.

Text generation: Start with an initial seed of n−1 words, predict the next word based on probabilities, and iteratively generate the next words to form a sequence.

Repeat generation: Continue generating words until the desired length or a stopping condition is reached.

Let’s see an example in action:

'''
import random

class NGramLanguageModel:
def **init**(self, n):
self.n = n
self.ngrams = {}
self.start_tokens = ['<start>'] \* (n - 1)

    def train(self, corpus):
        for sentence in corpus:
            tokens = self.start_tokens + sentence.split() + ['<end>']
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                if ngram in self.ngrams:
                    self.ngrams[ngram] += 1
                else:
                    self.ngrams[ngram] = 1

    def generate_text(self, seed_text, length=10):
        seed_tokens = seed_text.split()
        padded_seed_text = self.start_tokens[-(self.n - 1 - len(seed_tokens)):] + seed_tokens
        generated_text = list(padded_seed_text)
        current_ngram = tuple(generated_text[-self.n + 1:])

        for _ in range(length):
            next_words = [ngram[-1] for ngram in self.ngrams.keys() if ngram[:-1] == current_ngram]
            if next_words:
                next_word = random.choice(next_words)
                generated_text.append(next_word)
                current_ngram = tuple(generated_text[-self.n + 1:])
            else:
                break

        return ' '.join(generated_text[len(self.start_tokens):])

# Toy corpus

toy_corpus = [
"My name is Harshwardhan Fartale and i am currently a final year Electrical engineering student studying in NIT Hamirpur",
"The example demonstrates an N-gram language model.",
"N-grams are used in natural language processing.",
"This is a toy corpus for language modeling."
]

n = 6 # Change n-gram order here

# Example usage with seed text

model = NGramLanguageModel(n)  
model.train(toy_corpus)

seed_text = "My name is " # Change seed text here
generated_text = model.generate_text(seed_text, length=3)
print("Seed text:", seed_text)
print("Generated text:", generated_text)
'''
