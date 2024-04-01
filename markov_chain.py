# Markov chain is essentially just an N-gram

'''
Goals: we want to learn to make our own NLP program that is capable of generating more of
or merging some of our favorite texts. In order to do this, we will first start with a 
basic n-gram model. We will progressively add different context tensor data with the 
goal of optimizing our model for this specific task and understanding our given text very 
well. 

Some possible vector matrix layers to add to our tensor:
1. Word correlation within sentences/ association. Ex. Harry potter with magician, ron, hermione, (positive words too, Voldemort with negative) 
2. Sentence-level "mood" - ex. urgent, inquisitive etc. (sliding window of mood values at both the sentece and word level for 
understanding the context a feel of the story)
3. Paragraph/story level context - Exposition, rising action, climax, falling action, resolution
4. Frequency of following words (n-gram)
5. Current sentence length (we don't want run ons)
6. Keep track of exising noun words in action? (For instance we don't want random objects just appearing we want it to be realistic)


'''

import nltk
import random

with open("./Prisoner_of_Azkaban.txt", "r") as f: # update the path accordingly.
    data = f.read()

def split_to_sentences(data):
    return [sentence for sentence in map(str.strip, data.split('.')) if sentence] # split based on period and line break

def tokenize_sentences(sentences):
    """
    Tokenize sentences into tokens (words)
    
    Args:
        sentences: List of strings
    
    Returns:
        List of lists of tokens
    """
    
    tokenized_sentences = []
    for s in sentences:
        s = s.lower()
        tokens = nltk.word_tokenize(s)
        tokenized_sentences.append(tokens)
    
    return tokenized_sentences

sentences = tokenize_sentences(split_to_sentences(data))

# Generate n-grams of size n
# sentences: list of sentence to generate n-gram from
# n: size of each sequence
def generate_ngram(sentences, n):
    ngram = []
    for s in sentences:
        ngram.append([s[i:i+n] for i in range(len(s)-n+1)])
    return ngram

three_gram = generate_ngram(sentences, 3)
# print(three_gram)
# for gram in three_gram:
#     print(next(gram))


def possible_next_words(n, sentences):
    model = {}
    for s in sentences:
        s.append(None) # None indicate end of sentence
        for i in range(len(s)-n):
            ngram = tuple(s[i:i+n])
            next = s[i+n]
            if ngram not in model:
                model[ngram] = []
            model[ngram].append(next)
    return model

model = possible_next_words(2, sentences)


def generate_from_model(model, n, start=None, max_words=1000):
    if not start:
        start = random.choice(list(model.keys()))
    result = list(start)
    for i in range(max_words):
        start = tuple(result[-n:])
        next = random.choice(model[start])
        if not next:
           result[-1] += '.'
           result.extend(random.choice(list(model.keys())))
        else:
            result.append(next)
    return ' '.join(result)

print(generate_from_model(model, 2))