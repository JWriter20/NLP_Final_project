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
import math
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
nltk.data.path.append('.')

def remove_quotes(data):
    data = data.replace('\"', '')
    return data


def replace_quotes(data):
    isQuoteStart = True
    newData = []
    for i, character in enumerate(data):
        if character == '\"':
            if isQuoteStart:
                newData.append('startquote')
            else:
                newData.append('endquote')
            
            isQuoteStart = not isQuoteStart
        newData.append(character)
    return "".join(newData)

def split_to_sentences(data):
    sentences = [x for x in data.split(".") if x]
    return sentences

from nltk.tokenize import word_tokenize
def tokenize_sentences(sentences):
    tokenized_sentences = [word_tokenize(s.lower()) for s in sentences]
    return tokenized_sentences

def get_tokenized_data(data):
    # data = replace_quotes(data)
    data = remove_quotes(data)
    sentences = split_to_sentences(data)
    tokenized_sentences = tokenize_sentences(sentences)
    return tokenized_sentences

def count_words(tokenized_sentences):
    word_counts = {}
    for sentence in tokenized_sentences:
        for token in sentence:
            word_counts[token] = word_counts.get(token, 0) + 1
    return word_counts

def get_words_with_nplus_frequency(tokenized_sentences, minimum_freq):
    filtered_words = []
    word_counts = count_words(tokenized_sentences)
    filtered_words = [word for word, count in word_counts.items() if count >= minimum_freq]
    return filtered_words

def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_marker="<unk>"):
    vocabulary = set(vocabulary)  # this makes each search faster
    vocabulary.add("startquote")
    vocabulary.add("endquote")
    replaced_tokenized_sentences = []
    for sentence in tokenized_sentences:
        replaced_sentence = []
        for word in sentence:
            if word in vocabulary:
                replaced_sentence.append(word)
            else:
                replaced_sentence.append(unknown_marker)
        replaced_tokenized_sentences.append(replaced_sentence)
    return replaced_tokenized_sentences

def preprocess_data(train_data, test_data, minimum_freq):
    vocabulary = get_words_with_nplus_frequency(train_data, minimum_freq)
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary)
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary)
    return train_data_replaced, test_data_replaced, vocabulary

def count_n_grams(data, n):
    n_grams = {}
    for sentence in data:
        sentence = ["<s>"]*n + sentence + ["<e>"]
        sentence = tuple(sentence)
        for i in range(len(sentence) - n + 1):
            n_gram = sentence[i:i+n]
            n_gram = tuple(n_gram)
            if n_gram in n_grams:
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 1
    return n_grams

def estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    previous_n_gram = tuple(previous_n_gram)
    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)
    
    n_plus1_gram = previous_n_gram + (word,)
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)
    
    probability = (n_plus1_gram_count + k) / (previous_n_gram_count + k * vocabulary_size)

    return probability

def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    n = len(list(n_gram_counts.keys())[0]) 
    sentence = ["<s>"] * n + sentence + ["<e>"]
    sentence = tuple(sentence)
    N = len(sentence)
    p = 1.0
    for i in range(N - n):
        n_gram = sentence[i:i+n]
        n_plus1_gram = sentence[i:i+n+1]
        n_gram_count = n_gram_counts.get(n_gram, 0)
        n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)
        probability = (n_plus1_gram_count + k) / (n_gram_count + k * vocabulary_size)
        p *= probability
    perplexity = (1/p)**(1/N)
    return perplexity

def suggest_a_word(previous_tokens, n_gram_counts, new_n_gram_counts, vocabulary, k=1.0, start_with=None):
    n = len(list(n_gram_counts.keys())[0])
    previous_n_gram = tuple(previous_tokens[-n:]) if len(previous_tokens) >= n else tuple(["<s>"]*(n-len(previous_tokens)) + previous_tokens)

    probabilities = estimate_probabilities(previous_n_gram, n_gram_counts, new_n_gram_counts, vocabulary, k=k)
    
    if start_with:
        filtered_probabilities = {word: prob for word, prob in probabilities.items() if word.startswith(start_with)}
    else:
        filtered_probabilities = probabilities
     
    sorted_probabilities = sorted(filtered_probabilities.items(), key=lambda x: x[1], reverse=True)

    top_choices = sorted_probabilities[:4]

    total_prob = sum(prob for _, prob in top_choices)
    if total_prob == 0:
        return None, 0 
    
    normalized_choices = [(word, prob/total_prob) for word, prob in top_choices]
    words, probs = zip(*normalized_choices)  
    chosen_word = random.choices(words, weights=probs, k=1)[0]
    
    chosen_word_prob = dict(top_choices)[chosen_word]

    return chosen_word, chosen_word_prob

def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts-1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i+1]
        
        suggestion = suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=k, start_with=start_with)
        suggestions.append(suggestion)
    return suggestions

def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0):
    previous_n_gram = tuple(previous_n_gram)
    vocabulary = vocabulary + ["<e>", "<unk>"]
    vocabulary_size = len(vocabulary)
    
    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=k)
        probabilities[word] = probability

    return probabilities

def make_count_matrix(n_plus1_gram_counts, vocabulary):
    vocabulary = vocabulary + ["<e>", "<unk>"]
    n_grams = []
    for n_plus1_gram in n_plus1_gram_counts.keys():
        n_gram = n_plus1_gram[0:-1]
        n_grams.append(n_gram)
    n_grams = list(set(n_grams))
    row_index = {n_gram:i for i, n_gram in enumerate(n_grams)}
    col_index = {word:j for j, word in enumerate(vocabulary)}
    nrow = len(n_grams)
    ncol = len(vocabulary)
    count_matrix = np.zeros((nrow, ncol))
    for n_plus1_gram, count in n_plus1_gram_counts.items():
        n_gram = n_plus1_gram[0:-1]
        word = n_plus1_gram[-1]
        if word not in vocabulary:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i, j] = count
    count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=vocabulary)
    return count_matrix

def make_probability_matrix(n_plus1_gram_counts, vocabulary, k):
    count_matrix = make_count_matrix(n_plus1_gram_counts, vocabulary)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return prob_matrix

def generate_text(tokenized_data, n, num_words=1000):
    # Count n-grams in the tokenized data
    text = []
    n_gram_counts = [count_n_grams(tokenized_data, i) for i in range(1, n+1)]
    for _ in range(num_words):
        suggestions = [key for key, value in get_suggestions(text, n_gram_counts, vocabulary, k=1.0)]
        next_word = max(set(suggestions), key=suggestions.count)

        if next_word == "<unk>":
            for i in range(len(suggestions) - 1, -1, -1):
                if suggestions[i] != "<unk>":
                    next_word = suggestions[i]
                    break
        
        if next_word == "<e>":  # If the end token is generated, restart the n-gram context
            text += ["<s>"] * (n-1)
        else:
            text.append(next_word)
        
        if len(text) >= num_words + n-1:  # n-1 additional tokens for initial tokens
            break

    # Return the generated text
    return ' '.join(text[n-1:]).replace(" <e>", ".").replace("<unk>", "UNK").replace("<s>", "")

# Example usage
with open("./Prisoner_of_Azkaban.txt", "r") as f: # update the path accordingly.
    large_text = f.read()

preprocessed_data = get_tokenized_data(large_text)
train_data, test_data, vocabulary = preprocess_data(preprocessed_data, preprocessed_data, minimum_freq=2)

generated_text = generate_text(train_data, n=5, num_words=1000)
print(generated_text)