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
nltk.data.path.append('.')

def split_to_sentences(data):
    sentences = [x for x in data.split("\n") if x]
    return sentences

from nltk.tokenize import word_tokenize
def tokenize_sentences(sentences):
    tokenized_sentences = [word_tokenize(s.lower()) for s in sentences]
    return tokenized_sentences

def get_tokenized_data(data):
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
    previous_n_gram = previous_tokens[-n:]
    probabilities = estimate_probabilities(previous_n_gram, n_gram_counts, new_n_gram_counts, vocabulary, k=k)
    suggestion = None
    max_prob = 0
    for word, prob in probabilities.items():
        if start_with:
            if word.startswith(start_with) and prob > max_prob:
                suggestion = word
                max_prob = prob
        else:
            if prob > max_prob:
                suggestion = word
                max_prob = prob
    return suggestion, max_prob

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
    n_gram_counts = count_n_grams(tokenized_data, n)
    n_plus1_gram_counts = count_n_grams(tokenized_data, n+1)
    
    # Start with an initial n-gram
    text = ["<s>"] * (n-1)
    for _ in range(num_words):
        current_n_gram = tuple(text[-(n-1):]) if n > 1 else ()
        probabilities = estimate_probabilities(current_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary)
        
        # Weighted random choice of the next word
        words = list(probabilities.keys())
        probs = list(probabilities.values())
        next_word = np.random.choice(words, p=probs)
        
        if next_word == "<e>":  # If the end token is generated, restart the n-gram context
            text += ["<s>"] * (n-1)
        else:
            text.append(next_word)
        
        if len(text) >= num_words + n-1:  # n-1 additional tokens for initial tokens
            break

    # Return the generated text
    return ' '.join(text[n-1:]).replace(" <e>", ".").replace("<unk>", "UNK")

# Example usage
large_text = ""
preprocessed_data = get_tokenized_data(large_text)
train_data, _ = preprocess_data(preprocessed_data, preprocessed_data, minimum_freq=2)  # Assuming all data is for training
vocabulary = set(sum(train_data, []))  # Flatten the list of lists to get the vocabulary

generated_text = generate_text(train_data, n=3, num_words=1000)
print(generated_text)