import nltk
import math
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
from markov_chain import join_punctuation
nltk.data.path.append('.')
from nltk.tokenize import word_tokenize

# Removes all quotation marks from the text data
# @param data  the string from which to remove quotation marks
# @return      the string after removing all quotation marks
def remove_quotes(data):
    data = data.replace('\"', '')
    return data

# This isn't currently used, but it would be used to improve quoting 
# replaces quotation marks in the text data with 'startquote' and 'endquote' to distinguish
# @param data  the string in which to replace quotation marks
# @return      the string after replacing all quotation marks with 'startquote' or 'endquote'
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

# Splits the provided text data into sentences based on periods
# @param data  the string to split into sentences
# @return      a list of sentences
def split_to_sentences(data):
    sentences = [x for x in data.split(".") if x]
    return sentences

# Tokenizes sentences into words, converting all words to lower case
# @param sentences  list of sentences to tokenize
# @return           list of tokenized sentences, each a list of words
def tokenize_sentences(sentences):
    tokenized_sentences = [word_tokenize(s.lower()) for s in sentences]
    return tokenized_sentences


# Processes the provided text data by removing quotes, splitting into sentences,
# and tokenizing into words
# @param data  the raw text data to process
# @return      a list of tokenized sentences
def get_tokenized_data(data):
    # data = replace_quotes(data)
    data = remove_quotes(data)
    sentences = split_to_sentences(data)
    tokenized_sentences = tokenize_sentences(sentences)
    return tokenized_sentences

# Counts the occurrences of each word in the tokenized sentences
# @param tokenized_sentences  list of tokenized sentences
# @return                     dictionary of word counts
def count_words(tokenized_sentences):
    word_counts = {}
    for sentence in tokenized_sentences:
        for token in sentence:
            word_counts[token] = word_counts.get(token, 0) + 1
    return word_counts

# Filters words by frequency, selecting words that appear often
# @param tokenized_sentences  list of tokenized sentences
# @param minimum_freq         minimum frequency a word must have to be included
# @return                     list of words that meet the frequency threshold
def get_words_with_nplus_frequency(tokenized_sentences, minimum_freq):
    filtered_words = []
    word_counts = count_words(tokenized_sentences)
    filtered_words = [word for word, count in word_counts.items() if count >= minimum_freq]
    return filtered_words

# Replaces out-of-vocabulary words with a specified marker in tokenized sentences
# @param tokenized_sentences  list of tokenized sentences
# @param vocabulary           list of vocabulary words to retain
# @param unknown_marker       marker to substitute for out-of-vocabulary words
# @return                     list of sentences with words replaced by the unknown marker
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

# Preprocesses training and testing data by replacing out-of-vocabulary words and filtering vocabulary
# @param train_data    training data (tokenized sentences)
# @param test_data     testing data (tokenized sentences)
# @param minimum_freq  minimum frequency for words to be included in the vocabulary
# @return              tuple of preprocessed training data, testing data, and the vocabulary
def preprocess_data(train_data, test_data, minimum_freq):
    vocabulary = get_words_with_nplus_frequency(train_data, minimum_freq)
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary)
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary)
    return train_data_replaced, test_data_replaced, vocabulary

# Counts n-grams in the provided data
# @param data  tokenized data (list of tokenized sentences)
# @param n     size of n-gram (integer)
# @return      dictionary of n-gram counts
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


# Estimates the probability of a word given the previous n-gram
# @param word                    word to estimate probability for
# @param previous_n_gram         previous n-gram (tuple of words)
# @param n_gram_counts           dictionary of n-gram counts
# @param n_plus1_gram_counts     dictionary of (n+1)-gram counts
# @param vocabulary_size         size of the vocabulary
# @param k                       smoothing parameter
# @return                        estimated probability of the word
def estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=0.0):
    previous_n_gram = tuple(previous_n_gram)
    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)
    
    n_plus1_gram = previous_n_gram + (word,)
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)
    if n_plus1_gram_count == 0 or previous_n_gram_count == 0: return 0
    
    probability = n_plus1_gram_count / (previous_n_gram_count * vocabulary_size)

    return probability

# Calculates the perplexity of a sentence
# @param sentence                the sentence to calculate perplexity for
# @param n_gram_counts           dictionary of n-gram counts
# @param n_plus1_gram_counts     dictionary of (n+1)-gram counts
# @param vocabulary_size         size of the vocabulary
# @param k                       smoothing parameter
# @return                        perplexity of the sentence
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

# Suggests a word based on the previous context with specified constraints
# @param previous_tokens       list of previous tokens to consider for suggesting a new word
# @param n_gram_counts         dictionary of n-gram counts
# @param new_n_gram_counts     dictionary of (n+1)-gram counts used for probability estimation
# @param vocabulary            list of all known vocabulary words
# @param k                     smoothing parameter used in probability calculation
# @param start_with            optional prefix to filter suggestions that start with this substring
# @return                      tuple containing the suggested word and its probability
def suggest_a_word(previous_tokens, n_gram_counts, new_n_gram_counts, vocabulary, k=1.0, start_with=None):
    n = len(list(n_gram_counts.keys())[0])
    previous_n_gram = tuple(previous_tokens[-n:]) if len(previous_tokens) >= n else tuple(["<s>"]*(n-len(previous_tokens)) + previous_tokens)

    probabilities = estimate_probabilities(previous_n_gram, n_gram_counts, new_n_gram_counts, vocabulary, k=k)
    
    # Filter probabilities to only those that start with the specified prefix, if any
    if start_with:
        filtered_probabilities = {word: prob for word, prob in probabilities.items() if word.startswith(start_with)}
    else:
        filtered_probabilities = {word: prob for word, prob in probabilities.items() if prob > 0}
    
    # Sort words by probability, highest first
    sorted_probabilities = sorted(filtered_probabilities.items(), key=lambda x: x[1], reverse=True)

    # select top 4
    top_choices = sorted_probabilities[:min(4, len(sorted_probabilities))]
    if len(top_choices) == 0:
        return None, 0

    # Apply a square root transformation to probabilities to make them closer together
    transformed_probs = [(word, math.sqrt(prob)) for word, prob in top_choices]

    # Normalize their probabilities to sum to 1
    total_prob = sum(prob for _, prob in transformed_probs)
    if total_prob == 0:
        return None, 0  # no div by zero
    
    normalized_choices = [(word, prob/total_prob) for word, prob in transformed_probs]

    # choose word randomly 
    words, probs = zip(*normalized_choices)  # Unzip the list of tuples
    chosen_word = random.choices(words, weights=probs, k=1)[0]
    
    # Find the probability of the chosen word (before transformation and normalization)
    chosen_word_prob = dict(top_choices)[chosen_word]

    return chosen_word, chosen_word_prob

# Aggregates suggestions from multiple n-gram models
# @param previous_tokens       list of tokens to use for generating suggestions
# @param n_gram_counts_list    list of dictionaries, each being n-gram counts for a different model
# @param vocabulary            list of all known vocabulary words
# @param k                     smoothing parameter
# @param start_with            optional prefix for filtering suggestions
# @return                      list of suggestions from each model
def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts-1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i+1]
        
        suggestion = suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=k, start_with=start_with)
        suggestions.append(suggestion)
    return suggestions

# Estimates probabilities of all words given a previous n-gram
# @param previous_n_gram       the n-gram to use as context for the probability estimates
# @param n_gram_counts         dictionary of n-gram counts
# @param n_plus1_gram_counts   dictionary of (n+1)-gram counts used for probability estimation
# @param vocabulary            list of all known vocabulary words, including special tokens
# @param k                     smoothing parameter used in probability calculation
# @return                      dictionary of word probabilities given the previous n-gram
def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=0.0):
    previous_n_gram = tuple(previous_n_gram)
    vocabulary = vocabulary + ["<e>", "<unk>"]
    vocabulary_size = len(vocabulary)
    
    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=k)
        probabilities[word] = probability

    return probabilities

# Constructs a count matrix from (n+1)-gram counts
# @param n_plus1_gram_counts   dictionary of (n+1)-gram counts
# @param vocabulary            list of all vocabulary words, including special tokens
# @return                      DataFrame representing the count matrix
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


# Creates a probability matrix from count matrix by applying smoothing
# @param n_plus1_gram_counts   dictionary of (n+1)-gram counts
# @param vocabulary            list of all vocabulary words, including special tokens
# @param k                     smoothing parameter
# @return                      DataFrame representing the probability matrix
def make_probability_matrix(n_plus1_gram_counts, vocabulary, k):
    count_matrix = make_count_matrix(n_plus1_gram_counts, vocabulary)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return prob_matrix

# Generates text based on a model of n-grams
# @param tokenized_data        data from which to generate text (list of tokenized sentences)
# @param n                     the n-gram size to use for text generation
# @param num_words             number of words to generate
# @return                      generated text string
def generate_text(tokenized_data, n, num_words=100):
    # Count n-grams in the tokenized data
    text = []
    n_gram_counts = [count_n_grams(tokenized_data, i) for i in range(1, n+1)]
    for _ in range(num_words):
        suggestions = [key for key, value in get_suggestions(text, n_gram_counts, vocabulary, k=0.0)]
        next_word = suggestions[-1]
        if next_word == "<unk>" or next_word == None:
            for i in range(len(suggestions) - 1, -1, -1):
                if suggestions[i] != "<unk>" and suggestions[i] != None:
                    next_word = suggestions[i]
                    break

        if next_word == None:
            next_word = "<e>"
        
        if next_word == "<e>":  # If the end token is generated, restart the n-gram context
            text.append(next_word)
            text += ["<s>"] * (n-1)
        else:
            text.append(next_word)
        
        if len(text) >= num_words + n:  # n-1 additional tokens for initial tokens
            break

    # Return the generated text
    return ' '.join([word for word in text[n-1:] if word != "<s>"]).replace(" <e>", ".").replace("<unk>", "UNK")

# Example usage
with open(".base_texts/Prisoner_of_Azkaban.txt", "r") as f: # update the path accordingly.
    large_text = f.read()

preprocessed_data = get_tokenized_data(large_text)
train_data, test_data, vocabulary = preprocess_data(preprocessed_data, preprocessed_data, minimum_freq=2)

generated_text = generate_text(train_data, n=5, num_words=1000)
generated_text = " ".join(join_punctuation(generated_text.split(" ")))
print(generated_text.capitalize())