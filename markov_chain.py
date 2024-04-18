import nltk
import random

# split the corpus into sentences based on period and line break
# @param data  the input text
# @return      a list of sentences
def split_to_sentences(data):
    return [sentence for sentence in map(str.strip, data.split('.')) if sentence]


# tokenize the sentences and remove unwanted punctuations from the tokens
# @param sentences  list of sentences
# @return           list of tokenized sentences
def tokenize_sentences(sentences):
    unwanted_punct = ['``', '--', '\'\'']
    
    tokenized_sentences = []
    for s in sentences:
        s = s.lower()
        tokens = nltk.word_tokenize(s)
        tokens = list(filter(lambda x: x not in unwanted_punct, tokens))
        tokenized_sentences.append(tokens)
    
    return tokenized_sentences


# create a model of possible subsequent words of each word
# @param n          size of N-gram
# @param sentences  list of tokenized sentences
# @return           dictionary that maps a list of possible words that follows an N-gram to the N-gram
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


# generate sentences based on a model using n-grams with size n and up to max_words amount of words
# @param model       dictionary that maps a list of possible words that follows an N-gram to the N-gram
# @param n           size of N-gram
# @param max_words   max number of words to generate
# @return            string representing the generated text
def generate_from_model(model, n, max_words):
    # randomly choose a start n-gram
    start = random.choice(list(model.keys()))
    while(not start[0].isalpha()):
        # make sure sentence starts with an English word
        start = random.choice(list(model.keys()))

    result = list(start)
    for i in range(max_words):
        start = tuple(result[-n:])
        next = random.choice(model[start])
        if not next:
           result[-1] += '.' # end of sentence, start new ngram
           start = random.choice(list(model.keys()))
           while(not start[0].isalpha()):
               start = random.choice(list(model.keys()))
           result.extend(start)
        else:
            result.append(next)

    result[0] = result[0].capitalize()
    return ' '.join(join_punctuation(result))
  

# join punctuations with previous word in a list and capitalize words at beginning of sentences
# @param seq  list of strings
# @yield      current token, either a word or word plus punctuation     
def join_punctuation(seq):
    characters = ['.',',', ';', '?', '!', '\'s', 'n\'t', '\'ll', '\'ve', '\'re', ':', '\'m']
    seq = iter(seq)
    current = next(seq)
    prev = 'a' # initilize prev using 'a' as placeholder

    for nxt in seq:
        if prev[-1] == '.' or prev[-1] == '!' or prev[-1] == '?':
            # capitalize beginning of a sentence
            current = current.capitalize()
        if current == 'i':
            # capitalize single 'I'
            current = 'I'
        if nxt in characters:
            # combine single punctuation with the end of last word
            current += nxt
        else:
            yield current
            prev = current
            current = nxt

    yield current


# print out a story combining corpus of text1, text2, using n-gram with size n and up to max_words amount of words
# @param text1       the first input text
# @param text2       the second input text
# @param n           size of N-gram
# @param max_words   max number of words to generate
def generate_story(text1, text2, n, max_words):
    with open(text1, "r", encoding="latin-1") as f1:
        data1 = f1.read()    
    with open(text2, "r", encoding="latin-1") as f2:
        data2 = f2.read()

    sentence_splits = split_to_sentences(data1)
    sentence_splits.extend(split_to_sentences(data2))
    sentences = tokenize_sentences(sentence_splits)
    model = possible_next_words(n, sentences)
    print(generate_from_model(model, n, max_words))


# generate_story(".base_texts/Prisoner_of_Azkaban.txt", ".base_texts/atlasshrugged.txt", 3, 100)