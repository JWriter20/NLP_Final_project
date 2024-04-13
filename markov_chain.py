import nltk
import random

# split the corpus into sentences based on period and line break
def split_to_sentences(data):
    return [sentence for sentence in map(str.strip, data.split('.')) if sentence]


# tokenize the sentences and remove unwanted punctuations from the tokens
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

    result[0].capitalize()
    result = ' '.join(result)
    return join_punctuation(result)


def join_punctuation(seq):
    result = ''
    for i, s in enumerate(seq):
        if i+1 < len(seq):
            if s[-1] == '.' and seq[i+1].alpha():
                seq[i+1].capitalize()
            if not s[i+1].alpha:
                result.append(s[i+1])
            else:
                result.append(' ' + s[i+1])
    return result

        

# # join punctuations with previous word in a list
# def join_punctuation(seq):
#     characters = ['.',',', ';', '?', '!', '\'s', 'n\'t', '\'ll', '\'ve', '\'re', ':']
#     seq = iter(seq)
#     current = next(seq)
#     prev = 'a'

#     for nxt in seq:
#         if prev[-1] == '.':
#             current.capitalize()
#         if nxt in characters:
#             current += nxt
#         else:
#             yield current
#             prev = current
#             current = nxt

#     yield current


# print out a story combining corpus of text1, text2, using n-gram with size n and up to max_words amount of words
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


generate_story("./Prisoner_of_Azkaban.txt", "./atlasshrugged.txt", 2, 1000)