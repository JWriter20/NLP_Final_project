import gensim.downloader as api

# Load a pre-trained word2vec model (this is just an example, might need internet connection to download)
model = api.load("word2vec-google-news-300")

import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')  # Download necessary datasets

with open('macbeth.txt') as f:
    macbeth_text = f.read()
macbeth_tokens = word_tokenize(macbeth_text.lower())

macbeth_vectors = {word: model[word] for word in macbeth_tokens if word in model.key_to_index}

def find_similar_shakespeare(word, model, shakespeare_vectors):
    if word in model.key_to_index:
        # Find most similar words according to the model
        similarities = model.most_similar(positive=[model[word]], topn=10)
        # Filter to find the most similar word that is also in Macbeth's vector space
        similar_words = [sim_word for sim_word, _ in similarities if sim_word in shakespeare_vectors]
        if similar_words:
            return similar_words[0]  # Return the most similar word found in Macbeth
    return word  # Return the original word if no similar word is found

with open('HarryPotterSection.txt') as f:
    other_text = f.read()
other_tokens = word_tokenize(other_text.lower())

transformed_text = [find_similar_shakespeare(word, model, macbeth_vectors) for word in other_tokens]
shakespearified_text = ' '.join(transformed_text)
print(shakespearified_text)
