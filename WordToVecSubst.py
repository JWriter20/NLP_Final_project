import gensim.downloader as api
import nltk
from nltk.tokenize import word_tokenize
import string

# Load a pre-trained word2vec model (might need internet connection to download)
model = api.load("word2vec-google-news-300")
nltk.download('punkt')  # Download necessary datasets

def load_text_vectors(file_path, model):
    """Load text and return a dictionary of word vectors from the text that are in the model's vocabulary."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    tokens = word_tokenize(text)
    return {word: model[word] for word in tokens if word in model.key_to_index}

def find_similar_shakespeare(word, model, shakespeare_vectors):
    """Find and return the most similar word from Shakespeare's texts if it exists in the model."""
    if word in model.key_to_index:
        similarities = model.most_similar(positive=[model[word]], topn=10)
        similar_words = [sim_word for sim_word, _ in similarities if sim_word in shakespeare_vectors]
        return similar_words[0] if similar_words else word
    return word

def shakespearify_text(input_file_path, reference_file_path, model):
    """
    Transforms the text in input_file_path into a style that reflects the vocabulary of the text in reference_file_path.
    """
    # Load vectors for the reference text (e.g., Macbeth)
    reference_vectors = load_text_vectors(reference_file_path, model)
    
    # Load the text to be transformed
    with open(input_file_path, 'r', encoding='utf-8') as file:
        input_text = file.read().lower()
    input_tokens = word_tokenize(input_text)
    
    # Transform words based on similarity to reference text
    transformed_text = [find_similar_shakespeare(word, model, reference_vectors) for word in input_tokens]
    
    # Join words to form the final text, handling punctuation correctly
    shakespearified_text = ''.join([
        (word if any(char in string.punctuation for char in word) else ' ' + word)
        for word in transformed_text
    ]).strip()

    return shakespearified_text

# transformed = shakespearify_text('./base_texts/AtlasShruggedSection.txt', './base_texts/macbeth.txt', model)
# print(transformed)
