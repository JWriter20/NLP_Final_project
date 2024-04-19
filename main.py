from markov_chain import generate_story
from n_gram_model import generate_from_text
from llm_based import main
from WordToVecSubst import shakespearify_text


# Baseline model:  n-gram-model.py
# ____________________________________________________________________________________________________

print("____________________________________________________________________________________________________")
print("Baseline model: n-gram-model.py")
print("____________________________________________________________________________________________________")
print(generate_from_text("./base_texts/Prisoner_of_Azkaban.txt", 2, 5, 100))

# Improved model:  markov_chain.py
# ____________________________________________________________________________________________________

print("____________________________________________________________________________________________________")
print("Improved model: markov_chain.py")
print("____________________________________________________________________________________________________")
print(generate_story("./base_texts/Prisoner_of_Azkaban.txt", "./base_texts/atlasshrugged.txt", 3, 100))


# LLM model: llm-based.py
# This function requires an openai API key in the .env file to run.
# ____________________________________________________________________________________________________

print("____________________________________________________________________________________________________")
print("LLM model: llm-based.py")
print("____________________________________________________________________________________________________")
# text1 = "The Great Gatsby"
# text2 = "1984"
# print(main(text1, text2))

# Word2Vec model: WordToVecSubset.py
# ____________________________________________________________________________________________________
import gensim.downloader as api
import nltk

print("____________________________________________________________________________________________________")
print("Word2Vec model: WordToVecSubset.py")
print("____________________________________________________________________________________________________")

model = api.load("word2vec-google-news-300")
nltk.download('punkt') 
transformed = shakespearify_text('./base_texts/AtlasShruggedSection.txt', './base_texts/macbeth.txt', model)
print(transformed)