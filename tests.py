import unittest
from markov_chain import generate_story
from n_gram_model import generate_from_text

# Very basic tests. Since the word generation is random, we can't test for exact output.

def is_valid_story_len(text):
    words = text.split()
    if len(words) <= 100:
        return True

    text_after_100_words = ' '.join(words[100:])
    
    # Check if there's any sentence starting after the 100 words
    if not text_after_100_words.strip():
        return True
    
    # Ensure that no new sentence starts after the 100th word
    if text_after_100_words[0] in ".!?":
        return False
    return True

class TestStringLength(unittest.TestCase):
    def test_markov_string_length_100(self):
        # Allow for sentence to finish past 100 words, but no new sentence should start
        story = generate_story("./base_texts/Prisoner_of_Azkaban.txt", "./base_texts/atlasshrugged.txt", 3, 100)
        self.assertTrue(is_valid_story_len(story))

    def test_ngram_string_length_100(self):
        # Allow for sentence to finish past 100 words, but no new sentence should start
        story = generate_from_text("./base_texts/Prisoner_of_Azkaban.txt", 2, 5, 100)
        self.assertTrue(is_valid_story_len(story))

    

if __name__ == "__main__":
    unittest.main()