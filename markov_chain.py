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



with open("./en_US.twitter.txt", "r") as f: # update the path accordingly.
    data = f.read()