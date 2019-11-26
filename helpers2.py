import numpy as np

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def get_cosine_sim(*strs): 
    vectors = [t for t in get_vectors(*strs)]
    return cosine_similarity(vectors)
    
def get_vectors(*strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()

def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def containment(ngram_array):
    ''' Containment is a measure of text similarity. It is the normalized, 
       intersection of ngram word counts in two texts.
       :param ngram_array: an array of ngram counts for an answer and source text.
       :return: a normalized containment value.'''
    #Creating the intersection list
    intersection_list = np.amin(ngram_array, axis=0)

    #Sum up number of the intersection counts
    intersection = np.sum(intersection_list)

    #Counting up the number of n-grams in the answer text
    answer_idx = 0
    answer_cnt = np.sum(ngram_array[answer_idx])

    # normalize and get final containment value

    return intersection / answer_cnt