from gensim.models import Word2Vec
from config import *

def learn_subproblem_features(C, w=W):
    """
    Learn subproblem features using Word2Vec.
    Parameters:
        C is the random walk generated 
        w is the size of the feature vector for each wscenario 
    """
    model = Word2Vec(sentences=C, vector_size=w, window=5, min_count=1, sg=1) 
    return model