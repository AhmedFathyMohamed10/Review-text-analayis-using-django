import pickle
import os

def load_model():
    # load the ML model
    model = pickle.load(open(os.path.join(os.path.dirname(__file__), 'Model/model.pkl'), 'rb'))
    return model

# load the vectorizer
def load_vectorizer():
    vectorizer = pickle.load(open(os.path.join(os.path.dirname(__file__), 'Model/vectorizer.pkl'), 'rb'))
    return vectorizer

