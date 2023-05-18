# twitter sentiment analysis using NLP

import nltk
import random
import re
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.classify.scikitlearn import SklearnClassifier
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

# read the data from the file
data = pd.read_csv('data/training.csv')
data.drop(['2401', 'Borderlands'], axis=1, inplace=True)



