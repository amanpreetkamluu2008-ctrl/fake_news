import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import ssl
import nltk

