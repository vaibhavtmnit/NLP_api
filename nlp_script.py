import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import unicodedata
import string
from nltk.stem import WordNetLemmatizer
import copy
import nltk
import string
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from nltk.sentiment.vader import SentimentIntensityAnalyzer


snetiment_analyser = SentimentIntensityAnalyzer()
url_filter = re.compile(r'(\w+:\/\/\S+)|^rt|http.+?|[^\w\s]')
esc_rem = re.compile(r'\\n|\\u|\\xa0')
sym_rem = re.compile(r'[/(){}\[\]\|@,;$\"\'.%*]')
bad_sym = re.compile('[^a-z ]') # Only keeping characters and space
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
trailing_space = re.compile(' +')


def CleanString(text, stop_words=stopwords,
                lemma_func = lemmatizer.lemmatizer):
    text = text.lower()  # lowercase text
    text = text.strip()
    text = url_filter.sub('', text)
    text = esc_rem.sub('', text)  # Remove esc_rem symbols
    text = sym_rem.sub(' ', text)  # Replace symbols with space
    text = bad_sym.sub('', text)  # Removing all other symbols
    text = trailing_space.sub(' ', text)  # Removing trailing space
    text = nltk.tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [word for word in text if word not in string.punctuation]
    text = [lemma_func(i) for i in text]

    # TODO : Vaibhav to apply encoding processing later
    ## unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore'))

    text = " ".join([w for w in text if len(w) > 1])

    return text


def apply_vader_polarity_score(text):

    return snetiment_analyser.polarity_scores(text)


def process_data(json_file):

    data = data['text'].apply(lambda x:CleanString(x))

    data = pd.concat([data,
                      pd.json_normalize(data['text'].apply(
                          apply_vader_polarity_score))],
                     axis=1, ignore_index=True)

    return data


def produce_analytics(processed_data):

    processed_data['tag'] = np.where(processed_data['compound'] > .05, 'Positive',
                                     np.where(processed_data['compound'] < -.05,'Negative', 'Neutral'))

    return processed_data







