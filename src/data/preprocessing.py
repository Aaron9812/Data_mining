import string
from xmlrpc.client import Boolean
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
from datasets import load_dataset
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.model_selection as ms
from sklearn.utils import resample

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def setup(rem_stop = True, do_stem = True, do_lem = False, split = True, split_on = 'preprocessed', upsample = True):
    df = load_data()
    df['preprocessed'] = preprocess(df['tweet'], rem_stop = rem_stop, do_stem = do_stem, do_lem = do_lem)

    tfidf = train_tfidf(df['preprocessed'])
    
    if split is True:
        df_train , df_test = split_data(df, split_on)
        if upsample is True:
            df_train = upsampling(df_train)
        return tfidf, df_train, df_test
    else:
        return tfidf, df

    

def load_data():
    dataset = load_dataset("tweets_hate_speech_detection")
    df = pd.DataFrame.from_dict(dataset['train'])
    return df

def preprocess(data, rem_stop = True, do_stem = True, do_lem = False):
    
    preprocessed = []
    for tweet in data:
        tokens = tokenization(remove_punctuation(tweet))
        if rem_stop is True:
            tokens = remove_stopwords(tokens)
        if do_stem is True and do_lem is False:
            tokens = stemming(tokens)
        if do_lem is True and do_stem is False:
            tokens = lemmatization(tokens)
        preprocessed.append(np.array(tokens))
        
    return preprocessed

def train_tfidf(data):
    def dummy(text):
        return text
    
    tf = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy,
        preprocessor=dummy,
        token_pattern=None)
    
    return  tf.fit(data)

def split_data(df: pd.DataFrame, split_on = 'tweet', test_size = 0.2, random_state = 17):
    y = df['label']
    X = df[split_on]
    (X_train, X_test, y_train, y_test) = ms.train_test_split(X, y, test_size=test_size, random_state = random_state, stratify=y)

    df_train = pd.concat([y_train,X_train], axis=1)
    df_test = pd.concat([y_test,X_test], axis = 1)

    return df_train, df_test

def upsampling(df: pd.DataFrame, replace = True, n_samples = 23775, random_state = 55):
    data_minority = df[df.label == 1]
    data_majority = df[df.label == 0]
    data_minority = resample(data_minority, replace = replace, n_samples=n_samples, random_state=random_state)

    return pd.concat([data_majority, data_minority])

def tokenization(text: str):
    return pd.Series(nltk.word_tokenize(text.lower()))

def remove_punctuation(tokens: pd.Series):
    return "".join([i for i in tokens if i not in punctuation])

def remove_stopwords(tokens: pd.Series):
    stopwords_list = stopwords.words("english")
    return tokens.apply(lambda token: token if token not in stopwords_list and token != '' else None).dropna()

def stemming(tokens: pd.Series):
    stemmer = PorterStemmer()

    return tokens.apply(lambda token: stemmer.stem(token))

def lemmatization(tokens: pd.Series):
    lemmatizer = WordNetLemmatizer()

    return tokens.apply(lambda token: lemmatizer.lemmatize(token))








