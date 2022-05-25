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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import sklearn.model_selection as ms
from sklearn.utils import resample
import demoji
import re
import spacy
from typing import Tuple

demoji.download_codes()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#!python -m spacy download en_core_web_lg

def setup(rem_stop=True, do_stem=True, do_lem=False, split=True, upsample=True, do_emojis=True, no_user=False, vectorizer='tfidf'):
    ''' Downloads dataset and performs preprocessing.

    Args:
        rem_stop (boolean): Remove stopwords
        do_stem (boolean): Stemm tokens using porter stemmer
        do_lem (boolean): Lemmatize tokens
        split (boolean): Split data set
        upsample (boolean): Upsample data set
        do_emojis (boolean): Convert emojis to str
        no_user (boolean): remove "user" str
        vectorizer (str): Word vectorizer, TF-IDF and CountVectorizer implemented

    Returns:
        vect (TfidfVectorizer | CountVectorizer): Trained vectorizer
        df (pd.DataFrame): Returned if split is False. Contains preprocessed tweets
        df_train (pd.DataFrame): Returned if split is True. Contains preprocessed train tweets
        df_test (pd.DataFrame): Returned if split is True. Contains preprocessed test tweets
    
    ''' 
    df = load_data()

    df['preprocessed'] = preprocess(
        df['tweet'], rem_stop=rem_stop, do_stem=do_stem, do_lem=do_lem, do_emojis=do_emojis, no_user=no_user)

    if split is True:
        df_train, df_test = split_data(df)
        if vectorizer == 'tfidf':
            vect = train_tfidf(df_train['preprocessed'])
        else:
            vect = train_count_vectorizer(df_train['preprocessed'])
        if upsample is True:
            df_train = upsampling(df_train)
        return vect, df_train, df_test
    else:
        if vectorizer == 'tfidf':
            vect = train_tfidf(df['preprocessed'])
        else:
            vect = train_count_vectorizer(df['preprocessed'])
        return vect, df


def load_data():
    ''' Downloads "tweets_hate_speech_detection" dataset.
    
    Returns:
        df (pd.DataFrame): DataFrame containing labeled tweets
    
    ''' 
    dataset = load_dataset("tweets_hate_speech_detection")
    df = pd.DataFrame.from_dict(dataset['train'])
    return df


def preprocess(data: pd.Series, rem_stop=True, do_stem=True, do_lem=False, do_emojis=True, no_user=False):
    ''' Performs preprocessing.
    Args:
        data (pd.Series): Series containing tweets as str
        rem_stop (boolean): Remove stopwords
        do_stem (boolean): Stemm tokens using porter stemmer
        do_lem (boolean): Lemmatize tokens
        do_emojis (boolean): Convert emojis to str
        no_user (boolean): remove "user" str

    Returns:
        preprocessed (list): List containing preprocessed tweets
    
    ''' 
    preprocessed = []
    if no_user is True:
        data = data.str.replace("user","")
    for tweet in data:
        if do_emojis is True:
            tweet, _ = convert_emoji(tweet)
        tokens = tokenization(remove_punctuation(tweet))
        if rem_stop is True:
            tokens = remove_stopwords(tokens)
        if do_stem is True and do_lem is False:
            tokens = stemming(tokens)
        if do_lem is True and do_stem is False:
            tokens = lemmatization(tokens)
        preprocessed.append(np.array(tokens))

    return preprocessed


def train_tfidf(data: pd.Series):
    ''' Trains tfidf model using custom preprocessing.
    Args:
        data (pd.Series): Series containing lists of preprocessed tweets

    Returns:
        (TfidfVectorizer): Trained TfidfVectorizer
    
    ''' 
    def dummy(text):
        return text

    tf = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy,
        preprocessor=dummy,
        token_pattern=None)

    return tf.fit(data)

def train_count_vectorizer(data: pd.Series):
    ''' Trains CountVectorizer model using custom preprocessing.
    Args:
        data (pd.Series): Series containing lists of preprocessed tweets

    Returns:
        (CountVectorizer): Trained CountVectorizer
    
    ''' 
    def dummy(text):
        return text

    co = CountVectorizer(
        analyzer='word',
        tokenizer=dummy,
        preprocessor=dummy,
        token_pattern=None)

    return co.fit(data)


def split_data(df: pd.DataFrame, test_size=0.2, random_state=17):

    df_train, df_test = ms.train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["label"])

    print('There is {} training data, of which {}% is hate speech '.format(df_train['label'].count(), round(df_train['label'].sum()/df_train['label'].count()*100,2)))
    print('There is {} test data, of which {}% is hate speech '.format(df_test['label'].count(), round(df_test['label'].sum()/df_test['label'].count()*100,2)))

    return df_train, df_test


def upsampling(df: pd.DataFrame, replace=True, n_samples=23775, random_state=55):
    data_minority = df[df.label == 1]
    data_majority = df[df.label == 0]
    data_minority = resample(
        data_minority, replace=replace, n_samples=n_samples, random_state=random_state)

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


def convert_emoji(text: str):
    # convert string to binary representation
    binary = ' '.join(format(ord(x), 'b') for x in text)

    # convert binary representation to utf8 representation
    listRes = list(binary.split(" "))
    try:
        text_with_emoji = bytes([int(x, 2) for x in listRes]).decode('utf-8')
    except UnicodeDecodeError:
        return text, []

    # get all emojis
    dictionary = demoji.findall(text_with_emoji)

    # replace emojis with text representation
    emojis = []
    for key in dictionary.keys():
        if key in text_with_emoji: emojis.append(dictionary[key])
        text_with_emoji = text_with_emoji.replace(key, dictionary[key] + " ")

    return text_with_emoji, emojis

def emb_data(data):
    nlp = spacy.load("en_core_web_lg") #If you are using colab and this buggs out: Restart runtime but DO NOT install the "en_core_web_lg" again.
    tweets = data.values.tolist()
    nlp.disable_pipes("parser", "ner") #remove pipe we do not need
    embeddings = [sum([word.vector for word in item])/len(item) for item in nlp.pipe(tweets)] #Takes some time...
    return pd.Series(embeddings).values

def get_features(df: pd.DataFrame):
    df["n_mentions"] = df["tweet"].apply(lambda x: count_user_mentions(x))
    df["hashtags"] = df["tweet"].apply(lambda x: identify_hashtags(x))
    df["emojis"] = df["tweet"].apply(lambda x: convert_emoji(x)[1])
    df["emb"] = emb_data(df["tweet"])
    return df

def count_user_mentions(text:str) ->int:
    return text.count("@user")

def identify_hashtags(text:str) -> list:
    pattern = re.compile(r"#(\w+)")
    return pattern.findall(text)



