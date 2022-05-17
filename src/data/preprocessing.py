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
import demoji
import re
import spacy
from typing import Tuple

demoji.download_codes()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#!python -m spacy download en_core_web_lg

def setup(rem_stop=True, do_stem=True, do_lem=False, split=True, upsample=True, do_emojis=True):
    df = load_data()

    df['preprocessed'] = preprocess(
        df['tweet'], rem_stop=rem_stop, do_stem=do_stem, do_lem=do_lem, do_emojis=do_emojis)

    if split is True:
        df_train, df_test = split_data(df)
        tfidf = train_tfidf(df_train['preprocessed'])
        if upsample is True:
            df_train = upsampling(df_train)
        return tfidf, df_train, df_test
    else:
        tfidf = train_tfidf(df['preprocessed'])
        return tfidf, df


def load_data():
    dataset = load_dataset("tweets_hate_speech_detection")
    df = pd.DataFrame.from_dict(dataset['train'])
    return df


def preprocess(data, rem_stop=True, do_stem=True, do_lem=False, do_emojis=True):
    # assert do_stem != do_lem
    preprocessed = []
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


def train_tfidf(data):
    def dummy(text):
        return text

    tf = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy,
        preprocessor=dummy,
        token_pattern=None)

    return tf.fit(data)


def split_data(df: pd.DataFrame, test_size=0.2, random_state=17):

    df_train, df_test = ms.train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["label"])

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



