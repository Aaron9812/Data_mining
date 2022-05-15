import numpy as np
import pandas as pd
from sklearn import naive_bayes
import sklearn.model_selection as ms
import sklearn.feature_extraction.text as text
import sklearn.naive_bayes as nb
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def train_mvb_bayes(df_train: pd.DataFrame, tfidf: TfidfVectorizer):

    Xt_train = tfidf.transform(df_train['preprocessed'])
    y_train = df_train['label']

    # Multi-variate Bernoulli Naive Bayes
    bnb = ms.GridSearchCV(nb.BernoulliNB(), param_grid={'alpha':np.logspace(-2., 2., 50)})
    bnb.fit(Xt_train, y_train);

    return bnb

def train_mn_bayes(df_train: pd.DataFrame, tfidf: TfidfVectorizer):

    Xt_train = tfidf.transform(df_train['preprocessed'])
    y_train = df_train['label']

    # Multinominal Naive Bayes
    mnb = ms.GridSearchCV(nb.MultinomialNB(), param_grid={'alpha':np.logspace(-2., 2., 50)})
    mnb.fit(Xt_train, y_train);

    return mnb


def test_model(model, df_test: pd.DataFrame, tfidf: TfidfVectorizer):
    
    Xt_test = tfidf.transform(df_test['preprocessed'])
    y_test = df_test['label']
    y_pred = model.predict(Xt_test)

    predictions = []

    predictions.append(precision_score(y_test, y_pred))
    predictions.append(recall_score(y_test, y_pred))
    predictions.append(accuracy_score(y_test, y_pred))
    predictions.append(f1_score(y_test, y_pred))

    return predictions

