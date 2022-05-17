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

def train_mvb_bayes(df_train: pd.DataFrame, vect: TfidfVectorizer):

    Xt_train = vect.transform(df_train['preprocessed'])
    y_train = df_train['label']

    # Multi-variate Bernoulli Naive Bayes
    bnb = ms.GridSearchCV(nb.BernoulliNB(), param_grid={'alpha':np.logspace(-2., 2., 50)})
    bnb.fit(Xt_train, y_train);

    return bnb

def train_mn_bayes(df_train: pd.DataFrame, vect: TfidfVectorizer):

    Xt_train = vect.transform(df_train['preprocessed'])
    y_train = df_train['label']

    # Multinominal Naive Bayes
    mnb = ms.GridSearchCV(nb.MultinomialNB(), param_grid={'alpha':np.logspace(-2., 2., 50)})
    mnb.fit(Xt_train, y_train);

    return mnb


def test_model(model, df_test: pd.DataFrame, vect: TfidfVectorizer):
    
    Xt_test = vect.transform(df_test['preprocessed'])
    y_test = df_test['label']
    y_pred = model.predict(Xt_test)

    predictions = []

    predictions.append(precision_score(y_test, y_pred))
    predictions.append(recall_score(y_test, y_pred))
    predictions.append(accuracy_score(y_test, y_pred))
    predictions.append(f1_score(y_test, y_pred))

    return predictions

def get_impact_words(df_train: pd.DataFrame, vect: TfidfVectorizer, model = nb.MultinomialNB()):
    
    Xt_train = vect.transform(df_train['preprocessed'])
    y_train = df_train['label']
  
    model.fit(Xt_train, y_train);
    
    model.feature_log_prob_
    model.coef_

    feature_names = vect.get_feature_names_out()
    for i, class_label in enumerate(['no_hate', 'hate']):
        top10 = np.argsort(model.feature_log_prob_[i])[-10:]
        print("%s: %s" % (class_label,
            " ".join(feature_names[j] for j in top10)))