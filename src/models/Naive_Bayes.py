from xmlrpc.client import Boolean
import numpy as np
import pandas as pd
from sklearn import naive_bayes
import sklearn
import sklearn.model_selection as ms
import sklearn.feature_extraction.text as text
import sklearn.naive_bayes as nb
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def train_mvb_bayes(df_train: pd.DataFrame, vect: TfidfVectorizer):
    ''' Trains Multi-variate Bernoulli Naive Bayes model using GrtidSearchCV with 5-fold cross validation for smoothing parameter alpha.

    Args:
        df_train (pd.DataFrame): DataFrame containing training data
        vect (): Trained (TF-IDF) vectorizer

    Returns:
        bnb (GridSearchCV): Returns trained model as GridSearchCV object
    
    '''
    Xt_train = vect.transform(df_train['preprocessed'])
    y_train = df_train['label']

<<<<<<< HEAD
    bnb = ms.GridSearchCV(nb.BernoulliNB(), param_grid={'alpha': np.logspace(-2., 2., 50)})
=======
    # Multi-variate Bernoulli Naive Bayes
    bnb = ms.GridSearchCV(nb.BernoulliNB(), param_grid={'alpha': np.logspace(-5., 2., 100)})
>>>>>>> e8116e49583e7aa64f9c4c1661a0fcc75cf3c241
    bnb.fit(Xt_train, y_train);

    return bnb

def train_mn_bayes(df_train: pd.DataFrame, vect: TfidfVectorizer):
    ''' Trains Multinominal Naive Bayes model using GrtidSearchCV with 5-fold cross validation for smoothing parameter alpha.

    Args:
        df_train (pd.DataFrame): DataFrame containing training data
        vect (): Trained (TF-IDF) vectorizer

    Returns:
        mnb (GridSearchCV): Returns trained model as GridSearchCV object
    
    '''
    Xt_train = vect.transform(df_train['preprocessed'])
    y_train = df_train['label']

<<<<<<< HEAD
    mnb = ms.GridSearchCV(nb.MultinomialNB(), param_grid={'alpha':np.logspace(-2., 2., 50)})
=======
    # Multinominal Naive Bayes
    mnb = ms.GridSearchCV(nb.MultinomialNB(), param_grid={'alpha':np.logspace(-5., 2., 100)})
>>>>>>> e8116e49583e7aa64f9c4c1661a0fcc75cf3c241
    mnb.fit(Xt_train, y_train);

    return mnb

def test_model(model: ms.GridSearchCV, df_test: pd.DataFrame, vect: TfidfVectorizer, plt_confusion = False, get_params = False):
    ''' Tests sklearn classifier model as GridSearchCV object on test data.

<<<<<<< HEAD
    Args:
        model (): Trained sklearn classifier model
        df_test (pd.DataFrame): DataFrame containing test data
        vect (): Trained (TF-IDF) vectorizer
        plt_confusion (boolean): Plots confusion matrix
        get_params (boolean): Returns best performing hyperparameters for model

    Returns:
        predictions (list): List containing precision, recall, accuracy, f1 and best hyperparameter
=======
def test_model(model, df_test: pd.DataFrame, vect: TfidfVectorizer, plt_confusion = False, get_params = False):
>>>>>>> e8116e49583e7aa64f9c4c1661a0fcc75cf3c241
    
    '''    
    Xt_test = vect.transform(df_test['preprocessed'])
    y_test = df_test['label']
    y_pred = model.predict(Xt_test)

    predictions = []

    predictions.append(precision_score(y_test, y_pred))
    predictions.append(recall_score(y_test, y_pred))
    predictions.append(accuracy_score(y_test, y_pred))
    predictions.append(f1_score(y_test, y_pred))
    if get_params is True:
        predictions.append(model.best_params_)

    if plt_confusion is True:
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plot_confusion_matrix(conf_mat=cm)
        plt.title("Confusion Matrix")
        plt.show()

    return predictions

def get_impact_words(df_train: pd.DataFrame, vect: TfidfVectorizer, model = nb.MultinomialNB()):
    ''' Returns most impactful words of classes labeled hate and no hate.

    Args:
        df_train (pd.DataFrame): DataFrame containing training data
        vect (): Trained (TF-IDF) vectorizer
        model (nb.MultinomialNB): sklearn classifier model. Can not use GridSearchCV object

    Returns:
        none
    
    '''     
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