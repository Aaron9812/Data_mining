from string import punctuation
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


# Currently not in use
def remove_punctioation(text:str) -> str:
    return "".join([i for i in text if i not in punctuation])

def tokenization(text:str) -> list:
    return nltk.word_tokenize(text)

def remove_stopwords(tokens) ->list:
    stopwords_list = stopwords.words("english")
    return [token for token in tokens if token not in stopwords_list]

porter_stemmer = PorterStemmer()

def stemming(text:list) -> list:
    return [porter_stemmer.stem(word) for word in text]

def preProcess(list):
    return list.apply(lambda x: stemming(remove_stopwords(tokenization(remove_punctioation(x.lower())))))

def preProcess2(list):
    return list.apply(lambda x: remove_stopwords(tokenization(remove_punctioation(x.lower()))))

def dummy(text):
    return text

def validate(y_test,y_pred):
    print('Precision: %.3f' % precision_score(y_test, y_pred))
    print('Recall: %.3f' % recall_score(y_test, y_pred))
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    print('F1 Score: %.3f' % f1_score(y_test, y_pred))