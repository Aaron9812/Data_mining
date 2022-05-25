from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def train_log_regression(df_train: pd.DataFrame, vect: TfidfVectorizer, C: int = 1):

    Xt_train = vect.transform(df_train['preprocessed'])
    y_train = df_train['label']

    model = LogisticRegression(C=C).fit(Xt_train, y_train);
    model.fit(Xt_train, y_train);

    return model