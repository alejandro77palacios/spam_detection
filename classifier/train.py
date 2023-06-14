import pickle
from pathlib import Path

from django.conf import settings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from classifier.cleaning import NlpDF


class NlpModel():
    def __init__(self, nlp_df: NlpDF):
        self.nlp_df = nlp_df
        self.features = nlp_df['text']
        self.X = self.features
        # self.features = nlp_df.drop(columns=['doc', 'pos', 'label'])
        self.target = nlp_df['label']
        self.y = self.target
        self.estimator = make_pipeline(CountVectorizer(), MultinomialNB())

    def train_test_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify=self.y)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self):
        self.estimator.fit(self.X_train, self.y_train)

    def export(self, name):
        path = settings.BASE_DIR / 'classifier' / f'{name}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.estimator, f)