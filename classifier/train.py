import pickle

from django.conf import settings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from classifier.cleaning import NlpDF


class NlpModel():
    def __init__(self, nlp_df: NlpDF):
        self.nlp_df = nlp_df
        self.features = nlp_df['text']
        self.target = nlp_df['label']
        self.X = self.features
        # self.features = nlp_df.drop(columns=['doc', 'pos', 'label'])
        self.y = self.target
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.estimator = make_pipeline(CountVectorizer(), MultinomialNB())
        self.evaluation_score = None
        self.roc_auc = None

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                stratify=self.y)

    def fit(self):
        self.estimator.fit(self.X_train, self.y_train)

    def evaluate(self):
        self.evaluation_score = self.estimator.score(self.X_test, self.y_test)
        self.roc_auc = roc_auc_score(self.y_test, self.estimator.predict_proba(self.X_test)[:, 1])

    def export(self, name):
        path = settings.BASE_DIR / 'classifier' / f'{name}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.estimator, f)
