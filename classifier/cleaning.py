import numpy as np
import pandas as pd
import spacy

from .models import Message


class NlpDF(pd.DataFrame):
    nlp = spacy.load('en_core_web_sm')

    @staticmethod
    def count_words(text):
        return len(text.split())

    @staticmethod
    def compute_mean_word_length(text):
        return np.mean([len(word) for word in text.split()])

    def compute_basic_features(self):
        self['total_characters'] = self['text'].str.len()
        self['total_words'] = self['text'].apply(self.count_words)
        self['mean_word_length'] = self['text'].apply(self.compute_mean_word_length)

    def create_doc(self):
        self['doc'] = self['text'].apply(lambda x: self.nlp(x))

    def compute_pos(self):
        self['pos'] = self['doc'].apply(lambda x: [token.pos_ for token in x])

    def compute_pos_counts(self):
        self['total_nouns'] = self['pos'].apply(lambda x: x.count('NOUN'))
        self['total_proper_nouns'] = self['pos'].apply(lambda x: x.count('PROPN'))

    def compute_nlp_features(self):
        self.create_doc()
        self.compute_pos()
        self.compute_pos_counts()


# sms = NlpDF(Message.objects.all().values('text', 'label'))
