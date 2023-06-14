import pytest

from classifier.cleaning import NlpDF


@pytest.fixture
def nlp_df():
    data = {
        'text': ['WINNER free tickets!', 'I got home fine'],
        'label': ['spam', 'ham']
    }
    return NlpDF(data)


def test_constructor(nlp_df):
    assert len(nlp_df) == 2
    assert nlp_df.columns.tolist() == ['text', 'label']


def test_count_words(nlp_df):
    assert nlp_df.count_words('WINNER free tickets!') == 3
    assert nlp_df.count_words('I got home fine') == 4


def test_compute_mean_word_length(nlp_df):
    assert nlp_df.compute_mean_word_length('WINNER free tickets!') == 6.0
    assert nlp_df.compute_mean_word_length('I got home fine') == 3.0


def test_compute_basic_features(nlp_df):
    nlp_df.compute_basic_features()
    assert nlp_df.columns.tolist() == ['text', 'label', 'total_characters', 'total_words', 'mean_word_length']
    assert nlp_df['total_characters'].tolist() == [20, 15]
    assert nlp_df['total_words'].tolist() == [3, 4]
    assert nlp_df['mean_word_length'].tolist() == [6.0, 3.0]


def test_create_doc(nlp_df):
    nlp_df.create_doc()
    docs = nlp_df['doc'].tolist()
    assert docs[0].text == 'WINNER free tickets!'
    assert docs[1].text == 'I got home fine'


def test_compute_pos(nlp_df):
    nlp_df.create_doc()
    nlp_df.compute_pos()
    pos = nlp_df['pos'].tolist()
    assert pos[0] == ['PROPN', 'ADJ', 'NOUN', 'PUNCT']
    assert pos[1] == ['PRON', 'VERB', 'NOUN', 'NOUN']

def test_compute_pos_counts(nlp_df):
    nlp_df.create_doc()
    nlp_df.compute_pos()
    nlp_df.compute_pos_counts()
    assert nlp_df['total_nouns'].tolist() == [1, 2]
    assert nlp_df['total_proper_nouns'].tolist() == [1, 0]

def test_compute_nlp_features(nlp_df):
    nlp_df.compute_nlp_features()
    assert nlp_df.columns.tolist() == ['text', 'label', 'doc', 'pos', 'total_nouns', 'total_proper_nouns']
    assert nlp_df['total_nouns'].tolist() == [1, 2]
    assert nlp_df['total_proper_nouns'].tolist() == [1, 0]
