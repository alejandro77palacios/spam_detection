from classifier.machine_learning.cleaning import NlpDF
from classifier.machine_learning.train import NlpModel
from classifier.models import Message


def main():
    sms = NlpDF(Message.objects.all().values('text', 'label'))
    nlp_model = NlpModel(sms)
    nlp_model.train_test_split()
    nlp_model.fit()
    nlp_model.evaluate()
    nlp_model.export('estimator')


if __name__ == '__main__':
    main()
