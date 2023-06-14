from classifier.cleaning import NlpDF
from classifier.models import Message
from classifier.train import NlpModel

sms = NlpDF(Message.objects.all().values('text', 'label'))
estimator = NlpModel(sms)
estimator.train_test_split()
estimator.fit()
estimator.export('estimator')
