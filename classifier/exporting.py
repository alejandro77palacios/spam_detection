from classifier.cleaning import NlpDF
from classifier.models import Message
from classifier.train import NlpModel

sms = NlpDF(Message.objects.all().values('text', 'label'))
nlp_model = NlpModel(sms)
nlp_model.train_test_split()
nlp_model.fit()
nlp_model.evaluate()
nlp_model.export('nlp_model')
