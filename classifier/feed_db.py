from django.conf import settings

from classifier.models import Message

data = settings.BASE_DIR / 'classifier' / 'raw_data' / 'spam_collection.txt'

with open(data) as f:
    for line in f:
        label, text = line.split('\t')
        label = 1 if label == 'spam' else 0
        Message.objects.create(text=text.strip(), label=label)
