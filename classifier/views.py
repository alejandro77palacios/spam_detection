import pickle

from django.conf import settings
from django.shortcuts import render

estimator_path = settings.BASE_DIR / 'classifier' / 'estimator.pkl'

with open(estimator_path, 'rb') as f:
    classifier = pickle.load(f)


def index(request):
    if request.method == 'GET':
        return render(request, 'classifier/index.html')
    elif request.method == 'POST':
        sms = request.POST['input']
        prediction = classifier.predict([sms])
        predicted_label = 'Ham' if prediction[0] == 0 else 'Spam'
        estimated_probabilities = classifier.predict_proba([sms])[0]
        probability = f'{max(estimated_probabilities):.2%}'
        context = {'prediction': predicted_label,
                   'probability': probability,
                   'sms': sms}
        return render(request, 'classifier/index.html', context)
