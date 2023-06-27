import pickle

import pandas as pd
from django.conf import settings
from django.shortcuts import render, redirect

from classifier.forms import UploadFileForm

app_path = settings.BASE_DIR / 'classifier'
estimator_path = app_path / 'estimator.pkl'

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


def process_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            if file.name.endswith('.csv') or file.name.endswith('.xlsx'):
                df = predict_csv(file)
                result_path = settings.BASE_DIR / 'classifier' / 'results' / f'{file.name[:-4]}_result.csv'
                df.to_csv(result_path, index=False)
                print(result_path.resolve())
                return download_file(request, result_path)
            return redirect('classifier:process_file')
        else:
            print('invalid')
    else:
        form = UploadFileForm()
    return render(request, 'classifier/uploading_files.html', {'form': form})


def predict_csv(file):
    """This function takes in a csv file and predict the labels for each row and return a new csv file with the result"""
    df = pd.read_csv(file, header=None, names=['text'])
    df['predicted_label'] = classifier.predict(df['text'])
    df['predicted_label'].replace({0: 'Ham', 1: 'Spam'}, inplace=True)
    df['probability_spam'] = classifier.predict_proba(df['text'])[:, 1]
    print(classifier.predict_proba(df['text'])[:, 1])
    print(df.head())

    print('hadaf')
    return df


from django.http import FileResponse


def download_file(request, file_path):
    response = FileResponse(open(file_path, 'rb'), content_type='text/plain')
    response['Content-Disposition'] = f'attachment; filename="{file_path.name}"'
    return response
