from django.http import HttpResponse
from django.shortcuts import render

from classifier.cleaning import NlpDF, sms


def index(request):
    print(sms)
    return render(request, 'classifier/index.html')
