from django.urls import path

from . import views

app_name = 'classifier'
urlpatterns = [
    path('', views.index, name='predict'),
    path('process_file/', views.process_file, name='process_file'),
]
