from django.db import models


class Message(models.Model):
    text = models.TextField()
    label = models.IntegerField()

    def __str__(self):
        return f'{self.text}: {self.text}'
