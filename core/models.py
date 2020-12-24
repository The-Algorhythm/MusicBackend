from django.db import models


class User(models.Model):
    spotify_id = models.CharField(max_length=200)
    profile_vec = models.JSONField(default=list)


class UserActivity(models.Model):
    class ActivityType(models.TextChoices):
        LIKE = 'LK'
        SHARE = 'SH'
        OPEN = 'OP'
        LISTEN_LENGTH = 'LL'

    activity_type = models.CharField(
        max_length=2,
        choices=ActivityType.choices,
    )
    user = models.ForeignKey('User', on_delete=models.CASCADE)
    spotify_song_id = models.CharField(max_length=200)
    data = models.JSONField(default=dict)
