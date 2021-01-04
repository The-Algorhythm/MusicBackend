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
        DISLIKE = 'DL'

    class ObjectType(models.TextChoices):
        TRACK = 'TR'
        ARTIST = 'AR'
        ALBUM = 'AL'
        PLAYLIST = 'PL'

    activity_type = models.CharField(
        max_length=2,
        choices=ActivityType.choices,
    )
    object_type = models.CharField(
        max_length=2,
        choices=ObjectType.choices,
        default=ObjectType.TRACK
    )
    user = models.ForeignKey('User', on_delete=models.CASCADE)
    spotify_id = models.CharField(max_length=22)
    data = models.JSONField(default=dict)
