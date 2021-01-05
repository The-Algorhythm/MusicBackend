from django.db import models
from django.utils import timezone


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

    def save(self, *args, **kwargs):
        """ Append created timestamp when a new entry is created """
        if not self.id:
            self.created = timezone.now()
        return super(UserActivity, self).save(*args, **kwargs)

    created = models.DateTimeField(default=timezone.now())
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
