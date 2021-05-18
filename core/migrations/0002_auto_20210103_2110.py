# Generated by Django 3.1.2 on 2021-01-04 02:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='useractivity',
            name='spotify_song_id',
        ),
        migrations.AddField(
            model_name='useractivity',
            name='spotify_id',
            field=models.TextField(default=''),
        ),
        migrations.AlterField(
            model_name='useractivity',
            name='activity_type',
            field=models.CharField(choices=[('LK', 'Like'), ('SH', 'Share'), ('OP', 'Open'), ('LL', 'Listen Length'), ('DL', 'Dislike')], max_length=2),
        ),
    ]