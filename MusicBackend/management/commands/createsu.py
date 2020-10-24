from django.core.management.base import BaseCommand
from django.contrib.auth.models import User

import os


class Command(BaseCommand):

    def handle(self, *args, **options):
        if not User.objects.filter(username="admin").exists():
            User.objects.create_superuser(os.getenv('ADMIN_USERNAME'), os.getenv('ADMIN_EMAIL'), os.getenv('ADMIN_PASSWORD'))
