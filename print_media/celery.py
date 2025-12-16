import os
from celery import Celery

# Set the default Django settings module.
# Based on your folder structure, it is 'print_media.settings'
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'print_media.settings')

app = Celery('print_media')

# Load config from Django settings, using CELERY_ namespace
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load tasks from all registered apps
app.autodiscover_tasks()

# GPU Optimization: Prevent worker from taking too many tasks at once
app.conf.worker_prefetch_multiplier = 1 
app.conf.task_acks_late = True