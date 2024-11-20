# Import the Celery library
import os
from celery import Celery

# os.environ.setdefault('FORKED_BY_MULTIPROCESSING', '1')

# Import the time module for the sleep function
import time

# Create a Celery instance named 'app'
app = Celery(
    'tasks',
    backend='db+sqlite:///results.db', 
    broker='amqp://guest@localhost//')

# Define a Celery task named 'add'
@app.task()
def simple_task():
    return "Test Successful"