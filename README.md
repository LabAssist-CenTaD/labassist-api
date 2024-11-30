# labassist-api
AI-powered analysis tools for scientific experiments

Guide on setting up pipenv:
1. Run `pip install pipenv`
2. Run `pipenv install`

Required tools:
https://www.erlang.org/downloads
https://www.rabbitmq.com/docs/install-windows
https://www.apachefriends.org/download.html (for sql)

Command to launch celery:
`celery -A run.celery_app worker --loglevel=info --pool=gevent --concurrency=8`