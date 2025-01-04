# labassist-api
AI-powered analysis tools for scientific experiments

Usage guide:
1. Install WSL2 and Docker desktop
2. Run `docker compose pull`
3. Run `docker compose up`

NOTE: The instructions below are only for running without docker

Guide on setting up pipenv:
1. Run `pip install pipenv`
2. Run `pipenv install`

Required tools:
https://www.erlang.org/downloads
https://www.rabbitmq.com/docs/install-windows
https://www.apachefriends.org/download.html (for sql)

Command to launch celery:
`celery -A run.celery_app worker --loglevel=info --pool=gevent`