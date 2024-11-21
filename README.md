# labassist-api
AI-powered analysis tools for scientific experiments

Required tools:
https://www.erlang.org/downloads
https://www.rabbitmq.com/docs/install-windows


celery -A run.celery_app worker --loglevel=info --pool=gevent --concurrency=8