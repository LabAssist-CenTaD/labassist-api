import os
import subprocess
from io import BytesIO
from flask import Flask
from celery import Celery, Task, shared_task
from celery.result import AsyncResult

def celery_init_app(app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask)
    celery_app.config_from_object(app.config["CELERY"])
    celery_app.set_default()
    app.extensions["celery"] = celery_app
    return celery_app

def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        CELERY=dict(
            broker_url="pyamqp:/guest@localhost/",
            # result_backend="redis://localhost",
            task_ignore_result=True,
        ),
    )
    app.config.from_prefixed_env()
    celery_app = celery_init_app(app)
    
    @celery_app.task(ignore_result=False)
    def add_together(a: int, b: int) -> int:
        return a + b
    
    @app.route("/add")
    def add():
        print("Adding 1 + 2")
        result = add_together.delay(1, 2)
        return {"task_id": result.id}
    
    @app.route("/result/<task_id>")
    def result(task_id):
        result = AsyncResult(task_id)
        return {
            "ready": result.ready(),
            "successful": result.successful(),
            "value": result.result if result.ready() else None,
        }
    
    return app

flask_app = create_app()
celery_app = flask_app.extensions["celery"]

# subprocess.Popen(["celery", "-A", '"test.celery_app"', "worker", "--loglevel=info"])

# flask_app.run()