from celery.utils import uuid
from celery import chord

class ProgressChord(chord):

    def __call__(self, body=None, **kwargs):
        _chord = self.type
        body = (body or self.kwargs['body']).clone()
        kwargs = dict(self.kwargs, body=body, **kwargs)
        if _chord.app.conf.CELERY_ALWAYS_EAGER:
            return self.apply((), kwargs)
        callback_id = body.options.setdefault('task_id', uuid())
        r= _chord(**kwargs)
        return _chord.AsyncResult(callback_id), r