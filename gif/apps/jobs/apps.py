from django.apps import AppConfig


class JobsConfig(AppConfig):
    name = 'apps.jobs'

    def ready(self):
        import apps.jobs.signals  # noqa: F401
