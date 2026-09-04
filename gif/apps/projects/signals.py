import os
from django.db.models.signals import post_delete
from django.dispatch import receiver
from .models import Project

@receiver(post_delete, sender=Project)
def delete_project_files(sender, instance, **kwargs):
    """
    Deletes the associated image file from the filesystem
    when a Project is deleted.
    """
    if instance.image:
        if os.path.isfile(instance.image.path):
            os.remove(instance.image.path)
