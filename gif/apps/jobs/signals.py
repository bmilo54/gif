import os
import shutil
from pathlib import Path
from django.db.models.signals import post_delete
from django.dispatch import receiver
from .models import AnimationJob

@receiver(post_delete, sender=AnimationJob)
def delete_job_files(sender, instance, **kwargs):
    """
    Deletes the associated MP4 and GIF files from the filesystem
    when an AnimationJob is deleted.
    """
    if instance.video_file:
        if os.path.isfile(instance.video_file.path):
            os.remove(instance.video_file.path)
            
    if instance.gif_file:
        if os.path.isfile(instance.gif_file.path):
            os.remove(instance.gif_file.path)

    # Also clean up the tmp working directory if it exists
    # The tmp dir is usually named based on job version/pk but might be harder to track.
    # We will just clean up the output files.
