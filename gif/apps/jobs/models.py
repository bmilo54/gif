import os

from django.db import models
from django.utils.text import get_valid_filename
from core.abstract_models import TimeStampedModel
from apps.projects.models import Project, DetectionObject
from PIL import Image
from .choices import STATUS_CHOICES


def generate_gif_file(instance, filename):
    filename = get_valid_filename(os.path.basename(filename))
    project_id = instance.project.project_id or "temp"
    version = instance.version or 1
    return f"Project/{project_id}/GIF/{version}/{filename}"


class AnimationJob(TimeStampedModel):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, verbose_name="Project", related_name="animations")
    selected_objects = models.ManyToManyField(DetectionObject, verbose_name="Detection Objects", related_name="animation_jobs")

    status = models.CharField(verbose_name="Status", max_length=255, choices=STATUS_CHOICES, default='pending')
    version = models.PositiveIntegerField(verbose_name="Version", default=1)
    task_id = models.CharField(verbose_name="Task ID", max_length=255, blank=True, null=True)

    gif_file = models.FileField(verbose_name="GIF File", upload_to=generate_gif_file, blank=True, null=True)

    # -- Extra GIF Metadata (Optional) --
    frame_count = models.PositiveIntegerField(verbose_name="Frame Count", blank=True, null=True)
    file_size = models.PositiveIntegerField(verbose_name="File Size", blank=True, null=True)

    def __str__(self):
        return f"Animation Job {self.id}"

    class Meta:
        verbose_name = "Animation Job"
        verbose_name_plural = "Animation Jobs"
        unique_together = ['project', 'version']
        ordering = ['-created']