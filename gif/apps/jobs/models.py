import os

from django.db import models
from django.utils.text import get_valid_filename
from core.abstract_models import TimeStampedModel
from apps.projects.models import Project, DetectionObject
from .choices import (
    ANIMATION_TYPE_CHOICES,
    DEFAULT_ANIMATION_TYPES,
    STATUS_CHOICES,
)


def generate_gif_file(instance, filename):
    filename = get_valid_filename(os.path.basename(filename))
    project_id = instance.project.project_id or "temp"
    version = instance.version or 1
    return f"Project/{project_id}/GIF/{version}/{filename}"


class AnimationJob(TimeStampedModel):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, verbose_name="Project", related_name="animations")
    selected_objects = models.ManyToManyField(DetectionObject, verbose_name="Detection Objects", related_name="animation_jobs")

    status = models.CharField(verbose_name="Status", max_length=255, choices=STATUS_CHOICES, default='pending')
    animation_types = models.JSONField(
        verbose_name="Animation Types",
        default=list,
        blank=True,
        help_text="One or more effect keys applied in order.",
    )
    regions = models.JSONField(
        verbose_name="Regions",
        default=list,
        blank=True,
        help_text="Adjusted boxes used for GIF generation (normalised 0-1).",
    )
    version = models.PositiveIntegerField(verbose_name="Version", default=1)
    task_id = models.CharField(verbose_name="Task ID", max_length=255, blank=True, null=True)

    gif_file = models.FileField(verbose_name="GIF File", upload_to=generate_gif_file, blank=True, null=True)

    frame_count = models.PositiveIntegerField(verbose_name="Frame Count", blank=True, null=True)
    file_size = models.PositiveIntegerField(verbose_name="File Size", blank=True, null=True)

    def __str__(self):
        return f"Animation Job {self.id}"

    def gif_filename(self):
        if not self.gif_file:
            return ''
        return os.path.basename(self.gif_file.name)

    def get_animation_types(self):
        values = [item for item in (self.animation_types or []) if item]
        return values or list(DEFAULT_ANIMATION_TYPES)

    def get_animation_type_labels(self):
        lookup = dict(ANIMATION_TYPE_CHOICES)
        return [lookup.get(value, value) for value in self.get_animation_types()]

    def get_regions(self):
        if self.regions:
            return self.regions
        return [
            {
                'key': f'det-{item.pk}',
                'label': item.label or 'Region',
                'source': item.source or 'manual',
                'x': item.x,
                'y': item.y,
                'width': item.width,
                'height': item.height,
            }
            for item in self.selected_objects.all()
        ]

    class Meta:
        verbose_name = "Animation Job"
        verbose_name_plural = "Animation Jobs"
        unique_together = ['project', 'version']
        ordering = ['-created']
