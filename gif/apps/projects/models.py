import os
from datetime import datetime

from django.db import models
from django.dispatch import receiver
from django.utils.text import get_valid_filename
from core.abstract_models import TimeStampedModel
from PIL import Image
from .choices import SOURCE_CHOICES
from core.utils import generate_thumbnail


def generate_project_image(instance, filename):
    filename = get_valid_filename(os.path.basename(filename))
    project_id = instance.project_id or "temp"
    return f"Project/{project_id}/{filename}"


class Project(TimeStampedModel):
    project_id = models.CharField(verbose_name="Project ID", max_length=255, blank=True, null=True)
    image = models.ImageField(
        verbose_name="Image",
        upload_to=generate_project_image,
        blank=True,
        null=True,
    )

    def __str__(self):
        return f"Project {self.project_id}"

    class Meta:
        verbose_name = "Project"
        verbose_name_plural = "Projects"

    def get_image_thumbnail(self):
        if not self.image:
            return None
        return generate_thumbnail(self.image, 'x80')

    def save(self, *args, **kwargs):
        # ImageField.upload_to runs before post_save assigns project_id.
        # Hold the file until the ID exists so it lands in Project/{project_id}/.
        is_new = self.pk is None
        pending_image = None
        if is_new and self.image:
            pending_image = self.image
            self.image = None

        super(Project, self).save(*args, **kwargs)

        if pending_image:
            self.image = pending_image
            super(Project, self).save(update_fields=["image"])

    

@receiver(models.signals.post_save, sender=Project)
def Project_post_create(sender, instance, created, **kwargs):
    if created and instance.project_id is None:
        current_year = datetime.now().year
        prefix = "PJ"

        current_year_id = prefix + str(current_year)[2:4]
        running_number = f"{instance.pk:06d}"
        instance.project_id = f"{current_year_id}{running_number}"
        instance.save(update_fields=['project_id'])


class DetectionObject(TimeStampedModel):
    """
    A single YOLO or OCR hit on a project image.

    Bounding boxes are stored normalised (0-1) relative to the image, not in
    pixels, so they stay valid regardless of the resolution detection ran at
    or the size the browser renders the image.
    """

    project = models.ForeignKey(Project, on_delete=models.CASCADE, verbose_name="Project", related_name="detections")

    # Detection Label
    label = models.CharField(verbose_name="Label", max_length=255, blank=True, null=True)
    confidence = models.FloatField(verbose_name="Confidence", default=0.0)
    source = models.CharField(verbose_name="Source", max_length=255, choices=SOURCE_CHOICES, blank=True, null=True)

    # Bounding Box
    x = models.FloatField(verbose_name="X", default=0.0)
    y = models.FloatField(verbose_name="Y", default=0.0)
    width = models.FloatField(verbose_name="Width", default=0.0)
    height = models.FloatField(verbose_name="Height", default=0.0)

    # OCR Text
    text_content = models.CharField(verbose_name="Text Content", max_length=1000, blank=True, null=True)


    def __str__(self):
        return f"{self.label} (conf: {self.confidence:.2f}) in Project {self.project.project_id}"

    class Meta:
        verbose_name = "Detection Object"
        verbose_name_plural = "Detection Objects"