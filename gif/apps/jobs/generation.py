import io
import logging
import math

import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from PIL import Image, ImageChops, ImageEnhance, ImageFilter

from apps.projects.services.preprocessing import load_preprocessed_image

from .models import AnimationJob

logger = logging.getLogger(__name__)

STATUS_PENDING = 'pending'
STATUS_PROCESSING = 'processing'
STATUS_COMPLETED = 'completed'
STATUS_FAILED = 'failed'


def _pixel_box(detection, size):
    width, height = size
    left = int(round(detection.x * width))
    top = int(round(detection.y * height))
    box_width = max(1, int(round(detection.width * width)))
    box_height = max(1, int(round(detection.height * height)))

    left = min(max(left, 0), width - 1)
    top = min(max(top, 0), height - 1)
    box_width = min(box_width, width - left)
    box_height = min(box_height, height - top)
    return left, top, box_width, box_height


def _feather_mask(width, height, feather):
    """Opaque centre with a blurred edge so the effect does not look like a hard rectangle."""
    feather = max(1, min(feather, max(1, width // 3), max(1, height // 3)))
    mask = Image.new('L', (width, height), 0)
    inset = Image.new('L', (max(1, width - 2 * feather), max(1, height - 2 * feather)), 255)
    mask.paste(inset, (feather, feather))
    return mask.filter(ImageFilter.GaussianBlur(feather))


def _shine_mask(width, height, progress, width_frac):
    """Diagonal highlight that travels across the region as progress goes 0 → 1."""
    band = max(4, int(max(width, height) * width_frac))
    span = width + height + 2 * band
    center = progress * span - band
    rows, cols = np.mgrid[0:height, 0:width]
    falloff = np.clip(1.0 - np.abs((cols + rows) - center) / band, 0.0, 1.0) ** 2
    return Image.fromarray((falloff * 255).astype(np.uint8), mode='L')


def _animate_region(crop, wave, progress, glow, shine_width):
    """
    Keep the original pixels in place. Brighten them and screen a gold streak
    so text and characters glitter instead of being torn out as a moving box.
    """
    lit = ImageEnhance.Brightness(crop).enhance(1.0 + glow * wave)
    gold = Image.new('RGB', crop.size, (255, 214, 96))
    streak = Image.composite(
        gold,
        Image.new('RGB', crop.size, (0, 0, 0)),
        _shine_mask(crop.width, crop.height, progress, shine_width),
    )
    return ImageChops.screen(lit, streak)


def render_gif_bytes(image, detections, *, frame_count=None, duration_ms=None, glow=None, shine_width=None):
    """
    Build a looping GIF with a glow pulse and shine sweep on each selected region.

    The rest of the artwork stays still. That avoids the rectangular pop/bounce
    look you get from scaling cropped boxes on a composite promo image.
    """
    frame_count = frame_count or settings.GIF_FRAME_COUNT
    duration_ms = duration_ms or settings.GIF_DURATION_MS
    glow = glow if glow is not None else settings.GIF_GLOW
    shine_width = shine_width if shine_width is not None else settings.GIF_SHINE_WIDTH
    feather = settings.GIF_FEATHER

    regions = []
    for detection in detections:
        left, top, box_width, box_height = _pixel_box(detection, image.size)
        crop = image.crop((left, top, left + box_width, top + box_height)).convert('RGB')
        regions.append({
            'crop': crop,
            'left': left,
            'top': top,
            'mask': _feather_mask(box_width, box_height, feather),
        })

    frames = []
    for index in range(frame_count):
        wave = 0.5 * (1.0 - math.cos(2.0 * math.pi * index / frame_count))
        progress = index / float(frame_count)
        frame = image.convert('RGB')
        for region in regions:
            animated = _animate_region(
                region['crop'],
                wave,
                progress,
                glow,
                shine_width,
            )
            frame.paste(animated, (region['left'], region['top']), region['mask'])
        frames.append(frame)

    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return buffer.getvalue(), frame_count


def generate_gif(job):
    """
    Render the GIF for an AnimationJob and store it on gif_file.

    Uses the project image and the job's selected boxes. On success the job is
    completed; on failure it is marked failed and the exception is re-raised
    so the view can show a message.
    """
    if not isinstance(job, AnimationJob):
        job = AnimationJob.objects.select_related('project').prefetch_related('selected_objects').get(pk=job)

    job.status = STATUS_PROCESSING
    job.save(update_fields=['status'])

    try:
        project = job.project
        if not project.image:
            raise ValueError('Project has no image to animate.')

        detections = list(job.selected_objects.all())
        if not detections:
            raise ValueError('Animation job has no selected regions.')

        image = load_preprocessed_image(project.image, max_side=settings.GIF_MAX_SIDE)
        payload, frame_count = render_gif_bytes(image, detections)
        filename = f"{project.project_id or 'project'}_v{job.version}.gif"
        if job.gif_file:
            job.gif_file.delete(save=False)
        job.gif_file.save(filename, ContentFile(payload), save=False)
        job.frame_count = frame_count
        job.file_size = len(payload)
        job.status = STATUS_COMPLETED
        job.save(update_fields=['gif_file', 'frame_count', 'file_size', 'status'])
        logger.info(
            "GIF generated for %s v%s (%s frames, %s bytes)",
            project.project_id,
            job.version,
            frame_count,
            len(payload),
        )
        return job
    except Exception:
        job.status = STATUS_FAILED
        job.save(update_fields=['status'])
        logger.exception("GIF generation failed for job %s", job.pk)
        raise
