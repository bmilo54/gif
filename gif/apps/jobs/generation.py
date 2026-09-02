import logging
import os
import shutil
import tempfile

import cv2
import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from PIL import Image

from apps.projects.services.preprocessing import load_preprocessed_image
from apps.projects.services.segmentation import segment_characters

from .encoding import encode_gif_from_video
from .models import AnimationJob
from .remotion_render import remotion_available, render_promo_video

logger = logging.getLogger(__name__)

STATUS_PROCESSING = 'processing'
STATUS_COMPLETED = 'completed'
STATUS_FAILED = 'failed'


def _as_region(item):
    if isinstance(item, dict):
        return item
    return {
        'source': (getattr(item, 'source', None) or 'manual').lower(),
        'label': getattr(item, 'label', None) or '',
        'x': float(item.x),
        'y': float(item.y),
        'width': float(item.width),
        'height': float(item.height),
        'effects': [],
    }


def _regions_payload(detections):
    payload = []
    for item in detections:
        det = _as_region(item)
        payload.append({
            'source': (det.get('source') or 'manual').lower(),
            'label': det.get('label') or '',
            'x': float(det['x']),
            'y': float(det['y']),
            'width': float(det['width']),
            'height': float(det['height']),
            'effects': list(det.get('effects') or []),
        })
    return payload


def _save_job_outputs(job, gif_bytes, video_bytes, frame_count):
    project = job.project
    stem = f"{project.project_id or 'project'}_v{job.version}"
    if job.gif_file:
        job.gif_file.delete(save=False)
    if job.video_file:
        job.video_file.delete(save=False)
    job.gif_file.save(f'{stem}.gif', ContentFile(gif_bytes), save=False)
    if video_bytes:
        job.video_file.save(f'{stem}.mp4', ContentFile(video_bytes), save=False)
    else:
        job.video_file = None
    job.frame_count = frame_count
    job.file_size = len(gif_bytes)
    job.status = STATUS_COMPLETED
    job.save(update_fields=['gif_file', 'video_file', 'frame_count', 'file_size', 'status'])
    logger.info(
        "Animation generated for %s v%s (%s frames, gif=%s bytes, mp4=%s bytes)",
        project.project_id,
        job.version,
        frame_count,
        len(gif_bytes),
        len(video_bytes) if video_bytes else 0,
    )
    return job


def _get_depth_map(image):
    """Generate a depth map using Depth-Anything. Returns PIL Image or None."""
    try:
        from apps.projects.services.depth import get_depth_map
        return get_depth_map(image)
    except Exception as e:
        logger.warning("Depth map generation failed: %s", e)
        return None


def generate_gif(job):
    """
    Full pipeline:
      1. Load image
      2. SAM segment any person/character regions → per-character RGBA PNGs
      3. Inpaint their area from the background
      4. Generate Depth-Anything depth map for the inpainted background
      5. Render via Remotion (WebGL depth shader + CSS effects per region)
      6. Encode as GIF
    """
    if not isinstance(job, AnimationJob):
        job = AnimationJob.objects.select_related('project').prefetch_related(
            'selected_objects',
            'project__detections',
        ).get(pk=job)

    job.status = STATUS_PROCESSING
    job.save(update_fields=['status'])

    try:
        project = job.project
        if not project.image:
            raise ValueError('Project has no image to animate.')

        detections = job.get_regions()
        if not detections:
            raise ValueError('Animation job has no selected regions.')

        if not remotion_available():
            raise RuntimeError(
                'Remotion is required. Install Node.js and run npm install in the remotion/ folder.'
            )

        image = load_preprocessed_image(project.image, max_side=settings.GIF_MAX_SIDE)
        duration_ms = settings.GIF_DURATION_MS
        frame_count = settings.GIF_FRAME_COUNT
        fps = 1000.0 / float(duration_ms)

        tmp = tempfile.mkdtemp(prefix='remotion-job-')
        try:
            poster_path = os.path.join(tmp, 'poster.png')
            mp4_path = os.path.join(tmp, 'out.mp4')

            # ── Phase 1: SAM segmentation ──────────────────────────────────
            characters, combined_mask, unconsumed_regions = segment_characters(
                image, detections, tmp
            )

            # Do NOT inpaint the background. cv2.inpaint on large characters
            # creates terrible blurry smears. We keep the original image intact.
            # Lighting effects (glow, rim) on characters will overlay perfectly.
            background = image.convert('RGB')
            background.save(poster_path, format='PNG')
            
            if characters:
                logger.info('Segmented %d character(s) for job %s', len(characters), job.pk)
            else:
                logger.info('No person regions found.')

            # ── Phase 2: Depth Map ─────────────────────────────────────────
            depth_path = None
            depth_pil = _get_depth_map(background)
            if depth_pil is not None:
                depth_path = os.path.join(tmp, 'depth.png')
                depth_pil.save(depth_path, format='PNG')
                logger.info('Depth map generated for job %s', job.pk)

            # ── Phase 3: Build region/character payloads ───────────────────
            if characters:
                regions_payload = _regions_payload(unconsumed_regions)
            else:
                regions_payload = _regions_payload(detections)

            characters_payload = [
                {
                    'src': f'char_{c.character_index}.png',
                    'index': c.character_index,
                    'bbox': c.bbox_norm,
                    'effects': c.effects,
                    'color': c.source_region.get('color'),
                }
                for c in characters
            ]

            # ── Phase 4: Remotion render ───────────────────────────────────
            render_promo_video(
                poster_path,
                regions=regions_payload,
                width=image.width,
                height=image.height,
                fps=fps,
                frame_count=frame_count,
                output_mp4=mp4_path,
                characters=characters_payload,
                depth_map_path=depth_path,
            )

            with open(mp4_path, 'rb') as handle:
                video_bytes = handle.read()
            gif_bytes = encode_gif_from_video(mp4_path, fps)
            return _save_job_outputs(job, gif_bytes, video_bytes, frame_count)

        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    except Exception:
        job.status = STATUS_FAILED
        job.save(update_fields=['status'])
        logger.exception("GIF generation failed for job %s", job.pk)
        raise
