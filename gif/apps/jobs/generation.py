import logging
import os
import shutil
import tempfile

from django.conf import settings
from django.core.files.base import ContentFile

from apps.projects.services.preprocessing import load_preprocessed_image
from apps.projects.services.segmentation import (
    inpaint_masked,
    segment_characters,
    segment_ui_cutouts,
)

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
            'color': det.get('color') or None,
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
    People: existing SAM character layers, original poster (unchanged).
    Cards: separate UI cut-outs so zoom follows the gold frame, not a rectangle.
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

            characters, person_mask, leftover = segment_characters(
                image, detections, tmp
            )
            cutouts, cutout_mask, leftover = segment_ui_cutouts(
                image, leftover, tmp, person_mask=person_mask
            )

            # Inpaint only card holes. Person SAM still composites on the
            # original poster pixels, same as before.
            if cutouts:
                background = inpaint_masked(image, cutout_mask, protect_mask=person_mask)
                logger.info('Segmented %d UI cut-out(s) for job %s', len(cutouts), job.pk)
            else:
                background = image.convert('RGB')
            background.save(poster_path, format='PNG')

            if characters:
                logger.info('Segmented %d character(s) for job %s', len(characters), job.pk)

            regions_payload = _regions_payload(leftover)
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
            cutouts_payload = [
                {
                    'src': f'ui_{c.character_index}.png',
                    'index': c.character_index,
                    'bbox': c.bbox_norm,
                    'effects': c.effects,
                    'color': c.source_region.get('color'),
                    'source': (c.source_region.get('source') or 'card').lower(),
                    'label': c.source_region.get('label') or '',
                }
                for c in cutouts
            ]

            render_promo_video(
                poster_path,
                regions=regions_payload,
                width=image.width,
                height=image.height,
                fps=fps,
                frame_count=frame_count,
                output_mp4=mp4_path,
                characters=characters_payload,
                cutouts=cutouts_payload,
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
