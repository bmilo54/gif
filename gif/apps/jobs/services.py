import json

from django.core.exceptions import ValidationError
from django.db import transaction

from apps.projects.choices import SOURCE_MANUAL
from apps.projects.models import DetectionObject

from .models import AnimationJob


def _clamp_region(region):
    try:
        x = float(region['x'])
        y = float(region['y'])
        width = float(region['width'])
        height = float(region['height'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError('Each drawn region needs x, y, width and height.') from exc

    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    width = min(max(width, 0.0), 1.0 - x)
    height = min(max(height, 0.0), 1.0 - y)

    if width < 0.01 or height < 0.01:
        raise ValidationError('Drawn regions are too small. Drag a larger box.')

    return {'x': x, 'y': y, 'width': width, 'height': height}


def parse_detection_ids(raw):
    if not raw or not str(raw).strip():
        return []

    ids = []
    for part in str(raw).split(','):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError as exc:
            raise ValidationError('Selected detections are invalid.') from exc
    return ids


def parse_manual_regions(raw):
    if not raw or not str(raw).strip():
        return []

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError('Drawn regions could not be read.') from exc

    if not isinstance(payload, list):
        raise ValidationError('Drawn regions must be a list.')

    return [_clamp_region(region) for region in payload]


@transaction.atomic
def create_animation_job(project, detection_ids, manual_regions):
    """
    Persist an AnimationJob for this project.

    Clicked detections must already belong to the project. Drawn regions are
    stored as DetectionObject rows with source='manual' so later GIF frames
    can treat them the same as YOLO/OCR boxes. Version is the next integer
    for that project.
    """
    unique_ids = list(dict.fromkeys(detection_ids))
    detections = list(project.detections.filter(pk__in=unique_ids))
    if len(detections) != len(unique_ids):
        raise ValidationError('One or more selected detections do not belong to this project.')

    if not detections and not manual_regions:
        raise ValidationError('Select at least one box, or draw a region around something YOLO missed.')

    manual_objects = [
        DetectionObject(
            project=project,
            label='Manual region',
            confidence=1.0,
            source=SOURCE_MANUAL,
            x=region['x'],
            y=region['y'],
            width=region['width'],
            height=region['height'],
        )
        for region in manual_regions
    ]
    if manual_objects:
        created_manual = DetectionObject.objects.bulk_create(manual_objects)
        detections = detections + list(created_manual)

    last_version = (
        AnimationJob.objects.select_for_update()
        .filter(project=project)
        .order_by('-version')
        .values_list('version', flat=True)
        .first()
    )
    job = AnimationJob.objects.create(
        project=project,
        version=(last_version or 0) + 1,
        status='pending',
    )
    job.selected_objects.set(detections)
    return job
