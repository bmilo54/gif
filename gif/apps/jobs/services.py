import json

from django.core.exceptions import ValidationError
from django.db import transaction

from apps.projects.choices import SOURCE_MANUAL
from apps.projects.models import DetectionObject

from .choices import DEFAULT_ANIMATION_TYPES, ANIMATION_TYPE_CHOICES, ANIMATION_TYPE_LABELS
from .models import AnimationJob


def _clamp_region(region):
    try:
        x = float(region['x'])
        y = float(region['y'])
        width = float(region['width'])
        height = float(region['height'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError('Each region needs x, y, width and height.') from exc

    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    width = min(max(width, 0.0), 1.0 - x)
    height = min(max(height, 0.0), 1.0 - y)

    if width < 0.01 or height < 0.01:
        raise ValidationError('Regions are too small. Drag a larger box.')

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


def parse_animation_types(raw_list):
    allowed = {value for value, _label in ANIMATION_TYPE_CHOICES}
    if raw_list is None:
        return list(DEFAULT_ANIMATION_TYPES)

    values = []
    for item in raw_list:
        if item is None or str(item).strip() == '':
            continue
        item = str(item).strip()
        if item not in allowed:
            raise ValidationError('Unknown animation type.')
        if item not in values:
            values.append(item)
    return values or list(DEFAULT_ANIMATION_TYPES)


def parse_regions(raw):
    if not raw or not str(raw).strip():
        raise ValidationError('Adjust at least one region.')

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError('Adjusted regions could not be read.') from exc

    if not isinstance(payload, list) or not payload:
        raise ValidationError('Adjust at least one region.')

    regions = []
    allowed_effects = set(ANIMATION_TYPE_LABELS.keys())
    for item in payload:
        if not isinstance(item, dict):
            raise ValidationError('Each region must be an object.')
        box = _clamp_region(item)
        # Preserve per-region effects; silently ignore unknown keys.
        raw_effects = item.get('effects') or []
        effects = [e for e in raw_effects if e in allowed_effects]
        regions.append({
            'key': str(item.get('key') or '')[:64],
            'label': str(item.get('label') or 'Region')[:255],
            'source': str(item.get('source') or 'manual')[:32],
            'effects': effects,
            **box,
        })
    return regions


def snapshot_regions(detections, manual_regions):
    regions = []
    for detection in detections:
        regions.append({
            'key': f'det-{detection.pk}',
            'label': detection.label or detection.text_content or 'Region',
            'source': detection.source or 'manual',
            'x': detection.x,
            'y': detection.y,
            'width': detection.width,
            'height': detection.height,
            'effects': [],  # user fills this in on the adjust page
        })
    for index, region in enumerate(manual_regions):
        regions.append({
            'key': f'manual-{index}',
            'label': 'Manual region',
            'source': SOURCE_MANUAL,
            'effects': [],
            **region,
        })
    return regions


@transaction.atomic
def create_animation_job(project, detection_ids, manual_regions, animation_types=None):
    """
    Persist a pending AnimationJob. GIF is generated after the user adjusts
    boxes on the next page.
    """
    unique_ids = list(dict.fromkeys(detection_ids))
    detections = list(project.detections.filter(pk__in=unique_ids))
    if len(detections) != len(unique_ids):
        raise ValidationError('One or more selected detections do not belong to this project.')

    if not detections and not manual_regions:
        raise ValidationError('Select at least one box, or draw a region around something detection missed.')

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
        animation_types=animation_types or list(DEFAULT_ANIMATION_TYPES),
        regions=snapshot_regions(detections, []),
    )
    job.selected_objects.set(detections)
    return job


def save_job_adjustments(job, regions, animation_types):
    job.regions = regions
    job.animation_types = animation_types
    job.save(update_fields=['regions', 'animation_types'])
    return job
