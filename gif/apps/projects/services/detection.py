import logging

from django.conf import settings
from django.db import transaction

from ..models import DetectionObject
from .detectors import get_object_detector, get_text_detector
from .preprocessing import load_preprocessed_image

logger = logging.getLogger(__name__)


def _intersection_over_union(a, b):
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.x + a.width, b.x + b.width)
    bottom = min(a.y + a.height, b.y + b.height)

    if right <= left or bottom <= top:
        return 0.0

    overlap = (right - left) * (bottom - top)
    union = (a.width * a.height) + (b.width * b.height) - overlap
    return overlap / union if union else 0.0


def merge_detections(detections, min_confidence=None, iou_threshold=None):
    """
    Combine YOLO and OCR results into one list.

    Drops anything below the confidence floor, then removes near-duplicate
    boxes of the same label, keeping the most confident. Boxes from different
    labels are always kept, even when they overlap: a text box sitting on top
    of an object is a legitimate pair of things to animate.
    """
    if min_confidence is None:
        min_confidence = settings.DETECTION_MIN_CONFIDENCE
    if iou_threshold is None:
        iou_threshold = settings.DETECTION_IOU_THRESHOLD

    candidates = sorted(
        (d.clamped() for d in detections if d.confidence >= min_confidence),
        key=lambda d: d.confidence,
        reverse=True,
    )

    merged = []
    for candidate in candidates:
        duplicate = any(
            kept.label == candidate.label
            and _intersection_over_union(kept, candidate) >= iou_threshold
            for kept in merged
        )
        if not duplicate:
            merged.append(candidate)

    return merged


def detect(image):
    """Run both engines over a preprocessed image and merge the results."""
    detections = []
    detections.extend(get_object_detector()(image))
    detections.extend(get_text_detector()(image))
    return merge_detections(detections)


@transaction.atomic
def run_detection(project):
    """
    Detect objects and text on a project image and persist the results.

    Replaces any previous detections so re-running is idempotent. Returns the
    created `DetectionObject` rows.
    """
    if not project.image:
        raise ValueError("Project has no image to run detection on.")

    image = load_preprocessed_image(project.image)
    detections = detect(image)

    project.detections.all().delete()
    created = DetectionObject.objects.bulk_create([
        DetectionObject(
            project=project,
            label=detection.label[:255],
            confidence=detection.confidence,
            source=detection.source,
            x=detection.x,
            y=detection.y,
            width=detection.width,
            height=detection.height,
            text_content=detection.text_content[:1000] or None,
        )
        for detection in detections
    ])

    logger.info(
        "Detection complete for project %s: %s objects", project.project_id, len(created)
    )
    return created
