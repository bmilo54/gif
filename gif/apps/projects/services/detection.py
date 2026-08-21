import logging

from django.conf import settings
from django.db import transaction

from ..choices import (
    SOURCE_BUTTON,
    SOURCE_CARD,
    SOURCE_MANUAL,
    SOURCE_OCR,
    SOURCE_TITLE,
    SOURCE_YOLO,
)
from ..models import DetectionObject
from .detectors import get_object_detector, get_text_detector
from .layout import group_ui_regions
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
            kept.source == candidate.source
            and kept.label == candidate.label
            and _intersection_over_union(kept, candidate) >= iou_threshold
            for kept in merged
        )
        if not duplicate:
            merged.append(candidate)

    # Stock YOLO (COCO) often tags UI chrome as random objects. Those blue
    # boxes sit on top of the real OCR hits and make the overlay look like
    # PaddleOCR is duplicated or shifted. Keep the text box; drop the YOLO
    # box when they overlap.
    text_like = [
        item for item in merged
        if item.source in (SOURCE_OCR, SOURCE_CARD, SOURCE_BUTTON, SOURCE_TITLE)
    ]
    others = []
    for item in merged:
        if item.source == SOURCE_YOLO and any(
            _intersection_over_union(item, text) >= 0.15 for text in text_like
        ):
            continue
        others.append(item)
    return others


def detect(image):
    """Run YOLO and OCR, and also group OCR words into title/card/button regions."""
    detections = []
    detections.extend(get_object_detector()(image))
    ocr = get_text_detector()(image)
    detections.extend(ocr)
    detections.extend(group_ui_regions(ocr))
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

    # Drop previous auto detections so Re-run does not stack a second copy.
    # Manual regions the user drew are kept.
    project.detections.exclude(source=SOURCE_MANUAL).delete()
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
