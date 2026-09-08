import logging

from django.conf import settings
from django.db import transaction

from ..choices import (
    SOURCE_BUTTON,
    SOURCE_CARD,
    SOURCE_MANUAL,
    SOURCE_OCR,
    SOURCE_PROP,
    SOURCE_TITLE,
    SOURCE_YOLO,
)
from ..models import DetectionObject
from .detectors import get_character_detector, get_object_detector, get_text_detector
from .layout import group_ui_regions
from .preprocessing import load_preprocessed_image
from .segmentation import detect_props

logger = logging.getLogger(__name__)


def _intersection_area(a, b):
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.x + a.width, b.x + b.width)
    bottom = min(a.y + a.height, b.y + b.height)
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def _intersection_over_union(a, b):
    overlap = _intersection_area(a, b)
    union = (a.width * a.height) + (b.width * b.height) - overlap
    return overlap / union if union else 0.0


def _box_center_in(inner, outer):
    cx = inner.x + inner.width / 2.0
    cy = inner.y + inner.height / 2.0
    return (
        outer.x <= cx <= outer.x + outer.width
        and outer.y <= cy <= outer.y + outer.height
    )


def _fraction_inside(inner, outer):
    area = inner.width * inner.height
    if area <= 0:
        return 0.0
    return _intersection_area(inner, outer) / area


class _Box:
    __slots__ = ('x', 'y', 'width', 'height')

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


def _inflate(item, pad=0.12):
    return _Box(
        item.x - item.width * pad,
        item.y - item.height * pad,
        item.width * (1 + 2 * pad),
        item.height * (1 + 2 * pad),
    )


def _hangs_under(inner, outer):
    """Coins/gifts often sit just below the % text, slightly outside the OCR box."""
    icx = inner.x + inner.width / 2.0
    if icx < outer.x or icx > outer.x + outer.width:
        return False
    outer_bottom = outer.y + outer.height
    inner_bottom = inner.y + inner.height
    return inner.y <= outer_bottom + 0.09 and inner_bottom >= outer_bottom - 0.05


def _prop_is_ui_art(item, region):
    padded = _inflate(region)
    return (
        _box_center_in(item, padded)
        or _fraction_inside(item, padded) >= 0.35
        or _hangs_under(item, region)
    )


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
        if item.source == SOURCE_YOLO and item.label != 'person':
            if any(_intersection_over_union(item, text) >= 0.15 for text in text_like):
                continue
        others.append(item)
    return _drop_props_inside_ui(others)


def _drop_props_inside_ui(detections):
    """Keep dragon/gift/coins on the character; drop props that are just card art.

    IoU is the wrong test: a small gift on a large bonus card has tiny IoU
    even when it sits fully inside the plaque, so it used to survive and get
    SAM-cut as a second layer on top of the card (jitter / melt).
    """
    ui = [
        item for item in detections
        if item.source in (SOURCE_CARD, SOURCE_BUTTON, SOURCE_TITLE)
    ]
    kept = []
    for item in detections:
        if item.source == SOURCE_PROP and any(
            _prop_is_ui_art(item, region) for region in ui
        ):
            continue
        kept.append(item)
    return kept


def detect(image):
    """Run GPT-4o characters, PaddleOCR text, UI grouping, and open-vocab props."""
    detections = []

    # GPT-4o (or other configured) object detector for characters/persons
    detections.extend(get_object_detector()(image))

    # YOLO-World character detector (disabled by default when GPT-4o is active)
    char_detector = get_character_detector()
    if char_detector is not None:
        detections.extend(char_detector(image))

    # OCR lines → cards / buttons / titles. Do not also keep every leftover
    # word: those are what painted a second grid over the characters.
    ocr = get_text_detector()(image)
    ui_groups = group_ui_regions(ocr)

    group_boxes = [(g.x, g.y, g.x + g.width, g.y + g.height) for g in ui_groups]

    def _covered_by_group(word):
        cx = word.x + word.width / 2
        cy = word.y + word.height / 2
        for gx1, gy1, gx2, gy2 in group_boxes:
            if gx1 <= cx <= gx2 and gy1 <= cy <= gy2:
                return True
        return False

    orphan_ocr = [
        word for word in ocr
        if not _covered_by_group(word)
        and word.width * word.height >= 0.006
        and sum(ch.isalnum() for ch in (word.label or '')) >= 4
    ]
    detections.extend(orphan_ocr)
    detections.extend(ui_groups)

    detections.extend(detect_props(image))
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
