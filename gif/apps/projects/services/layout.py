"""
Turn word-level OCR boxes into selectable UI regions (cards, buttons, title).

Promo creatives usually have stacked offer cards and a bottom row of pill
buttons. PaddleOCR only sees the glyphs inside those, so this step unions
nearby text into the surrounding card or button.
"""

from ..choices import SOURCE_BUTTON, SOURCE_CARD, SOURCE_TITLE
from .detectors import Detection


def _center_y(det):
    return det.y + det.height / 2.0


def _union(dets):
    x1 = min(item.x for item in dets)
    y1 = min(item.y for item in dets)
    x2 = max(item.x + item.width for item in dets)
    y2 = max(item.y + item.height for item in dets)
    return x1, y1, x2, y2


def _clamp_box(x, y, w, h):
    x = min(max(x, 0.0), 0.98)
    y = min(max(y, 0.0), 0.98)
    w = min(max(w, 0.02), 1.0 - x)
    h = min(max(h, 0.02), 1.0 - y)
    return x, y, w, h


def _is_heading(det):
    letters = sum(ch.isalpha() for ch in (det.label or ''))
    return letters >= 6 and det.width >= 0.15 and det.height <= 0.12


def _heading_label(members, source=None):
    if source == SOURCE_TITLE:
        return sorted(members, key=lambda item: item.width * item.height, reverse=True)[0].label
    headings = [item for item in members if _is_heading(item)]
    if headings:
        return sorted(headings, key=lambda item: item.y)[0].label
    longest = sorted(members, key=lambda item: len(item.label or ''), reverse=True)[0]
    return longest.label


def _region(source, members, x, y, w, h):
    x, y, w, h = _clamp_box(x, y, w, h)
    label = _heading_label(members, source)
    text = ' '.join(item.text_content or item.label or '' for item in members)
    confidence = max(item.confidence for item in members)
    return Detection(
        label=label,
        confidence=confidence,
        source=source,
        x=x,
        y=y,
        width=w,
        height=h,
        text_content=text,
    )


def _button_row(ocr):
    """Horizontal row of short labels near the bottom → equal pill buttons."""
    candidates = [
        item for item in ocr
        if item.height <= 0.08 and _center_y(item) >= 0.78
    ]
    if len(candidates) < 2:
        return [], ocr

    candidates = sorted(candidates, key=_center_y)
    best = [candidates[0]]
    for item in candidates[1:]:
        if abs(_center_y(item) - _center_y(best[0])) <= 0.04:
            best.append(item)
    if len(best) < 2:
        return [], ocr

    best = sorted(best, key=lambda item: item.x)
    consumed = {id(item) for item in best}
    leftover = [item for item in ocr if id(item) not in consumed]

    heights = sorted(item.height for item in best)
    median_h = heights[len(heights) // 2]
    centers = sorted(_center_y(item) for item in best)
    median_cy = centers[len(centers) // 2]
    pad_y = median_h * 0.70
    icon = median_h * 1.90
    height = median_h + 2 * pad_y
    y = median_cy - height / 2.0

    buttons = []
    for index, item in enumerate(best):
        x = item.x - icon
        w = item.width + icon + median_h * 0.55
        if index + 1 < len(best):
            next_left = best[index + 1].x - icon
            w = min(w, next_left - x - 0.012)
        if index > 0:
            prev_right = buttons[-1].x + buttons[-1].width
            if x < prev_right + 0.008:
                x = prev_right + 0.008
                w = max(0.04, w - (x - (item.x - icon)))
        buttons.append(_region(SOURCE_BUTTON, [item], x, y, w, height))
    return buttons, leftover


def _title_and_cards(ocr):
    if not ocr:
        return []

    headings = sorted(
        (
            item for item in ocr
            if _is_heading(item) and _center_y(item) < 0.86
        ),
        key=lambda item: item.y,
    )

    column = [
        item for item in headings
        if item.x <= 0.45 and item.y >= 0.26
    ]
    if len(column) < 2:
        column = []

    regions = []
    assigned = set()
    pad_x, pad_y = 0.010, 0.010

    for index, heading in enumerate(column):
        y_top = heading.y
        y_limit = column[index + 1].y - 0.01 if index + 1 < len(column) else 0.88
        heading_right = heading.x + heading.width
        members = [
            item for item in ocr
            if item.y + item.height * 0.35 >= y_top - 0.01
            and item.y <= y_limit
            and item.x + item.width * 0.5 <= heading_right + 0.06
        ]
        if not members:
            members = [heading]
        assigned.update(id(item) for item in members)

        x1, y1, x2, y2 = _union(members)
        left_icons = [item for item in members if item.x + item.width * 0.5 < heading.x]
        if left_icons:
            x1 = min(item.x for item in left_icons)
        else:
            x1 = min(x1, heading.x - min(0.13, heading.height * 3.5))
        x1 -= pad_x
        y1 -= pad_y
        x2 = min(max(x2, heading_right) + pad_x, 0.56)
        y2 += pad_y
        if index + 1 < len(column):
            y2 = min(y2, column[index + 1].y - 0.008)
        regions.append(_region(SOURCE_CARD, members, x1, y1, x2 - x1, y2 - y1))

    if len(regions) >= 2:
        col_left = min(item.x for item in regions)
        col_right = min(0.56, max(item.x + item.width for item in regions))
        regions = [
            Detection(
                label=item.label,
                confidence=item.confidence,
                source=SOURCE_CARD,
                x=col_left,
                y=item.y,
                width=col_right - col_left,
                height=item.height,
                text_content=item.text_content,
            )
            for item in regions
        ]

    leftover = [item for item in ocr if id(item) not in assigned]
    if leftover:
        x1, y1, x2, y2 = _union(leftover)
        if y1 < 0.28:
            card_top = min((item.y for item in regions if item.source == SOURCE_CARD), default=y2)
            y2 = min(y2, card_top - 0.008)
            regions.append(_region(
                SOURCE_TITLE,
                leftover,
                x1 - pad_x,
                y1 - pad_y,
                (x2 - x1) + 2 * pad_x,
                (y2 - y1) + pad_y,
            ))
        else:
            regions.extend(leftover)
    return regions


def group_ui_regions(ocr_detections):
    """
    Replace word-level OCR with card / button / title regions.

    YOLO and manual boxes are not passed in. Word-level OCR is still returned
    separately by the detection pipeline so both the grouped region and the
    individual text can be selected.
    """
    if len(ocr_detections) < 2:
        return list(ocr_detections)

    buttons, rest = _button_row(ocr_detections)
    regions = buttons + _title_and_cards(rest)
    return regions or list(ocr_detections)
