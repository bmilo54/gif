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
        parts = sorted(members, key=lambda item: (item.y, item.x))
        joined = ' '.join(item.label for item in parts if item.label).strip()
        return joined or parts[0].label
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
        if 0.81 <= _center_y(item) <= 0.89
        and 0.10 <= item.width <= 0.40
        and item.height <= 0.06
        and sum(ch.isalpha() for ch in (item.label or '')) >= 4
    ]
    if len(candidates) < 2:
        return [], ocr

    remaining = sorted(candidates, key=_center_y)
    clusters = []
    while remaining:
        seed = remaining.pop(0)
        band = [seed]
        kept = []
        for item in remaining:
            if abs(_center_y(item) - _center_y(seed)) <= 0.02:
                band.append(item)
            else:
                kept.append(item)
        remaining = kept
        if len(band) >= 2:
            clusters.append(band)
    if not clusters:
        return [], ocr
    # Prefer the 3 trust badges over their smaller subtitle row underneath.
    threes = [band for band in clusters if len(band) >= 3]
    pool = threes or clusters
    best = min(pool, key=lambda band: sum(_center_y(item) for item in band) / len(band))

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


def _is_offer_heading(det):
    if not _is_heading(det):
        return False
    label = (det.label or '').lower()
    skip = ('claim', 'now', 'together', 'rewards', 'get extra', 'more fun', 'more wins')
    return not any(token in label for token in skip)


def _offer_column(ocr, *, min_x, max_right=0.98):
    """
    Stacked bonus cards on one side (this poster: right-hand neon panels).
    """
    headings = sorted(
        (
            item for item in ocr
            if _is_offer_heading(item)
            and item.x >= min_x
            and _center_y(item) < 0.82
        ),
        key=lambda item: item.y,
    )
    if len(headings) < 2:
        return [], set()

    regions = []
    assigned = set()
    pad_x, pad_y = 0.012, 0.012
    for index, heading in enumerate(headings):
        y_top = heading.y
        if index + 1 < len(headings):
            y_limit = headings[index + 1].y - 0.012
        else:
            y_limit = min(0.82, heading.y + heading.height + 0.22)
        members = [
            item for item in ocr
            if item.x + item.width * 0.35 >= min_x
            and item.y + item.height * 0.35 >= y_top - 0.012
            and item.y <= y_limit
        ]
        if not members:
            members = [heading]
        assigned.update(id(item) for item in members)
        x1, y1, x2, y2 = _union(members)
        x1 = min(x1, heading.x) - min(0.08, heading.height * 2.4) - pad_x
        x1 = max(min_x - 0.04, x1)
        y1 -= pad_y
        x2 = min(max_right, max(x2, heading.x + heading.width) + pad_x)
        y2 = min(y_limit, y2 + pad_y)
        regions.append(_region(SOURCE_CARD, members, x1, y1, x2 - x1, y2 - y1))
    return regions, assigned


def _title_and_cards(ocr):
    if not ocr:
        return []

    left_cards, left_assigned = _offer_column(
        [item for item in ocr if item.x + item.width * 0.5 <= 0.56],
        min_x=0.0,
        max_right=0.56,
    )
    # This creative puts the offer stack on the right.
    right_cards, right_assigned = _offer_column(
        [item for item in ocr if item.x >= 0.44],
        min_x=0.44,
        max_right=0.99,
    )

    if len(left_cards) >= 2:
        col_left = min(item.x for item in left_cards)
        col_right = min(0.56, max(item.x + item.width for item in left_cards))
        left_cards = [
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
            for item in left_cards
        ]
    else:
        left_cards, left_assigned = [], set()

    if len(right_cards) < 2:
        right_cards, right_assigned = [], set()

    regions = left_cards + right_cards
    assigned = left_assigned | right_assigned

    leftover = [item for item in ocr if id(item) not in assigned]
    logo = [
        item for item in leftover
        if item.y < 0.20 and item.x + item.width * 0.5 < 0.55
    ]
    if logo:
        x1, y1, x2, y2 = _union(logo)
        area = max(x2 - x1, 0) * max(y2 - y1, 0)
        if area <= 0.28:
            left_card_top = min(
                (
                    item.y for item in regions
                    if item.source == SOURCE_CARD
                    and item.x + item.width * 0.5 < 0.45
                ),
                default=1.0,
            )
            y2 = min(y2, left_card_top - 0.008)
            regions.append(_region(
                SOURCE_TITLE,
                logo,
                x1 - 0.010,
                y1 - 0.010,
                (x2 - x1) + 0.020,
                max((y2 - y1) + 0.010, 0.04),
            ))
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
