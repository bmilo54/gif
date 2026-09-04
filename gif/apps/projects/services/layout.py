"""
Turn OCR boxes into selectable UI regions (cards, buttons, title).

Promo creatives usually have stacked offer cards and a bottom row of pill
buttons. Raw OCR only sees the glyphs, so this step unions nearby text into
the surrounding card or button — without flood-filling the poster into a grid.
"""

from ..choices import SOURCE_BUTTON, SOURCE_CARD, SOURCE_OCR, SOURCE_TITLE
from .detectors import Detection


def _center_x(det):
    return det.x + det.width / 2.0


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


def _alnum_count(text):
    return sum(ch.isalnum() for ch in (text or ''))


def _looks_like_text(det):
    raw = (det.text_content or det.label or '').strip()
    if _alnum_count(raw) < 2:
        return False
    if det.width < 0.02 or det.height < 0.010:
        return False
    if det.width * det.height < 0.00035:
        return False
    return True


def _is_heading(det):
    letters = sum(ch.isalpha() for ch in (det.label or ''))
    return letters >= 4 and det.width >= 0.06 and det.height <= 0.16


def _heading_label(members, source=None):
    parts = sorted(members, key=lambda item: (item.y, item.x))
    if source in (SOURCE_TITLE, SOURCE_CARD):
        headings = [item.label for item in parts if item.label and _is_heading(item)]
        joined = ' '.join(headings if headings else (item.label for item in parts if item.label))
        return (joined or parts[0].label).strip()
    headings = [item for item in parts if _is_heading(item)]
    if headings:
        return headings[0].label
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


def _as_ocr_line(members):
    x1, y1, x2, y2 = _union(members)
    return Detection(
        label=_heading_label(members),
        confidence=max(item.confidence for item in members),
        source=SOURCE_OCR,
        x=x1,
        y=y1,
        width=x2 - x1,
        height=y2 - y1,
        text_content=' '.join(item.text_content or item.label or '' for item in members),
    )


def _words_to_lines(ocr, y_tol=0.016, gap_x=0.055):
    """
    Merge glyphs that share a baseline into one line, then split a line when
    a large horizontal gap means two cards sit on the same row.
    """
    items = sorted(ocr, key=lambda det: (det.y, det.x))
    used = set()
    lines = []
    for seed in items:
        if id(seed) in used:
            continue
        band = [seed]
        used.add(id(seed))
        cy = _center_y(seed)
        for other in items:
            if id(other) in used:
                continue
            tol = max(y_tol, min(seed.height, other.height) * 0.65)
            if abs(_center_y(other) - cy) <= tol:
                band.append(other)
                used.add(id(other))
                cy = sum(_center_y(item) for item in band) / len(band)
        band = sorted(band, key=lambda item: item.x)
        chunk = [band[0]]
        for item in band[1:]:
            prev = chunk[-1]
            gap = item.x - (prev.x + prev.width)
            if gap > gap_x:
                lines.append(_as_ocr_line(chunk))
                chunk = [item]
            else:
                chunk.append(item)
        lines.append(_as_ocr_line(chunk))
    return lines


def _button_row(ocr):
    """Horizontal row of short labels near the bottom → equal pill buttons."""
    candidates = [
        item for item in ocr
        if 0.81 <= _center_y(item) <= 0.91
        and 0.08 <= item.width <= 0.42
        and item.height <= 0.08
        and _alnum_count(item.label) >= 4
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
            if abs(_center_y(item) - _center_y(seed)) <= 0.025:
                band.append(item)
            else:
                kept.append(item)
        remaining = kept
        if len(band) >= 2:
            clusters.append(band)
    if not clusters:
        return [], ocr
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
    skip = (
        'claim', 'together', 'rewards', 'get extra', 'more fun', 'more wins',
        'click', 'games rtp', 'stats', 'heypokies', 'safe &', 'exclusive',
        'experience', 'fast &', 'pokie',
    )
    return not any(token in label for token in skip)


def _in_column(det, min_x, max_right):
    cx = _center_x(det)
    return min_x <= cx <= max_right


def _cluster_stacked_headings(headings, max_gap=0.055):
    """WELCOME + DEPOSIT BONUS + 69% is one card, not three stacked titles."""
    if not headings:
        return []
    clusters = [[headings[0]]]
    for item in headings[1:]:
        prev = clusters[-1][-1]
        gap = item.y - (prev.y + prev.height)
        if gap <= max_gap:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return clusters


def _expand_card_chrome(x1, y1, x2, y2, *, min_x, max_right, y_ceiling, y_floor):
    """
    OCR only sees glyphs. Promo cards also have a gold frame and often a
    crown above the title — grow the box to cover that chrome.
    """
    text_w = max(x2 - x1, 0.04)
    text_h = max(y2 - y1, 0.03)
    pad_x = min(0.042, max(0.024, text_w * 0.16))
    pad_top = min(0.062, max(0.038, text_h * 0.32))
    pad_bottom = min(0.040, max(0.022, text_h * 0.16))
    x1 = max(min_x - 0.012, x1 - pad_x)
    x2 = min(max_right, x2 + pad_x)
    y1 = max(y_ceiling, y1 - pad_top)
    y2 = min(y_floor, y2 + pad_bottom)
    return x1, y1, x2, y2


def _offer_column(ocr, *, min_x, max_right=0.98):
    """Stacked bonus cards on one side. Nearby titles belong to the same card."""
    headings = sorted(
        (
            item for item in ocr
            if _is_offer_heading(item)
            and _in_column(item, min_x, max_right)
            and _center_y(item) < 0.82
        ),
        key=lambda item: item.y,
    )
    if not headings:
        return [], set()

    clusters = _cluster_stacked_headings(headings)
    regions = []
    assigned = set()
    for index, cluster in enumerate(clusters):
        y_top = cluster[0].y
        last = cluster[-1]
        if index + 1 < len(clusters):
            y_limit = clusters[index + 1][0].y - 0.010
        else:
            y_limit = min(0.82, last.y + last.height + 0.22)
        members = [
            item for item in ocr
            if _in_column(item, min_x, max_right)
            and item.y + item.height * 0.45 >= y_top - 0.04
            and item.y <= y_limit
        ]
        if not members:
            members = list(cluster)
        assigned.update(id(item) for item in members)
        x1, y1, x2, y2 = _union(members)
        prev_ceiling = 0.0 if index == 0 else (
            clusters[index - 1][-1].y + clusters[index - 1][-1].height
        )
        x1, y1, x2, y2 = _expand_card_chrome(
            x1, y1, x2, y2,
            min_x=min_x,
            max_right=max_right,
            y_ceiling=prev_ceiling + 0.006,
            y_floor=y_limit,
        )
        regions.append(_region(SOURCE_CARD, members, x1, y1, x2 - x1, y2 - y1))
    return regions, assigned


def _center_banner(ocr):
    """Wide headline across the middle (GAMES RTP STATS), one region not a grid."""
    lines = [
        item for item in ocr
        if item.width >= 0.16
        and 0.30 <= _center_y(item) <= 0.78
        and 0.22 <= _center_x(item) <= 0.72
    ]
    if not lines:
        return [], set()
    lines = sorted(lines, key=lambda item: item.y)
    band = [lines[0]]
    for item in lines[1:]:
        if item.y <= band[-1].y + band[-1].height + 0.04:
            band.append(item)
    x1, y1, x2, y2 = _union(band)
    if (x2 - x1) < 0.18:
        return [], set()
    return (
        [_region(SOURCE_TITLE, band, x1 - 0.008, y1 - 0.008, (x2 - x1) + 0.016, (y2 - y1) + 0.016)],
        {id(item) for item in band},
    )


def _logo_title(ocr, card_regions):
    logo = [
        item for item in ocr
        if item.y < 0.20 and _center_x(item) < 0.58
    ]
    if not logo:
        return []
    x1, y1, x2, y2 = _union(logo)
    if (x2 - x1) * (y2 - y1) > 0.28:
        return []
    left_card_top = min(
        (
            item.y for item in card_regions
            if item.source == SOURCE_CARD and _center_x(item) < 0.45
        ),
        default=1.0,
    )
    y2 = min(y2, left_card_top - 0.008)
    return [_region(
        SOURCE_TITLE,
        logo,
        x1 - 0.010,
        y1 - 0.010,
        (x2 - x1) + 0.020,
        max((y2 - y1) + 0.010, 0.04),
    )]


def _proximity_cluster(ocr, gap_x=0.025, gap_y=0.018):
    """
    Compact leftover groups only. Do not emit a partial cluster when the
    flood-fill hits a size cap — that is what painted a grid over the
    characters and the centre headline.
    """
    if not ocr:
        return []

    items = sorted(ocr, key=lambda det: (round(det.y / max(gap_y, 0.01)), det.x))
    clusters = []
    used = set()
    max_w, max_h = 0.42, 0.26

    for seed in items:
        if id(seed) in used:
            continue
        group = [seed]
        used.add(id(seed))
        changed = True
        overflow = False
        while changed:
            changed = False
            gx1, gy1, gx2, gy2 = _union(group)
            if (gx2 - gx1) > max_w or (gy2 - gy1) > max_h:
                overflow = True
                break
            for other in items:
                if id(other) in used:
                    continue
                ox1, oy1 = other.x, other.y
                ox2, oy2 = other.x + other.width, other.y + other.height
                if ox2 + gap_x >= gx1 and ox1 - gap_x <= gx2 and oy2 + gap_y >= gy1 and oy1 - gap_y <= gy2:
                    group.append(other)
                    used.add(id(other))
                    changed = True
        if overflow or len(group) < 2:
            for item in group:
                used.discard(id(item))
            used.add(id(seed))
            continue
        clusters.append(group)

    regions = []
    pad = 0.010
    for members in clusters:
        x1, y1, x2, y2 = _union(members)
        if (x2 - x1) > 0.50 or (y2 - y1) > 0.32:
            continue
        regions.append(_region(
            SOURCE_CARD,
            members,
            x1 - pad,
            y1 - pad,
            (x2 - x1) + 2 * pad,
            (y2 - y1) + 2 * pad,
        ))
    return regions


def group_ui_regions(ocr_detections):
    """
    Replace raw OCR with card / button / title regions.

    Word-level leftovers are filtered by the detection pipeline; this function
    only returns grouped UI boxes.
    """
    ocr = [item for item in ocr_detections if _looks_like_text(item)]
    if len(ocr) < 2:
        return list(ocr)

    buttons, rest = _button_row(ocr)
    lines = _words_to_lines(rest) if rest else []

    left_cards, left_ids = _offer_column(lines, min_x=0.0, max_right=0.42)
    unused = [item for item in lines if id(item) not in left_ids]
    right_cards, right_ids = _offer_column(unused, min_x=0.55, max_right=0.99)
    offer_cards = left_cards + right_cards

    def _inside_offer(item):
        cx, cy = _center_x(item), _center_y(item)
        return any(
            card.x <= cx <= card.x + card.width and card.y <= cy <= card.y + card.height
            for card in offer_cards
        )

    unused = [item for item in unused if id(item) not in right_ids and not _inside_offer(item)]

    banner, banner_ids = _center_banner(unused)
    unused = [item for item in unused if id(item) not in banner_ids]

    logo = _logo_title(
        [item for item in unused if item.y < 0.20],
        left_cards + right_cards,
    )
    logo_text = {item.text_content for item in unused if item.y < 0.20}
    if logo:
        unused = [
            item for item in unused
            if item.y >= 0.20 or item.text_content not in logo_text
        ]

    extra = _proximity_cluster(unused) if unused else []
    extra = [item for item in extra if _center_y(item) < 0.78]
    regions = buttons + left_cards + right_cards + banner + logo + extra
    return regions
