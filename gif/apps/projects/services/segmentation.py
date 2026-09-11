"""
Person / prop cut-outs for layered animation.

SAM 3 is used for open-vocabulary concepts when the weight is present.
On CPU we fall back to YOLO-World for those boxes, then SAM 2.1 for the
silhouette. GrabCut remains the last resort so GIF generation still runs
if weights are missing.
"""

from functools import lru_cache
import logging
import os

from django.conf import settings

from ..choices import SOURCE_PROP
from . import replicate_util
from .detectors import Detection

logger = logging.getLogger(__name__)

PERSON_CONCEPTS = ('person', 'man', 'woman', 'character')


def _device():
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except ImportError:
        pass
    return 'cpu'


@lru_cache(maxsize=1)
def _sam_interactive():
    if not getattr(settings, 'SAM_ENABLED', True):
        return None
    try:
        from ultralytics import SAM
    except ImportError:
        logger.warning('ultralytics is not installed; SAM cut-out disabled.')
        return None
    path = getattr(settings, 'SAM_MODEL', 'sam2.1_t.pt')
    try:
        model = SAM(path)
        logger.info('Loaded SAM interactive weights from %s', path)
        return model
    except Exception:
        logger.exception('Failed to load SAM model %s', path)
        return None


@lru_cache(maxsize=1)
def _sam3_semantic():
    if not getattr(settings, 'SAM3_ENABLED', False):
        return None
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
    except ImportError:
        logger.warning('SAM 3 predictor is not available in this ultralytics build.')
        return None
    path = getattr(settings, 'SAM3_MODEL', 'sam3.pt')
    try:
        predictor = SAM3SemanticPredictor(overrides={
            'model': path,
            'task': 'segment',
            'mode': 'predict',
            'conf': getattr(settings, 'SAM_MIN_CONFIDENCE', 0.30),
            'imgsz': 1008,
            'verbose': False,
        })
        logger.info('Loaded SAM 3 semantic weights from %s', path)
        return predictor
    except Exception:
        logger.exception('Failed to load SAM 3 model %s', path)
        return None


@lru_cache(maxsize=1)
def _world_detector():
    if not getattr(settings, 'SAM_CONCEPT_FALLBACK', True):
        return None
    try:
        from ultralytics import YOLOWorld
    except ImportError:
        return None
    path = getattr(settings, 'YOLOWORLD_MODEL', 'yolov8s-worldv2.pt')
    try:
        model = YOLOWorld(path)
        logger.info('Loaded YOLO-World concept detector from %s', path)
        return model
    except Exception:
        logger.exception('Failed to load YOLO-World model %s', path)
        return None


def segment_box(image, box_xyxy):
    """
    Return an HxW uint8 mask (0/255) for one xyxy box, or None.
    """
    import numpy as np

    model = _sam_interactive()
    if model is None:
        return None
    array = np.asarray(image.convert('RGB'))
    height, width = array.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    x1 = min(max(x1, 0.0), width - 1)
    y1 = min(max(y1, 0.0), height - 1)
    x2 = min(max(x2, x1 + 1.0), width)
    y2 = min(max(y2, y1 + 1.0), height)
    try:
        results = model.predict(
            array,
            bboxes=[[x1, y1, x2, y2]],
            verbose=False,
            device=_device(),
        )
    except Exception:
        logger.exception('SAM box predict failed for box [%s %s %s %s]', x1, y1, x2, y2)
        return None
    if not results:
        logger.warning('SAM predict returned empty results for box [%s %s %s %s]', x1, y1, x2, y2)
        return None
    if results[0].masks is None or len(results[0].masks.data) == 0:
        logger.warning('SAM predict returned no masks for box [%s %s %s %s] — '
                       'image shape=%s, box area=%dpx',
                       x1, y1, x2, y2, array.shape, int((x2-x1)*(y2-y1)))
        return None
    mask = results[0].masks.data[0]
    if hasattr(mask, 'cpu'):
        mask = mask.cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    if mask.shape[0] != height or mask.shape[1] != width:
        from PIL import Image
        mask = np.array(
            Image.fromarray(mask, mode='L').resize((width, height), Image.Resampling.NEAREST),
        )
    mask_area_px = mask.sum() / 255.0
    if mask_area_px < 50:
        logger.warning('SAM mask is nearly empty (area=%dpx) for box [%s %s %s %s]',
                       int(mask_area_px), x1, y1, x2, y2)
        return None
    logger.info('SAM mask OK for box [%s %s %s %s], area=%dpx', x1, y1, x2, y2, int(mask_area_px))
    return mask


def _replicate_output_bytes(output):
    """Turn a Replicate file / URL / list into PNG bytes."""
    import base64
    import urllib.request

    raw = output
    if hasattr(output, 'read'):
        raw = output.read()
    elif isinstance(output, list) and output:
        raw = output[0]
        if hasattr(raw, 'read'):
            raw = raw.read()
        elif hasattr(raw, 'url'):
            raw = raw.url
    elif hasattr(output, 'url'):
        raw = output.url
    if isinstance(raw, str) and raw.startswith('http'):
        with urllib.request.urlopen(raw, timeout=60) as resp:
            raw = resp.read()
    if isinstance(raw, str) and raw.startswith('data:'):
        raw = base64.b64decode(raw.split(',', 1)[-1])
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    return None


def _padded_xyxy(box_xyxy, width, height, pad_frac):
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(width, x2 + px),
        min(height, y2 + py),
    )


def _image_data_uri(image):
    import base64
    import io

    buf = io.BytesIO()
    image.convert('RGB').save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def _png_to_alpha(raw, size):
    import io
    import numpy as np
    from PIL import Image as _Image

    if not raw:
        return None
    cut = _Image.open(io.BytesIO(raw))
    if cut.mode == 'L':
        alpha = np.array(cut)
    elif cut.mode == 'RGBA':
        alpha = np.array(cut.split()[-1])
    else:
        alpha = np.array(cut.convert('L'))
    if cut.size != size:
        alpha = np.array(
            _Image.fromarray(alpha, mode='L').resize(size, _Image.Resampling.NEAREST)
        )
    return alpha


def _place_crop_mask(alpha, crop_xyxy, box_xyxy, width, height):
    """Paste a crop matte onto the full canvas, then clip to the card box."""
    import numpy as np

    cx1, cy1, cx2, cy2 = crop_xyxy
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    full = np.zeros((height, width), dtype=np.uint8)
    full[cy1:cy2, cx1:cx2] = alpha
    clipped = np.zeros_like(full)
    clipped[y1:y2, x1:x2] = full[y1:y2, x1:x2]
    if (clipped > 16).sum() < 50:
        return None
    return clipped


def _segment_box_grounded_sam(image, box_xyxy, *, prompt=None, negative=None):
    """Text-prompted card/object mask. Never used for the person layer."""
    model = getattr(settings, 'REPLICATE_PLAQUE_MODEL', 'schananas/grounded_sam')
    if not replicate_util.client() or not model:
        return None

    width, height = image.size
    crop_box = _padded_xyxy(box_xyxy, width, height, pad_frac=0.16)
    cx1, cy1, cx2, cy2 = crop_box
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = image.convert('RGB').crop(crop_box)
    payload = {
        'image': _image_data_uri(crop),
        'mask_prompt': prompt or getattr(
            settings, 'REPLICATE_PLAQUE_PROMPT',
            'bonus card, neon plaque, rectangular sign, game card',
        ),
        'negative_mask_prompt': negative or getattr(
            settings, 'REPLICATE_PLAQUE_NEGATIVE',
            'person, hand, finger, face, hair, background',
        ),
        'adjustment_factor': -4,
    }
    output = replicate_util.run(model, payload)
    if output is None:
        return None

    raw = _replicate_output_bytes(output)
    alpha = _png_to_alpha(raw, crop.size)
    if alpha is None:
        return None
    placed = _place_crop_mask(alpha, crop_box, box_xyxy, width, height)
    if placed is None:
        return None
    logger.info('Replicate Grounded SAM mask OK for box %s', [int(v) for v in box_xyxy])
    return placed


def _segment_box_rembg(image, box_xyxy, *, variant=None, pad_frac=0.0):
    """
    Crop the card box and matte it on Replicate (BiRefNet by default).

    cjwbw/rembg treats the dark plaque as background. BiRefNet keeps the
    neon frame; we then fill the interior so the card face stays opaque.
    """
    model = getattr(settings, 'REPLICATE_CUTOUT_MODEL', '')
    if not replicate_util.client() or not model:
        return None

    width, height = image.size
    crop_box = _padded_xyxy(box_xyxy, width, height, pad_frac)
    cx1, cy1, cx2, cy2 = crop_box
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = image.convert('RGB').crop(crop_box)
    payload = {'image': _image_data_uri(crop)}
    if 'birefnet' in model.lower():
        payload.update({
            'variant': variant or getattr(settings, 'REPLICATE_CUTOUT_VARIANT', 'general'),
            'output_format': 'mask',
            'resolution': 0,
            # Do not shrink — mask_offset < 0 eats the neon frame.
            'mask_offset': 0,
        })
    output = replicate_util.run(model, payload)
    if output is None:
        return None

    raw = _replicate_output_bytes(output)
    alpha = _png_to_alpha(raw, crop.size)
    if alpha is None:
        return None
    placed = _place_crop_mask(alpha, crop_box, box_xyxy, width, height)
    if placed is None:
        return None
    logger.info('Replicate %s mask OK for box %s', model, [int(v) for v in box_xyxy])
    return placed


def _detections_from_boxes(boxes, names, source, width, height):
    detections = []
    if boxes is None:
        return detections
    xyxy = boxes.xyxy
    confs = boxes.conf
    clss = boxes.cls
    if hasattr(xyxy, 'cpu'):
        xyxy = xyxy.cpu().numpy()
        confs = confs.cpu().numpy()
        clss = clss.cpu().numpy()
    for index, coords in enumerate(xyxy):
        x1, y1, x2, y2 = (float(value) for value in coords)
        cls_id = int(clss[index])
        label = str(names.get(cls_id, cls_id) if isinstance(names, dict) else names[cls_id])
        if label.lower() in PERSON_CONCEPTS:
            continue
        detections.append(Detection(
            label=label,
            confidence=float(confs[index]),
            source=source,
            x=x1 / width,
            y=y1 / height,
            width=(x2 - x1) / width,
            height=(y2 - y1) / height,
        ))
    return detections


def detect_props(image):
    """
    Find foreground props (dragon, gift, coins, pill) that sit on the character.

    Prefers SAM 3 text prompts; falls back to YOLO-World open vocabulary.
    """
    import numpy as np

    concepts = list(getattr(settings, 'SAM_PROP_CONCEPTS', ()))
    if not concepts:
        return []
    width, height = image.size
    array = np.asarray(image.convert('RGB'))
    min_conf = getattr(settings, 'SAM_MIN_CONFIDENCE', 0.30)

    predictor = _sam3_semantic()
    if predictor is not None:
        try:
            results = predictor(array, text=concepts, verbose=False)
            if results:
                result = results[0]
                names = result.names if isinstance(result.names, dict) else {
                    i: name for i, name in enumerate(result.names)
                }
                return _detections_from_boxes(
                    result.boxes, names, SOURCE_PROP, width, height,
                )
        except Exception:
            logger.exception('SAM 3 concept detection failed; trying YOLO-World.')

    world = _world_detector()
    if world is None:
        return []
    try:
        world.set_classes(concepts)
        results = world.predict(
            array,
            verbose=False,
            conf=min(min_conf, 0.15),
            iou=0.45,
            device=_device(),
        )
    except Exception:
        logger.exception('YOLO-World concept detection failed.')
        return []
    detections = []
    for result in results:
        names = result.names if isinstance(result.names, dict) else {
            i: name for i, name in enumerate(result.names)
        }
        detections.extend(_detections_from_boxes(
            result.boxes, names, SOURCE_PROP, width, height,
        ))
    return detections


def crop_mask(full_mask, box):
    """Slice a full-image mask down to a pixel box (left, top, width, height)."""
    from PIL import Image
    import numpy as np

    left, top, width, height = box
    if full_mask is None:
        return None
    if isinstance(full_mask, Image.Image):
        full_mask = np.array(full_mask)
    h, w = full_mask.shape[:2]
    left = min(max(int(left), 0), w - 1)
    top = min(max(int(top), 0), h - 1)
    width = min(int(width), w - left)
    height = min(int(height), h - top)
    return Image.fromarray(full_mask[top:top + height, left:left + width], mode='L')


# ---------------------------------------------------------------------------
# Public: per-character segmentation for the animation pipeline
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field as _field   # noqa: E402
from typing import List as _List                      # noqa: E402


@dataclass
class CharacterLayer:
    """One segmented character, ready for Remotion compositing."""
    mask_png_path: str          # absolute path to RGBA PNG (transparent bg)
    bbox_norm: dict             # {x, y, width, height} normalised 0-1 in full image
    character_index: int
    effects: _List[str]         # per-character effects chosen by the user
    source_region: dict         # original region dict
    hand_boxes: _List[tuple] = _field(default_factory=list)


_PERSON_SOURCES = {'yolo', 'sam'}


def _is_person_region(region: dict) -> bool:
    source = (region.get('source') or '').lower()
    label = (region.get('label') or '').lower()
    return source in _PERSON_SOURCES or 'person' in label


def _pixel_box_padded(region: dict, img_w: int, img_h: int, pad: float = 0.04):
    """Normalised region → integer pixel coords with padding."""
    x = region['x'] * img_w
    y = region['y'] * img_h
    w = region['width'] * img_w
    h = region['height'] * img_h
    px, py = w * pad, h * pad
    x1 = max(0, int(x - px))
    y1 = max(0, int(y - py))
    x2 = min(img_w, int(x + w + px))
    y2 = min(img_h, int(y + h + py))
    return x1, y1, x2, y2


def _norm_box(x1, y1, x2, y2, img_w, img_h) -> dict:
    return {
        'source': 'sam',
        'label': 'person',
        'x': x1 / img_w,
        'y': y1 / img_h,
        'width': (x2 - x1) / img_w,
        'height': (y2 - y1) / img_h,
    }


def _region_box(region):
    return (
        float(region.get('x') or 0),
        float(region.get('y') or 0),
        float(region.get('width') or 0),
        float(region.get('height') or 0),
    )


def _region_center_in(inner, outer):
    ix, iy, iw, ih = _region_box(inner)
    ox, oy, ow, oh = _region_box(outer)
    cx, cy = ix + iw / 2.0, iy + ih / 2.0
    return ox <= cx <= ox + ow and oy <= cy <= oy + oh


def _region_fraction_inside(inner, outer):
    ix, iy, iw, ih = _region_box(inner)
    ox, oy, ow, oh = _region_box(outer)
    area = iw * ih
    if area <= 0:
        return 0.0
    left = max(ix, ox)
    top = max(iy, oy)
    right = min(ix + iw, ox + ow)
    bottom = min(iy + ih, oy + oh)
    if right <= left or bottom <= top:
        return 0.0
    return ((right - left) * (bottom - top)) / area


def _inflate_region(region, pad=0.12):
    x, y, w, h = _region_box(region)
    return {
        'x': x - w * pad,
        'y': y - h * pad,
        'width': w * (1 + 2 * pad),
        'height': h * (1 + 2 * pad),
    }


def _region_hangs_under(inner, outer):
    ix, iy, iw, ih = _region_box(inner)
    ox, oy, ow, oh = _region_box(outer)
    icx = ix + iw / 2.0
    if icx < ox or icx > ox + ow:
        return False
    outer_bottom = oy + oh
    return iy <= outer_bottom + 0.09 and (iy + ih) >= outer_bottom - 0.05


def _region_hangs_off_plaque(inner, outer):
    """Gift/coins that sit on the card and stick out the right or bottom — not the left (toward the person)."""
    if _region_hangs_under(inner, outer):
        return True
    if _region_fraction_inside(inner, outer) < 0.12:
        return False
    ix, iy, iw, ih = _region_box(inner)
    ox, oy, ow, oh = _region_box(outer)
    icy = iy + ih / 2.0
    icx = ix + iw / 2.0
    if icx < ox:
        return False
    on_row = oy - 0.04 <= icy <= oy + oh + 0.04
    sticks_right = (ix + iw) > ox + ow - 0.02 and ix < ox + ow
    return on_row and sticks_right


def _looks_like_prop(region):
    source = (region.get('source') or '').lower()
    label = (region.get('label') or '').lower()
    if source == 'prop':
        return True
    return any(
        token in label
        for token in (
            'prop', 'pill', 'coin', 'dragon', 'gift', 'star', 'badge', 'chest',
            'chicken', 'food', 'drumstick', 'bucket', 'wing',
        )
    )


def _is_small_handheld(region):
    area = float(region.get('width') or 0) * float(region.get('height') or 0)
    return 0.0 < area < 0.12


def _boxes_overlap_px(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1)


def overlaps_character(region, person_regions, pad=0.28):
    """True if this leftover box sits on a person (raised hand, chicken, coins)."""
    src = (region.get('source') or '').lower()
    if src in _PLAQUE_SOURCES or src == 'ocr':
        return False
    for person in person_regions or []:
        outer = _inflate_region(person, pad=pad)
        if _region_center_in(region, outer) or _region_fraction_inside(region, outer) >= 0.18:
            return True
    return False


def _region_hits_person_mask(region, person_mask, img_w, img_h, thresh=0.10):
    import numpy as np

    if person_mask is None:
        return False
    arr = np.array(person_mask)
    if arr.max() == 0:
        return False
    x1, y1, x2, y2 = _pixel_box_padded(region, img_w, img_h, pad=0.04)
    patch = arr[y1:y2, x1:x2]
    if patch.size == 0:
        return False
    return float((patch > 16).mean()) >= thresh


def _is_card_art_prop(region, ui_regions):
    if not _looks_like_prop(region):
        return False
    for card in ui_regions:
        padded = _inflate_region(card)
        if (
            _region_center_in(region, padded)
            or _region_fraction_inside(region, padded) >= 0.35
            or _region_hangs_under(region, card)
        ):
            return True
    return False


def _covered_by_cutouts(region, layers):
    """True if this leftover box sits on a card that already has a PNG layer."""
    for layer in layers:
        box = getattr(layer, 'bbox_norm', None)
        if not box:
            continue
        if _region_center_in(region, box) or _region_fraction_inside(region, box) >= 0.30:
            return True
    return False


def segment_characters(image, regions: list, tmp_dir: str):
    import os
    import numpy as _np
    import cv2 as _cv2
    from PIL import Image as _Image

    img_rgb = image.convert('RGB')
    img_w, img_h = img_rgb.size
    layers: _List[CharacterLayer] = []
    
    bg_mask = _np.zeros((img_h, img_w), dtype=_np.uint8)

    # 1. Separate persons and props
    person_regions = [r for r in regions if _is_person_region(r)]
    other_regions = [r for r in regions if not _is_person_region(r)]
    logger.info('segment_characters: %d person regions, %d other regions',
                len(person_regions), len(other_regions))
    for r in regions:
        logger.info('  region source=%r label=%r is_person=%s',
                    r.get('source'), r.get('label'), _is_person_region(r))
    
    # 2. Assign each prop to the person whose box overlaps it the MOST.
    #    Using best-overlap (not first-hit) prevents a prop on the right
    #    from being stolen by the middle person that barely touches it.
    consumed_prop_ids = set()
    person_to_props = {i: [] for i in range(len(person_regions))}

    # Pre-compute pixel boxes for all persons
    person_boxes = [_pixel_box_padded(p, img_w, img_h) for p in person_regions]
    ui_regions = [
        r for r in other_regions
        if (r.get('source') or '').lower() in _PLAQUE_SOURCES
    ]

    for j, o_reg in enumerate(other_regions):
        src = (o_reg.get('source') or '').lower()
        if src in _PLAQUE_SOURCES or src == 'ocr':
            continue
        tagged = src == 'prop'
        ox1, oy1, ox2, oy2 = _pixel_box_padded(o_reg, img_w, img_h)
        on_plaque = _is_card_art_prop(o_reg, ui_regions) or _sits_on_plaque(o_reg, ui_regions)
        # Card gifts/coins/pills stay on the plaque. Do not attach them to a
        # person across the poster — that is what made 4 STREAK props look wrong.
        if on_plaque:
            overlaps_person = any(
                _boxes_overlap_px((ox1, oy1, ox2, oy2), person_box)
                for person_box in person_boxes
            )
            if not overlaps_person:
                continue

        best_person = -1
        best_area = 0
        for i, (px1, py1, px2, py2) in enumerate(person_boxes):
            ix1 = max(px1, ox1)
            iy1 = max(py1, oy1)
            ix2 = min(px2, ox2)
            iy2 = min(py2, oy2)
            if ix2 > ix1 and iy2 > iy1:
                i_area = (ix2 - ix1) * (iy2 - iy1)
                if i_area > best_area:
                    best_area = i_area
                    best_person = i

        if best_person < 0 and person_boxes:
            ocx = (ox1 + ox2) / 2.0
            ocy = (oy1 + oy2) / 2.0
            for i, (px1, py1, px2, py2) in enumerate(person_boxes):
                if px1 <= ocx <= px2 and py1 <= ocy <= py2:
                    best_person = i
                    break

        should_attach = tagged or _looks_like_prop(o_reg)
        if not should_attach and best_person >= 0 and _is_small_handheld(o_reg):
            px1, py1, px2, py2 = person_boxes[best_person]
            pad_x = int((px2 - px1) * 0.22)
            pad_y = int((py2 - py1) * 0.22)
            near = (
                px1 - pad_x,
                max(0, py1 - pad_y),
                px2 + pad_x,
                py2 + pad_y,
            )
            should_attach = _boxes_overlap_px((ox1, oy1, ox2, oy2), near)

        if should_attach and best_person >= 0 and _near_plaque_scene(o_reg, ui_regions):
            if not _region_center_in(o_reg, person_regions[best_person]):
                should_attach = False

        if should_attach and best_person >= 0:
            logger.info(
                '  Attaching handheld source=%r label=%r to person %d',
                o_reg.get('source'), o_reg.get('label'), best_person,
            )
            person_to_props[best_person].append(o_reg)
            consumed_prop_ids.add(id(o_reg))

    unconsumed_regions = [r for r in other_regions if id(r) not in consumed_prop_ids]

    char_idx = 0
    for i, region in enumerate(person_regions):
        x1, y1, x2, y2 = _pixel_box_padded(region, img_w, img_h)
        effects = list(region.get('effects') or [])
        assigned_props = person_to_props[i]

        # --- try SAM mask on the person box only (do not enlarge it with PROP) ---
        mask_np = segment_box(img_rgb, [x1, y1, x2, y2])

        # PROP boxes are recorded so inpaint cannot Telea the chicken.
        # Re-SAM only when the person silhouette missed the handheld;
        # a second SAM on fried chicken punches holes and looks muddy.
        extra_px = [
            _pixel_box_padded(prop, img_w, img_h, pad=0.02)
            for prop in assigned_props
        ]
        if mask_np is not None and assigned_props:
            for prop in assigned_props:
                px1, py1, px2, py2 = _pixel_box_padded(prop, img_w, img_h, pad=0.02)
                if _mask_fill_ratio(mask_np, px1, py1, px2, py2) >= 0.45:
                    logger.info(
                        '  PROP already in person SAM [%d %d %d %d], keep original pixels',
                        px1, py1, px2, py2,
                    )
                    continue
                prop_mask = segment_box(img_rgb, [px1, py1, px2, py2])
                if prop_mask is None:
                    # SAM missed the prop — stamp the bbox directly so the prop
                    # moves with the character rather than staying on the static poster.
                    logger.info(
                        '  SAM miss for prop [%d %d %d %d]; stamping bbox into character mask',
                        px1, py1, px2, py2,
                    )
                    mask_np = _stamp_boxes(mask_np, [(px1, py1, px2, py2)])
                    continue
                mask_np = _np.maximum(mask_np, prop_mask)
        elif mask_np is None and assigned_props:
            # Person SAM itself failed. Stamp all prop bboxes into a fresh mask
            # so they are included in the rect-fallback character layer.
            logger.info('  Person SAM failed; stamping %d prop bbox(es) into blank mask', len(assigned_props))
            blank = _np.zeros((img_h, img_w), dtype=_np.uint8)
            prop_boxes = [_pixel_box_padded(p, img_w, img_h, pad=0.02) for p in assigned_props]
            mask_np = _stamp_boxes(blank, prop_boxes)


        if mask_np is not None:
            mask_np = _erase_scene_props_from_person(
                mask_np, img_rgb, other_regions, ui_regions, region, img_w, img_h,
            )
            # Blown-out fingertips often miss SAM. A short dilate keeps the
            # hands on the person layer so cards composite behind the fingers.
            kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (5, 5))
            mask_np = _cv2.dilate(mask_np, kernel, iterations=2)
            mask_np = _erase_scene_props_from_person(
                mask_np, img_rgb, other_regions, ui_regions, region, img_w, img_h,
            )
            mask_clean = (mask_np > 127).astype(_np.uint8) * 255

            rgba = img_rgb.convert('RGBA')
            r, g, b, a = rgba.split()
            alpha_ch = _Image.fromarray(mask_clean, mode='L')
            rgba_masked = _Image.merge('RGBA', (r, g, b, alpha_ch))
            
            # Crop exactly to the non-transparent pixels (this includes the feathered bleed)
            # rather than the user's original box which might artificially cut off the blur.
            actual_bbox = alpha_ch.getbbox()
            if actual_bbox:
                crop = rgba_masked.crop(actual_bbox)
                cx1, cy1, cx2, cy2 = actual_bbox
            else:
                crop = rgba_masked.crop((x1, y1, x2, y2))
                cx1, cy1, cx2, cy2 = x1, y1, x2, y2

            mh, mw = mask_clean.shape[:2]
            if mh == img_h and mw == img_w:
                bg_mask = _np.maximum(bg_mask, mask_clean)
            
            logger.info('Character %d: SAM mask crop (%dx%d) with %d props', char_idx, cx2 - cx1, cy2 - cy1, len(assigned_props))
        else:
            # Rect fallback with soft edges so it's not a sharp box
            crop = img_rgb.convert('RGBA').crop((x1, y1, x2, y2))
            cx1, cy1, cx2, cy2 = x1, y1, x2, y2
            
            # Create a rounded feathered mask for the fallback crop
            w, h = crop.size
            fallback_mask = _Image.new('L', (w, h), 0)
            from PIL import ImageDraw, ImageFilter
            draw = ImageDraw.Draw(fallback_mask)
            # Draw a rounded rectangle mask with feathering
            draw.rounded_rectangle((10, 10, w - 10, h - 10), radius=20, fill=255)
            fallback_mask = fallback_mask.filter(ImageFilter.GaussianBlur(10))
            
            # Apply it to crop's alpha channel
            r, g, b, a = crop.split()
            crop = _Image.merge('RGBA', (r, g, b, fallback_mask))
            
            bg_mask[y1:y2, x1:x2] = 255
            logger.info('Character %d: rect RGBA crop (%dx%d)', char_idx, x2 - x1, y2 - y1)

        mask_path = os.path.join(tmp_dir, f'char_{char_idx}.png')
        crop.save(mask_path, format='PNG')

        layers.append(CharacterLayer(
            mask_png_path=mask_path,
            bbox_norm=_norm_box(cx1, cy1, cx2, cy2, img_w, img_h),
            character_index=char_idx,
            effects=effects,
            source_region=region,
            hand_boxes=list(extra_px),
        ))
        char_idx += 1

    return layers, _Image.fromarray(bg_mask, mode='L'), unconsumed_regions


_PLAQUE_SOURCES = {'card', 'button', 'title'}
_UI_CUTOUT_SOURCES = {'card', 'button', 'title', 'manual', 'prop'}
_PIXEL_MOTION = {
    'float', 'float-glow', 'breathe', 'natural-breathe', 'zoom', 'zoom-in', 'pulse',
    'bounce', 'shake', 'wave', 'spin', 'slide-left', 'slide-up', 'tilt',
}


def _wants_pixel_motion(region: dict) -> bool:
    return any(key in _PIXEL_MOTION for key in (region.get('effects') or []))


def _sits_on_plaque(region, plaques):
    """True if this box is decoration on a bonus card (gift, coins on the plaque).

    Coins between the hand and the card's left edge are scene props, not card art.
    """
    for card in plaques:
        padded = _inflate_region(card, pad=0.06)
        if _region_center_in(region, padded) or _region_fraction_inside(region, card) >= 0.40:
            rx, _ry, rw, _rh = _region_box(region)
            cx, _cy, _cw, _ch = _region_box(card)
            if rx + rw / 2.0 < cx - 0.005:
                continue
            return True
        if _region_hangs_off_plaque(region, card):
            return True
    return False


def _near_plaque_scene(region, plaques):
    """Coins/gifts sitting between a hand and a bonus card, or on the card."""
    if _sits_on_plaque(region, plaques):
        return True
    for card in plaques:
        if _region_fraction_inside(region, _inflate_region(card, pad=0.22)) >= 0.08:
            return True
    return False


def _erase_scene_props_from_person(
    mask_np, img_rgb, other_regions, plaques, person_region, img_w, img_h,
):
    """Keep gold coins intact: do not let the hand SAM swallow them."""
    import numpy as np

    if mask_np is None:
        return mask_np
    out = mask_np
    for region in other_regions or []:
        source = (region.get('source') or '').lower()
        if source in _PLAQUE_SOURCES:
            continue
        if not (_looks_like_prop(region) or source == 'prop'):
            continue
        if not _near_plaque_scene(region, plaques):
            continue
        if _region_center_in(region, person_region) and not _sits_on_plaque(region, plaques):
            continue
        box = _pixel_box_padded(region, img_w, img_h, pad=0.02)
        piece = segment_box(img_rgb, box)
        if piece is None:
            piece = _plaque_from_frame(img_rgb, box)
        if piece is None:
            continue
        out = np.where(piece > 127, 0, out)
        logger.info(
            '  Trimmed scene %s off person so coins stay whole',
            region.get('label') or source,
        )
    return out


def _wants_plaque_mask(region: dict) -> bool:
    """Large selected cards must move as one plaque, not as the gift inside them."""
    source = (region.get('source') or '').lower()
    if source in _PLAQUE_SOURCES:
        return True
    if source == 'manual':
        return float(region.get('width') or 0) * float(region.get('height') or 0) >= 0.03
    return False


def _mask_fill_ratio(mask, x1, y1, x2, y2):
    import numpy as np
    patch = mask[y1:y2, x1:x2]
    if patch.size == 0:
        return 1.0
    return float((patch > 127).mean())


def _fill_mask_holes(mask):
    """Keep gold coins on a gold card: they often punch holes in the SAM blob."""
    import cv2
    import numpy as np

    binary = (mask > 16).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _stamp_boxes(mask, boxes):
    import numpy as np

    out = mask.copy()
    height, width = out.shape[:2]
    for box in boxes or []:
        x1, y1, x2, y2 = [int(value) for value in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = 255
    return out


def _seal_plaque_interior(mask, box):
    """Fill the plaque so coins/gifts on a dark card stay opaque and sharp."""
    import cv2
    import numpy as np

    filled = _fill_mask_holes(mask)
    x1, y1, x2, y2 = [int(value) for value in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(filled.shape[1], x2), min(filled.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return filled
    patch = filled[y1:y2, x1:x2]
    points = cv2.findNonZero(patch)
    if points is None or len(points) < 20:
        return filled
    hull = cv2.convexHull(points)
    hull_mask = np.zeros_like(patch)
    cv2.fillConvexPoly(hull_mask, hull, 255)
    filled[y1:y2, x1:x2] = np.maximum(patch, hull_mask)
    return filled


def _or_masks(*masks):
    import numpy as np

    result = None
    for mask in masks:
        if mask is None:
            continue
        result = mask if result is None else np.maximum(result, mask)
    return result


def _mask_is_selection_rectangle(mask, box):
    """True when SAM/rembg filled the prompt box, not the rounded card."""
    if mask is None:
        return True
    x1, y1, x2, y2 = [int(value) for value in box]
    height, width = mask.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return False
    fill = _mask_fill_ratio(mask, x1, y1, x2, y2)
    if fill < 0.88:
        return False
    span = max(3, min(8, (x2 - x1) // 12, (y2 - y1) // 12))
    corners = (
        mask[y1:y1 + span, x1:x1 + span],
        mask[y1:y1 + span, x2 - span:x2],
        mask[y2 - span:y2, x1:x1 + span],
        mask[y2 - span:y2, x2 - span:x2],
    )
    filled_corners = sum(
        1 for patch in corners
        if patch.size and float((patch > 16).mean()) > 0.45
    )
    return filled_corners >= 3


def _usable_silhouette(mask, box, lo=0.22, hi=0.99):
    if mask is None:
        return False
    if (mask > 16).sum() < 80:
        return False
    if _mask_is_selection_rectangle(mask, box):
        return False
    return lo <= _mask_fill_ratio(mask, *box) <= hi


def _feather_edge_alpha(mask, blur=5):
    """Solid interior, soft neon bloom on the rim. Binary 0/255 looks harsh."""
    import cv2
    import numpy as np

    binary = (mask > 16).astype(np.uint8) * 255
    if binary.max() == 0 or blur < 1:
        return binary
    k = int(blur) * 2 + 1
    soft = cv2.GaussianBlur(binary, (k, k), max(0.8, blur * 0.5))
    core = cv2.erode(
        binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return np.maximum(core, soft)


def _inset_rounded_rect_mask(shape, box, inset=0.045, radius_frac=0.20):
    """Last-resort card shape: rounded plaque inside the box, never a sharp square."""
    import numpy as np
    from PIL import Image, ImageDraw

    height, width = shape[:2]
    x1, y1, x2, y2 = [int(value) for value in box]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    ix = max(3, int(bw * inset))
    iy = max(3, int(bh * inset))
    rx1, ry1 = x1 + ix, y1 + iy
    rx2, ry2 = x2 - ix, y2 - iy
    if rx2 - rx1 < 8 or ry2 - ry1 < 8:
        rx1, ry1, rx2, ry2 = x1, y1, x2, y2
    radius = max(10, int(min(rx2 - rx1, ry2 - ry1) * radius_frac))
    canvas = Image.new('L', (width, height), 0)
    ImageDraw.Draw(canvas).rounded_rectangle(
        (rx1, ry1, rx2 - 1, ry2 - 1), radius=radius, fill=255,
    )
    return np.array(canvas)


def _sw_corner_is_dirty(mask, box):
    """Rounded cards have an empty south-west corner. Fingers/coins fill it."""
    if mask is None:
        return True
    x1, y1, x2, y2 = [int(value) for value in box]
    height, width = mask.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    span_x = max(8, (x2 - x1) // 7)
    span_y = max(8, (y2 - y1) // 5)
    corner = mask[max(0, y2 - span_y):y2, x1:min(width, x1 + span_x)]
    if corner.size == 0:
        return False
    return float((corner > 16).mean()) > 0.22


def _clip_extras_to_card_right(mask, box):
    """Gifts on the right stay. Coins on the left of the card do not."""
    import numpy as np

    if mask is None:
        return None
    x1, y1, x2, y2 = [int(value) for value in box]
    height, width = mask.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    mid = x1 + int((x2 - x1) * 0.42)
    out = mask.copy()
    out[y1:y2, x1:mid] = 0
    if (out > 16).sum() < 40:
        return None
    return out


def _plaque_from_frame(img_rgb, box):
    """
    Trace the neon plaque: only the border ring, then fill what it encloses.

    Interior green type / left vortex used to join the ring, so flood-fill
    leaked and we fell back to SAM (harsh edges). Restrict neon to a donut
    around the box, then flood from outside.
    """
    import cv2
    import numpy as np

    rgb = np.asarray(img_rgb.convert('RGB'))
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad = max(8, int(0.05 * min(bw, bh)))
    ex1, ey1 = max(0, x1 - pad), max(0, y1 - pad)
    ex2, ey2 = min(width, x2 + pad), min(height, y2 + pad)
    crop = rgb[ey1:ey2, ex1:ex2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    gold = (hue <= 28) & (sat > 70) & (val > 80)
    green = (hue >= 35) & (hue <= 100) & (sat > 50) & (val > 45)
    purple = (hue >= 115) & (hue <= 180) & (sat > 28) & (val > 35)
    neon = ((green | purple) & ~gold).astype(np.uint8) * 255
    hot = ((val > 200) & (sat < 100)).astype(np.uint8) * 255
    near = cv2.dilate(neon, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    neon = cv2.bitwise_or(neon, cv2.bitwise_and(hot, near))

    crop_h, crop_w = neon.shape[:2]
    ox1, oy1 = x1 - ex1, y1 - ey1
    ox2, oy2 = x2 - ex1, y2 - ey1
    band = max(12, int(0.14 * min(bw, bh)))
    donut = np.zeros((crop_h, crop_w), dtype=np.uint8)
    donut[oy1:oy2, ox1:ox2] = 255
    iy1, iy2 = oy1 + band, oy2 - band
    ix1, ix2 = ox1 + band, ox2 - band
    if iy2 > iy1 and ix2 > ix1:
        donut[iy1:iy2, ix1:ix2] = 0
    neon = cv2.bitwise_and(neon, donut)
    neon = cv2.dilate(neon, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    neon = cv2.morphologyEx(
        neon, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    )
    if neon.max() == 0:
        logger.info('Neon plaque: no border pixels')
        return None
    inv = cv2.bitwise_not(neon)
    flood = inv.copy()
    ff_mask = np.zeros((crop_h + 2, crop_w + 2), np.uint8)
    for seed in (
        (0, 0), (crop_w - 1, 0), (0, crop_h - 1), (crop_w - 1, crop_h - 1),
        (crop_w // 2, 0), (crop_w // 2, crop_h - 1),
        (0, crop_h // 2), (crop_w - 1, crop_h // 2),
    ):
        sx, sy = seed
        if flood[sy, sx] > 0:
            cv2.floodFill(flood, ff_mask, (sx, sy), 0)
    enclosed = flood
    plaque = cv2.bitwise_or(neon, enclosed)
    # A little extra so the outer bloom is inside the cut-out, not chopped.
    plaque = cv2.dilate(
        plaque, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    full = np.zeros((height, width), dtype=np.uint8)
    full[ey1:ey2, ex1:ex2] = plaque
    clipped = np.zeros_like(full)
    clipped[y1:y2, x1:x2] = full[y1:y2, x1:x2]
    fill = _mask_fill_ratio(clipped, x1, y1, x2, y2)
    inner_fill = _mask_fill_ratio(
        clipped,
        x1 + int(bw * 0.18), y1 + int(bh * 0.18),
        x2 - int(bw * 0.18), y2 - int(bh * 0.18),
    )
    if fill < 0.28 or fill > 0.93 or inner_fill < 0.62:
        logger.info(
            'Neon plaque rejected fill=%.2f inner=%.2f box=%s',
            fill, inner_fill, [x1, y1, x2, y2],
        )
        return None
    full = clipped
    if _mask_is_selection_rectangle(full, box):
        logger.info('Neon plaque rejected: selection rectangle')
        return None
    if _sw_corner_is_dirty(full, box):
        gold_full = _gold_pixels(img_rgb, box)
        span_x = max(8, bw // 7)
        span_y = max(8, bh // 5)
        corner = full[max(0, y2 - span_y):y2, x1:min(width, x1 + span_x)]
        gold_c = gold_full[max(0, y2 - span_y):y2, x1:min(width, x1 + span_x)]
        corner[:] = np.where(gold_c > 0, 0, corner)
        if _sw_corner_is_dirty(full, box):
            logger.info('Neon plaque rejected: dirty SW corner')
            return None
    logger.info('Neon plaque OK fill=%.2f inner=%.2f', fill, inner_fill)
    return full


def _or_extra_silhouettes(img_rgb, extra_boxes, *, allow_boxy=False):
    """Union gift/icon shapes. Never stamp their selection rectangles."""
    combined = None
    for box in extra_boxes or []:
        piece = segment_box(img_rgb, box)
        if piece is not None:
            piece = _fill_mask_holes(piece)
        if piece is None or (not allow_boxy and _mask_is_selection_rectangle(piece, box)):
            piece = _plaque_from_frame(img_rgb, box)
        if piece is None:
            piece = _grabcut_box(img_rgb, box)
            if piece is not None:
                piece = _fill_mask_holes(piece)
        if piece is None:
            continue
        if not allow_boxy and _mask_is_selection_rectangle(piece, box):
            logger.info('Skipping square extra at %s', [int(v) for v in box])
            continue
        combined = piece if combined is None else _or_masks(combined, piece)
    return combined


def _sam_union(img_rgb, boxes):
    import numpy as np

    combined = None
    for box in boxes:
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            continue
        piece = segment_box(img_rgb, box)
        if piece is not None:
            combined = piece if combined is None else np.maximum(combined, piece)
    return combined


def _grabcut_box(image, box_xyxy):
    import cv2
    import numpy as np

    rgb = np.asarray(image.convert('RGB'))
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in box_xyxy]
    x1 = max(0, min(x1, width - 2))
    y1 = max(0, min(y1, height - 2))
    x2 = max(x1 + 2, min(x2, width))
    y2 = max(y1 + 2, min(y2, height))
    rect = (x1, y1, x2 - x1, y2 - y1)
    mask = np.zeros((height, width), dtype=np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rgb, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    result = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    clipped = np.zeros_like(result)
    clipped[y1:y2, x1:x2] = result[y1:y2, x1:x2]
    if (clipped > 16).sum() < 50:
        return None
    return clipped


def _resolve_ui_mask(img_rgb, box, *, plaque, extra_boxes=None):
    """
    Cut the card's own shape (rounded neon/gold plaque), never the
    selection rectangle. Extra gift/icon boxes are unioned as silhouettes,
    not stamped as squares — those stamps are the grid the user saw.
    """
    x1, y1, x2, y2 = box
    if plaque:
        extras = _clip_extras_to_card_right(
            _or_extra_silhouettes(img_rgb, extra_boxes, allow_boxy=True),
            box,
        )

        def _accept_plaque(candidate, *, fill_holes=True):
            if candidate is None:
                return None
            filled = _fill_mask_holes(candidate) if fill_holes else candidate
            if _mask_is_selection_rectangle(filled, box):
                return None
            if (filled > 16).sum() < 80:
                return None
            return filled

        # Neon ring first — BiRefNet returns a blob and loses the notched frame.
        frame = _accept_plaque(_plaque_from_frame(img_rgb, box), fill_holes=False)
        if frame is not None:
            logger.info('Using neon plaque frame')
            return _or_masks(frame, extras)
        rembg = _accept_plaque(
            _segment_box_rembg(img_rgb, box, variant='toonout', pad_frac=0.10)
        )
        if rembg is not None and not _sw_corner_is_dirty(rembg, box):
            logger.info('Using plaque silhouette')
            return _or_masks(rembg, extras)
        # SAM follows the green vortex and squares the notched frame.
        logger.info('Using inset rounded card')
        rounded = _inset_rounded_rect_mask(
            (img_rgb.size[1], img_rgb.size[0]), box,
        )
        return _or_masks(rounded, extras)

    rembg = _segment_box_rembg(img_rgb, box, variant='toonout', pad_frac=0.08)
    candidates = [rembg]
    if rembg is None:
        candidates.append(_segment_box_grounded_sam(
            img_rgb, box,
            prompt='object, item, gift, icon, coin, product',
            negative='person, hand, finger, face, hair, background',
        ))
    candidates.extend([segment_box(img_rgb, box), _grabcut_box(img_rgb, box)])
    for candidate in candidates:
        if candidate is None:
            continue
        filled = _fill_mask_holes(candidate)
        fill = _mask_fill_ratio(filled, x1, y1, x2, y2)
        if fill < 0.08 or fill > 0.93:
            continue
        if _mask_is_selection_rectangle(filled, box):
            continue
        return filled
    return None


def inpaint_masked(image, mask, protect_mask=None):
    """Fill card holes so a scaled cut-out does not ghost.

    Pixels under a person stay original — inpainting that junction is what
    painted the dark smear between arm and gold frame.
    """
    import cv2
    import numpy as np
    from PIL import Image as _Image

    if mask is None:
        return image.convert('RGB')
    if hasattr(mask, 'size'):
        mask = np.array(mask)
    if mask.max() == 0:
        return image.convert('RGB')
    rgb = np.asarray(image.convert('RGB')).copy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    kernel = np.ones((5, 5), np.uint8)
    hole = cv2.dilate(mask, kernel, iterations=1)
    if protect_mask is not None:
        protect = np.array(protect_mask) if hasattr(protect_mask, 'size') else protect_mask
        if protect.shape[:2] == hole.shape[:2] and protect.max() > 0:
            # Wide halo: VIP-card Telea next to the bucket is what muddied
            # the chicken after PROP was applied.
            protect_bin = cv2.dilate(
                (protect > 127).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=2,
            )
            hole = cv2.bitwise_and(hole, cv2.bitwise_not(protect_bin))
    if hole.max() == 0:
        return _Image.fromarray(rgb)
    filled = cv2.inpaint(bgr, hole, 3, cv2.INPAINT_TELEA)
    out = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
    if protect_mask is not None:
        protect = np.array(protect_mask) if hasattr(protect_mask, 'size') else protect_mask
        if protect.shape[:2] == out.shape[:2]:
            keep = protect > 127
            out[keep] = rgb[keep]
    return _Image.fromarray(out)


def protect_person_pixels(person_mask, characters, img_w, img_h):
    """Keep original person / chicken / handheld pixels out of Telea.

    SAM often misses fried-chicken crumbs. Those holes must still be the
    poster, not an inpainted smear from the nearby VIP card.
    """
    import cv2
    import numpy as np
    from PIL import Image as _Image

    protect = np.zeros((img_h, img_w), dtype=np.uint8)
    if person_mask is not None:
        arr = np.array(person_mask)
        if arr.shape[:2] == (img_h, img_w):
            protect = np.maximum(protect, (arr > 127).astype(np.uint8) * 255)
    for layer in characters or []:
        region = getattr(layer, 'source_region', None)
        if region:
            x1, y1, x2, y2 = _pixel_box_padded(region, img_w, img_h, pad=0.06)
            protect[y1:y2, x1:x2] = 255
        for box in getattr(layer, 'hand_boxes', None) or []:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 > x1 and y2 > y1:
                protect[y1:y2, x1:x2] = 255
    if protect.max() == 0:
        return person_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    protect = cv2.dilate(protect, kernel, iterations=2)
    return _Image.fromarray(protect, mode='L')


def _carve_person_from_ui_mask(card_mask, person_mask):
    """
    Remove only pixels that are actually on the person.

    A wide dilated halo left a dark gap between the arm and the gold frame.
    A 1px fringe left card glow on the sleeve, so both layers carried the
    same gold/red bloom. A short halo is enough for the person layer to own
    the junction.
    """
    import cv2
    import numpy as np

    if person_mask is None:
        return card_mask
    if hasattr(person_mask, 'size'):
        person_mask = np.array(person_mask)
    if person_mask.shape[:2] != card_mask.shape[:2] or person_mask.max() == 0:
        return card_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    keep_out = cv2.dilate((person_mask > 16).astype(np.uint8) * 255, kernel, iterations=1)
    carved = card_mask.copy()
    carved[keep_out > 0] = 0
    return carved


def _carve_foreign_props_from_card(
    card_mask, img_rgb, regions, this_card, extras, img_w, img_h,
):
    """Do not let the card cutout slice coins that sit beside it, near the hand."""
    import numpy as np

    if card_mask is None:
        return card_mask
    extra_ids = {id(item) for item in extras or []}
    out = card_mask
    for region in regions or []:
        if region is this_card or id(region) in extra_ids:
            continue
        source = (region.get('source') or '').lower()
        if source in _PLAQUE_SOURCES:
            continue
        if not (_looks_like_prop(region) or source == 'prop'):
            continue
        if _sits_on_plaque(region, [this_card]):
            continue
        box = _pixel_box_padded(region, img_w, img_h, pad=0.02)
        piece = segment_box(img_rgb, box)
        if piece is None:
            piece = _plaque_from_frame(img_rgb, box)
        if piece is None:
            continue
        out = np.where(piece > 127, 0, out)
        logger.info(
            '  Carved neighboring %s out of card so coins stay whole',
            region.get('label') or source,
        )
    return out


def _gold_pixels(img_rgb, box):
    """Gold coin pixels inside a box (HSV)."""
    import cv2
    import numpy as np

    rgb = np.asarray(img_rgb.convert('RGB'))
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    full = np.zeros((height, width), dtype=np.uint8)
    if x2 <= x1 or y2 <= y1:
        return full
    hsv = cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2HSV)
    gold = cv2.inRange(hsv, (8, 70, 90), (42, 255, 255))
    gold = cv2.morphologyEx(
        gold, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    full[y1:y2, x1:x2] = gold
    return full


def _carve_left_clutter_from_card(card_mask, img_rgb, box, person_mask):
    """Remove only gold blobs in the south-west corner — never a vertical strip."""
    import numpy as np

    if card_mask is None:
        return card_mask
    x1, y1, x2, y2 = [int(v) for v in box]
    height, width = card_mask.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return card_mask
    out = card_mask.copy()
    corner_w = max(10, int((x2 - x1) * 0.18))
    corner_h = max(10, int((y2 - y1) * 0.22))
    gold = _gold_pixels(img_rgb, box)
    out[y2 - corner_h:y2, x1:x1 + corner_w] = np.where(
        gold[y2 - corner_h:y2, x1:x1 + corner_w] > 0,
        0,
        out[y2 - corner_h:y2, x1:x1 + corner_w],
    )
    return out


def _bbox_in_front_of_people(bbox, person_bboxes):
    """Table cards (VIP, chips) sit in front of the legs/waist, not behind the arm."""
    if not bbox or not person_bboxes:
        return False
    cy = float(bbox['y']) + float(bbox['height']) / 2.0
    right = float(bbox['x']) + float(bbox['width'])
    bottom = float(bbox['y']) + float(bbox['height'])
    for person in person_bboxes:
        px = float(person['x'])
        py = float(person['y'])
        pr = px + float(person['width'])
        pb = py + float(person['height'])
        overlaps = float(bbox['x']) < pr and right > px and float(bbox['y']) < pb and bottom > py
        if overlaps and cy > py + float(person['height']) * 0.52:
            return True
    return False


def _card_sits_in_front(card_mask, person_mask):
    import numpy as np

    if person_mask is None or card_mask is None:
        return False
    person = np.array(person_mask)
    if person.max() == 0:
        return False
    person_bin = person > 16
    card_bin = card_mask > 16
    overlap = person_bin & card_bin
    if overlap.sum() < 40:
        return False
    ys = np.where(person_bin)[0]
    oys = np.where(overlap)[0]
    y_mid = float(ys.min()) + 0.72 * float(ys.max() - ys.min())
    return float(oys.mean()) > y_mid


def knock_card_glow_off_characters(characters, cutout_mask, img_w, img_h, cutouts=None):
    """Drop leftover card bloom from the character, fully where a table card is in front."""
    import cv2
    import numpy as np
    from PIL import Image as _Image

    if not characters or cutout_mask is None:
        return
    card = np.array(cutout_mask)
    if card.max() == 0:
        return
    card_bin = card > 80
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    card_halo = cv2.dilate(card_bin.astype(np.uint8), kernel, iterations=1) > 0
    front = np.zeros(card_bin.shape, dtype=bool)
    person_boxes = [getattr(layer, 'bbox_norm', None) for layer in characters]
    person_boxes = [box for box in person_boxes if box]
    for layer in cutouts or []:
        box = getattr(layer, 'bbox_norm', None)
        if not box or not _bbox_in_front_of_people(box, person_boxes):
            continue
        x1 = int(round(float(box['x']) * img_w))
        y1 = int(round(float(box['y']) * img_h))
        x2 = int(round((float(box['x']) + float(box['width'])) * img_w))
        y2 = int(round((float(box['y']) + float(box['height'])) * img_h))
        front[max(0, y1):min(card_bin.shape[0], y2), max(0, x1):min(card_bin.shape[1], x2)] = True

    for layer in characters:
        path = getattr(layer, 'mask_png_path', None)
        box = getattr(layer, 'bbox_norm', None)
        if not path or not box:
            continue
        crop = _Image.open(path).convert('RGBA')
        arr = np.array(crop)
        x1 = int(round(float(box['x']) * img_w))
        y1 = int(round(float(box['y']) * img_h))
        x2 = int(round((float(box['x']) + float(box['width'])) * img_w))
        y2 = int(round((float(box['y']) + float(box['height'])) * img_h))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        def _fit(mask):
            patch = mask[y1:y2, x1:x2]
            if patch.shape[0] == arr.shape[0] and patch.shape[1] == arr.shape[1]:
                return patch
            img = _Image.fromarray(patch.astype(np.uint8) * 255, mode='L')
            return np.array(img.resize((arr.shape[1], arr.shape[0]), _Image.Resampling.NEAREST)) > 80

        halo_patch = _fit(card_halo)
        front_patch = _fit(front)
        card_patch = _fit(card_bin)
        alpha = arr[:, :, 3]
        # Opaque chicken / sleeve / hair stay. Only knock leftover card bloom.
        # Zeroing solid pixels over the table VIP is what smeared the bucket.
        knock = (halo_patch | (front_patch & card_patch)) & (alpha < 242)
        arr[:, :, 3] = np.where(knock, 0, alpha)
        _Image.fromarray(arr).save(path)


def segment_ui_cutouts(image, regions: list, tmp_dir: str, person_mask=None):
    """
    Replicate silhouette for cards, buttons, titles, and standalone props.

    Does not change the person layer. person_mask is only used to keep
    overlapping pixels on the character, not to re-cut the person.
    """
    import os
    import numpy as _np
    import cv2 as _cv2
    from PIL import Image as _Image

    img_rgb = image.convert('RGB')
    img_w, img_h = img_rgb.size
    layers: _List[CharacterLayer] = []
    combined = _np.zeros((img_h, img_w), dtype=_np.uint8)
    leftover = []
    cut_idx = 0
    plaque_regions = [
        r for r in regions
        if (r.get('source') or '').lower() in _PLAQUE_SOURCES
    ]
    absorbed_ids = set()
    ordered = []
    extra_boxes_for = {}
    for region in regions:
        source = (region.get('source') or '').lower()
        if source not in _PLAQUE_SOURCES:
            continue
        extras = [
            extra for extra in regions
            if extra is not region
            and (extra.get('source') or '').lower() not in _PLAQUE_SOURCES
            and _sits_on_plaque(extra, [region])
            and not _region_hits_person_mask(extra, person_mask, img_w, img_h, thresh=0.18)
        ]
        if extras:
            logger.info(
                'Merging %d coin/gift box(es) into the %s silhouette',
                len(extras), source,
            )
            absorbed_ids.update(id(extra) for extra in extras)
            extra_boxes_for[id(region)] = extras
            ordered.append(region)
        else:
            ordered.append(region)
    for region in regions:
        if id(region) in absorbed_ids:
            continue
        source = (region.get('source') or '').lower()
        if source in _PLAQUE_SOURCES:
            continue
        ordered.append(region)

    for region in ordered:
        source = (region.get('source') or '').lower()
        if id(region) in absorbed_ids:
            continue
        if (
            _sits_on_plaque(region, plaque_regions)
            and source not in _PLAQUE_SOURCES
        ):
            continue
        if source not in _UI_CUTOUT_SOURCES or not _wants_pixel_motion(region):
            leftover.append(region)
            continue
        if source != 'card' and _region_hits_person_mask(region, person_mask, img_w, img_h):
            leftover.append(region)
            continue

        # pad=0.04 captures a slim fringe around the card edge; the outer
        # neon glow is replicated as a CSS drop-shadow in Remotion so it
        # moves with the card without bleeding into adjacent regions.
        pad = 0.04 if source in _PLAQUE_SOURCES else 0.03
        x1, y1, x2, y2 = _pixel_box_padded(region, img_w, img_h, pad=pad)
        plaque = _wants_plaque_mask(region)
        extra_px = None
        for extra in extra_boxes_for.get(id(region), []):
            extra_px = extra_px or []
            extra_px.append(_pixel_box_padded(extra, img_w, img_h, pad=0.02))
        # Keep the plaque box as the neon frame. Expanding it to gifts first
        # convex-hulls card + coins into a blob (the messy 4 STREAK cutout).
        mask_np = _resolve_ui_mask(
            img_rgb, [x1, y1, x2, y2], plaque=plaque, extra_boxes=extra_px,
        )
        if mask_np is None:
            leftover.append(region)
            continue

        sits_front = plaque and _card_sits_in_front(mask_np, person_mask)
        if plaque:
            mask_np = _carve_person_from_ui_mask(mask_np, person_mask)
            mask_np = _carve_foreign_props_from_card(
                mask_np, img_rgb, regions, region,
                extra_boxes_for.get(id(region), []),
                img_w, img_h,
            )
            mask_np = _carve_left_clutter_from_card(
                mask_np, img_rgb, [x1, y1, x2, y2], person_mask,
            )
        if (mask_np > 16).sum() < 50:
            leftover.append(region)
            continue

        # Binary interior. Do not seal after carving — hulling filled the
        # finger and coin holes back in on SHARE & WIN / 4 STREAK.
        mask_bin = (mask_np > 16).astype(_np.uint8) * 255
        if extra_px and not plaque:
            mask_bin = _stamp_boxes(mask_bin, extra_px)
        if plaque and _mask_is_selection_rectangle(mask_bin, [x1, y1, x2, y2]):
            logger.info('Dropping square card mask; leftover will use rounded crop')
            leftover.append(region)
            continue
        # blur=8 → ~17px feather, enough to soften the card silhouette edge
        # without bleeding far into neighbouring regions.
        mask_alpha = _feather_edge_alpha(mask_bin, blur=8) if plaque else mask_bin
        rgba = img_rgb.convert('RGBA')
        r, g, b, _a = rgba.split()
        alpha_ch = _Image.fromarray(mask_alpha, mode='L')
        rgba_masked = _Image.merge('RGBA', (r, g, b, alpha_ch))
        actual_bbox = alpha_ch.getbbox()
        if not actual_bbox:
            leftover.append(region)
            continue
        crop = rgba_masked.crop(actual_bbox)
        cx1, cy1, cx2, cy2 = actual_bbox
        hole = mask_bin
        if plaque:
            hole = _cv2.dilate(
                mask_bin,
                _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (7, 7)),
            )
        combined = _np.maximum(combined, hole)

        mask_path = os.path.join(tmp_dir, f'ui_{cut_idx}.png')
        crop.save(mask_path, format='PNG')
        src = dict(region)
        if sits_front:
            src['front'] = True
        layers.append(CharacterLayer(
            mask_png_path=mask_path,
            bbox_norm={
                'x': cx1 / img_w,
                'y': cy1 / img_h,
                'width': (cx2 - cx1) / img_w,
                'height': (cy2 - cy1) / img_h,
            },
            character_index=cut_idx,
            effects=list(region.get('effects') or []),
            source_region=src,
        ))
        logger.info('UI cut-out %d (%s) %dx%d', cut_idx, source, cx2 - cx1, cy2 - cy1)
        cut_idx += 1

    leftover = [
        r for r in leftover
        if not _region_hits_person_mask(r, person_mask, img_w, img_h)
        and not _sits_on_plaque(r, plaque_regions)
        and not _is_card_art_prop(r, plaque_regions)
        and not _covered_by_cutouts(r, layers)
    ]

    return layers, _Image.fromarray(combined, mode='L'), leftover

