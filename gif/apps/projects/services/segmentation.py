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


def _segment_box_rembg(image, box_xyxy):
    """
    Replicate rembg on the cropped box. Better than full-image automatic SAM
    for a single ornate card: the crop is small, the marble is treated as
    background, the gold plaque + crown stay in the alpha channel.
    """
    import base64
    import io
    import numpy as np
    from PIL import Image as _Image

    token = getattr(settings, 'REPLICATE_API_TOKEN', '')
    model = getattr(settings, 'REPLICATE_CUTOUT_MODEL', '')
    if not token or not model:
        return None
    try:
        import replicate
    except ImportError:
        return None

    os.environ['REPLICATE_API_TOKEN'] = token
    width, height = image.size
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image.convert('RGB').crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    crop.save(buf, format='PNG')
    data_uri = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
    try:
        output = replicate.run(model, input={'image': data_uri})
    except Exception:
        logger.exception('Replicate rembg cut-out failed')
        return None

    raw = output
    if hasattr(output, 'read'):
        raw = output.read()
    elif isinstance(output, list) and output:
        raw = output[0]
        if hasattr(raw, 'read'):
            raw = raw.read()
        elif isinstance(raw, str) and raw.startswith('http'):
            import urllib.request
            with urllib.request.urlopen(raw, timeout=60) as resp:
                raw = resp.read()
    if isinstance(raw, str) and raw.startswith('data:'):
        raw = base64.b64decode(raw.split(',', 1)[-1])
    if not isinstance(raw, (bytes, bytearray)):
        return None
    cut = _Image.open(io.BytesIO(raw)).convert('RGBA')
    if cut.size != crop.size:
        cut = cut.resize(crop.size, _Image.Resampling.LANCZOS)
    alpha = np.array(cut.split()[-1])
    if (alpha > 16).sum() < 50:
        return None
    full = np.zeros((height, width), dtype=np.uint8)
    full[y1:y2, x1:x2] = alpha
    logger.info('Replicate rembg mask OK for box [%s %s %s %s]', x1, y1, x2, y2)
    return full


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
    if src in _PLAQUE_SOURCES:
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
        if src in _PLAQUE_SOURCES:
            continue
        tagged = src == 'prop'
        # Card decorations stay on the plaque. User-tagged PROP coins in a
        # hand must not be swallowed just because they sit near a bonus card.
        if not tagged and (
            _is_card_art_prop(o_reg, ui_regions) or _sits_on_plaque(o_reg, ui_regions)
        ):
            consumed_prop_ids.add(id(o_reg))
            continue

        ox1, oy1, ox2, oy2 = _pixel_box_padded(o_reg, img_w, img_h)

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
            best_dist = None
            for i, (px1, py1, px2, py2) in enumerate(person_boxes):
                if px1 <= ocx <= px2 and py1 <= ocy <= py2:
                    best_person = i
                    break
                pcx = (px1 + px2) / 2.0
                pcy = (py1 + py2) / 2.0
                dist = (pcx - ocx) ** 2 + (pcy - ocy) ** 2
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_person = i

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
                    logger.info('  Skipping handheld SAM miss at [%d %d %d %d]', px1, py1, px2, py2)
                    continue
                mask_np = _np.maximum(mask_np, prop_mask)

        if mask_np is not None:
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
    """True if this box is decoration on a bonus card (gift, coins, extra draw)."""
    for card in plaques:
        padded = _inflate_region(card)
        if (
            _region_center_in(region, padded)
            or _region_fraction_inside(region, padded) >= 0.30
            or _region_hangs_under(region, card)
        ):
            return True
    return False


def _merge_effects_into(card, extras):
    merged = dict(card)
    effects = list(merged.get('effects') or [])
    for extra in extras:
        for key in extra.get('effects') or []:
            if key not in effects:
                effects.append(key)
    merged['effects'] = effects
    return merged


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


def _usable_silhouette(mask, box, lo=0.28, hi=0.93):
    if mask is None:
        return False
    if (mask > 16).sum() < 80:
        return False
    return lo <= _mask_fill_ratio(mask, *box) <= hi


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
    Cut a silhouette, never a rounded rectangle of the selection.

    Cards: SAM the plaque + the coin/gift strip, fill the interior so gold
    coins stay in the card, then rembg / GrabCut if SAM only caught the gift.
    """
    x1, y1, x2, y2 = box
    if plaque:
        height = y2 - y1
        parts = [box, [x1, y1 + int(height * 0.50), x2, y2]]
        if extra_boxes:
            parts.extend(extra_boxes)
        sam = _sam_union(img_rgb, parts)
        rembg = _segment_box_rembg(img_rgb, box)
        grab = _grabcut_box(img_rgb, box)
        merged = _or_masks(sam, rembg, grab)
        if merged is not None:
            merged = _seal_plaque_interior(merged, box)
            merged = _stamp_boxes(merged, extra_boxes)
        for candidate in (merged, sam, rembg, grab):
            if candidate is None:
                continue
            sealed = _seal_plaque_interior(candidate, box)
            sealed = _stamp_boxes(sealed, extra_boxes)
            if _usable_silhouette(sealed, box):
                logger.info('Using sealed plaque silhouette so card props stay sharp')
                return sealed
        if merged is not None and (merged > 16).sum() > 80:
            return merged
        return None

    sam = segment_box(img_rgb, box)
    if sam is not None:
        sam = _fill_mask_holes(sam)
        fill = _mask_fill_ratio(sam, x1, y1, x2, y2)
        if 0.08 <= fill <= 0.93:
            return sam
        logger.info('SAM fill=%.2f is not a usable prop silhouette', fill)

    rembg = _segment_box_rembg(img_rgb, box)
    if rembg is not None:
        rembg = _fill_mask_holes(rembg)
        fill = _mask_fill_ratio(rembg, x1, y1, x2, y2)
        if 0.08 <= fill <= 0.93:
            return rembg

    grab = _grabcut_box(img_rgb, box)
    if grab is not None:
        logger.info('Using GrabCut for small prop box')
        return grab
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
    keep_out = cv2.dilate((person_mask > 16).astype(np.uint8) * 255, kernel, iterations=2)
    carved = card_mask.copy()
    carved[keep_out > 0] = 0
    return carved


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
    y_mid = float(ys.min()) + 0.55 * float(ys.max() - ys.min())
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
    SAM (or Replicate rembg) silhouette for ornate cards / buttons / titles.

    Person SAM is not modified. If a person_mask is given, arm overlap is
    carved out of the card so it does not zoom with the plaque.
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
            and (extra.get('source') or '').lower() != 'prop'
            and _sits_on_plaque(extra, [region])
        ]
        if extras:
            logger.info(
                'Merging %d coin/gift box(es) into the %s silhouette',
                len(extras), source,
            )
            absorbed_ids.update(id(extra) for extra in extras)
            merged = _merge_effects_into(region, extras)
            extra_boxes_for[id(merged)] = extras
            ordered.append(merged)
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
            and source != 'prop'
        ):
            continue
        if source not in _UI_CUTOUT_SOURCES or not _wants_pixel_motion(region):
            leftover.append(region)
            continue
        if source != 'card' and _region_hits_person_mask(region, person_mask, img_w, img_h):
            leftover.append(region)
            continue

        pad = 0.02 if source in _PLAQUE_SOURCES else 0.03
        x1, y1, x2, y2 = _pixel_box_padded(region, img_w, img_h, pad=pad)
        plaque = _wants_plaque_mask(region)
        extra_px = None
        for extra in extra_boxes_for.get(id(region), []):
            extra_px = extra_px or []
            extra_px.append(_pixel_box_padded(extra, img_w, img_h, pad=0.02))
        mask_np = _resolve_ui_mask(
            img_rgb, [x1, y1, x2, y2], plaque=plaque, extra_boxes=extra_px,
        )
        if mask_np is None:
            leftover.append(region)
            continue

        sits_front = plaque and _card_sits_in_front(mask_np, person_mask)
        if plaque and not sits_front:
            mask_np = _carve_person_from_ui_mask(mask_np, person_mask)
        if (mask_np > 16).sum() < 50:
            leftover.append(region)
            continue

        # Binary interior. Eroding / blurring let inpaint show through
        # coins and gifts, which is what made card props look melted.
        mask_bin = (mask_np > 16).astype(_np.uint8) * 255
        if extra_px:
            mask_bin = _stamp_boxes(mask_bin, extra_px)
        if plaque:
            mask_bin = _seal_plaque_interior(mask_bin, [x1, y1, x2, y2])
        mask_clean = mask_bin
        rgba = img_rgb.convert('RGBA')
        r, g, b, _a = rgba.split()
        alpha_ch = _Image.fromarray(mask_clean, mode='L')
        rgba_masked = _Image.merge('RGBA', (r, g, b, alpha_ch))
        actual_bbox = alpha_ch.getbbox()
        if not actual_bbox:
            leftover.append(region)
            continue
        crop = rgba_masked.crop(actual_bbox)
        cx1, cy1, cx2, cy2 = actual_bbox
        combined = _np.maximum(combined, mask_clean)

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
        and (
            (r.get('source') or '').lower() == 'prop'
            or (
                not _sits_on_plaque(r, plaque_regions)
                and not _is_card_art_prop(r, plaque_regions)
                and not _covered_by_cutouts(r, layers)
            )
        )
    ]

    return layers, _Image.fromarray(combined, mode='L'), leftover

