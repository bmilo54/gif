"""
Person / prop cut-outs for layered animation.

SAM 3 is used for open-vocabulary concepts when the weight is present.
On CPU we fall back to YOLO-World for those boxes, then SAM 2.1 for the
silhouette. GrabCut remains the last resort so GIF generation still runs
if weights are missing.
"""

from functools import lru_cache
import logging

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

    for j, o_reg in enumerate(other_regions):
        is_explicit_prop = o_reg.get('source') == 'prop'
        is_auto_prop = o_reg.get('label', '').lower() in (
            'prop', 'pill', 'coin', 'dragon', 'gift', 'star', 'badge', 'chest'
        )
        if not is_explicit_prop and not is_auto_prop:
            continue

        ox1, oy1, ox2, oy2 = _pixel_box_padded(o_reg, img_w, img_h)

        best_person = -1
        best_area = 0
        for i, (px1, py1, px2, py2) in enumerate(person_boxes):
            ix1 = max(px1, ox1); iy1 = max(py1, oy1)
            ix2 = min(px2, ox2); iy2 = min(py2, oy2)
            if ix2 > ix1 and iy2 > iy1:
                i_area = (ix2 - ix1) * (iy2 - iy1)
                if i_area > best_area:
                    best_area = i_area
                    best_person = i

        if best_person >= 0:
            logger.info('  Assigning prop source=%r label=%r to person %d (overlap=%dpx²)',
                        o_reg.get('source'), o_reg.get('label'), best_person, best_area)
            person_to_props[best_person].append(o_reg)
            consumed_prop_ids.add(id(o_reg))

    unconsumed_regions = [r for r in other_regions if id(r) not in consumed_prop_ids]

    char_idx = 0
    for i, region in enumerate(person_regions):
        x1, y1, x2, y2 = _pixel_box_padded(region, img_w, img_h)
        effects = list(region.get('effects') or [])

        # --- try SAM mask ---
        mask_np = segment_box(img_rgb, [x1, y1, x2, y2])
        
        # Merge assigned props
        assigned_props = person_to_props[i]
        if mask_np is not None and assigned_props:
            for prop in assigned_props:
                px1, py1, px2, py2 = _pixel_box_padded(prop, img_w, img_h)
                prop_mask = segment_box(img_rgb, [px1, py1, px2, py2])
                if prop_mask is None:
                    # SAM failed for small prop (e.g., coin).
                    # Fallback: draw a soft ellipse (better than a rect to avoid sharp background tearing)
                    logger.info('  Prop SAM failed for prop at [%d %d %d %d] — using soft ellipse fallback', px1, py1, px2, py2)
                    prop_mask = _np.zeros((img_h, img_w), dtype=_np.uint8)
                    center = ((px1 + px2) // 2, (py1 + py2) // 2)
                    axes = ((px2 - px1) // 2, (py2 - py1) // 2)
                    _cv2.ellipse(prop_mask, center, axes, 0, 0, 360, 255, -1)
                    prop_mask = _cv2.GaussianBlur(prop_mask, (11, 11), 0)

                # Merge prop mask into character mask and expand bounding box
                mask_np = _np.maximum(mask_np, prop_mask)
                x1 = min(x1, px1)
                y1 = min(y1, py1)
                x2 = max(x2, px2)
                y2 = max(y2, py2)

        if mask_np is not None:
            # --- Step 1: Minimal edge erosion (just removes 1-2px SAM fringe) ---
            erode_kernel = _np.ones((2, 2), _np.uint8)
            mask_clean = _cv2.erode(mask_np, erode_kernel, iterations=1)

            # --- Step 2: Minimal feathering only (7px) ---
            # Heavy feathering creates semi-transparent edges that reveal the background
            # person underneath, causing a "ghost / soul-leaving-body" double image.
            mask_feathered = _cv2.GaussianBlur(mask_clean, (7, 7), 0)

            # Apply feathered mask to full image then crop
            rgba = img_rgb.convert('RGBA')
            r, g, b, a = rgba.split()
            alpha_ch = _Image.fromarray(mask_feathered, mode='L')
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

            mh, mw = mask_np.shape[:2]
            if mh == img_h and mw == img_w:
                bg_mask = _np.maximum(bg_mask, mask_np)
            
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
        ))
        char_idx += 1

    return layers, _Image.fromarray(bg_mask, mode='L'), unconsumed_regions
