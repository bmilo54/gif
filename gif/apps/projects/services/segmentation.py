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
        logger.exception('SAM box predict failed')
        return None
    if not results or results[0].masks is None:
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
    if mask.mean() < 4:
        return None
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
