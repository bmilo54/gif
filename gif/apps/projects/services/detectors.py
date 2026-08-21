"""
Detection backends.

A detector is any callable taking a preprocessed PIL image and returning a
list of `Detection`. Swapping engines is a settings change, so the rest of the
pipeline never imports ultralytics or paddleocr directly. Those libraries are
imported lazily inside each backend because they are multi-gigabyte optional
dependencies that the web tier does not need to load.
"""

from dataclasses import dataclass
from functools import lru_cache

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from ..choices import SOURCE_OCR, SOURCE_YOLO


@dataclass(frozen=True)
class Detection:
    """One detected box, with coordinates normalised to 0-1."""

    label: str
    confidence: float
    source: str
    x: float
    y: float
    width: float
    height: float
    text_content: str = ''

    def clamped(self):
        """Clip the box to the image, since engines can overshoot the edges."""
        x = min(max(self.x, 0.0), 1.0)
        y = min(max(self.y, 0.0), 1.0)
        return Detection(
            label=self.label,
            confidence=self.confidence,
            source=self.source,
            x=x,
            y=y,
            width=min(max(self.width, 0.0), 1.0 - x),
            height=min(max(self.height, 0.0), 1.0 - y),
            text_content=self.text_content,
        )


class StubObjectDetector:
    """Fixed boxes so the pipeline and UI can be exercised without YOLO."""

    def __call__(self, image):
        return [
            Detection('person', 0.94, SOURCE_YOLO, 0.08, 0.12, 0.34, 0.62),
            Detection('car', 0.71, SOURCE_YOLO, 0.55, 0.45, 0.36, 0.30),
        ]


class StubTextDetector:
    """Fixed text box so the pipeline and UI can be exercised without OCR."""

    def __call__(self, image):
        return [
            Detection('SALE', 0.88, SOURCE_OCR, 0.30, 0.80, 0.25, 0.10, text_content='SALE'),
        ]


class YoloObjectDetector:
    def __init__(self, model_path=None):
        self.model_path = model_path or settings.YOLO_MODEL
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImproperlyConfigured(
                    "The 'yolo' detection backend requires ultralytics. "
                    "Install it or set DETECTION_OBJECT_BACKEND = 'stub'."
                ) from exc
            self._model = YOLO(self.model_path)
        return self._model

    def __call__(self, image):
        model = self._load()
        width, height = image.size
        detections = []

        # Pass the configured confidence floor directly into YOLO so its own
        # NMS stage uses the same threshold as the pipeline's merge step.
        conf_threshold = settings.DETECTION_MIN_CONFIDENCE
        for result in model.predict(image, verbose=False, conf=conf_threshold):
            names = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                detections.append(Detection(
                    label=str(names[int(box.cls[0])]),
                    confidence=float(box.conf[0]),
                    source=SOURCE_YOLO,
                    x=x1 / width,
                    y=y1 / height,
                    width=(x2 - x1) / width,
                    height=(y2 - y1) / height,
                ))

        return detections


class PaddleTextDetector:
    def __init__(self, lang=None, enable_mkldnn=None):
        self.lang = lang or settings.PADDLEOCR_LANG
        if enable_mkldnn is None:
            enable_mkldnn = settings.PADDLEOCR_ENABLE_MKLDNN
        self.enable_mkldnn = enable_mkldnn
        self._engine = None

    def _load(self):
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise ImproperlyConfigured(
                    "The 'paddle' detection backend requires paddleocr and paddlepaddle. "
                    "Install them or set DETECTION_TEXT_BACKEND = 'stub'."
                ) from exc
    def _load(self):
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise ImproperlyConfigured(
                    "The 'paddle' detection backend requires paddleocr and paddlepaddle. "
                    "Install them or set DETECTION_TEXT_BACKEND = 'stub'."
                ) from exc
            self._engine = PaddleOCR(
                lang=self.lang,
                enable_mkldnn=self.enable_mkldnn,
                use_doc_orientation_classify=settings.PADDLEOCR_USE_DOC_ORIENTATION,
                use_doc_unwarping=settings.PADDLEOCR_USE_DOC_UNWARPING,
                use_textline_orientation=settings.PADDLEOCR_USE_TEXTLINE_ORIENTATION,
                text_det_limit_type=settings.PADDLEOCR_DET_LIMIT_TYPE,
                text_det_limit_side_len=settings.PADDLEOCR_DET_LIMIT_SIDE_LEN,
                text_det_unclip_ratio=settings.PADDLEOCR_DET_UNCLIP_RATIO,
            )
        return self._engine

    def __call__(self, image):
        import numpy as np

        engine = self._load()
        width, height = image.size
        array = np.asarray(image)

        detections = []
        for polygon, text, score in self._iter_results(engine, array):
            xs, ys = self._polygon_coords(polygon)
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            # Pad relative to the box (not the full image) so small button
            # labels expand toward the pill chrome without blowing up titles.
            box_w = max(x2 - x1, 1.0)
            box_h = max(y2 - y1, 1.0)
            pad_x = max(width * 0.004, box_w * 0.10)
            pad_y = max(height * 0.004, box_h * 0.20)
            x1 = max(0.0, x1 - pad_x)
            y1 = max(0.0, y1 - pad_y)
            x2 = min(float(width), x2 + pad_x)
            y2 = min(float(height), y2 + pad_y)

            detections.append(Detection(
                label=text,
                confidence=float(score),
                source=SOURCE_OCR,
                x=x1 / width,
                y=y1 / height,
                width=(x2 - x1) / width,
                height=(y2 - y1) / height,
                text_content=text,
            ))

        return detections

    @staticmethod
    def _aabb_to_polygon(box):
        x1, y1, x2, y2 = (float(value) for value in list(box)[:4])
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    @staticmethod
    def _polygon_coords(polygon):
        """
        Return (xs, ys) float lists from a PaddleOCR polygon.

        PaddleOCR 3.x may return a flat numpy array of length 2N (where N is
        the number of corners, usually 4) or an (N, 2) array. Normalise to
        (N, 2) before extracting coordinates so we never mis-index a flat
        array as a list of (x, y) pairs.
        """
        import numpy as np

        pts = np.asarray(polygon, dtype=float)
        if pts.ndim == 1:
            # flat [x0, y0, x1, y1, ...] → reshape to (N, 2)
            pts = pts.reshape(-1, 2)
        xs = pts[:, 0].tolist()
        ys = pts[:, 1].tolist()
        return xs, ys

    @staticmethod
    def _iter_results(engine, array):
        """
        Yield (polygon, text, score) across PaddleOCR result shapes.

        PaddleOCR 3.x returns dict-like results from predict(); older releases
        return nested lists from ocr().
        """
        if hasattr(engine, 'predict'):
            for result in engine.predict(array):
                texts = result['rec_texts']
                scores = result['rec_scores']
                # rec_boxes are axis-aligned in the input image. Prefer them
                # over dt_polys, which are the raw quadrilaterals and look
                # skewed once we take min/max on stylized or slightly rotated text.
                if result.get('rec_boxes') is not None:
                    polygons = [
                        PaddleTextDetector._aabb_to_polygon(box)
                        for box in result['rec_boxes']
                    ]
                else:
                    polygons = result['dt_polys']
                yield from zip(polygons, texts, scores)
            return

        for page in engine.ocr(array) or []:
            for polygon, (text, score) in page or []:
                yield polygon, text, score


OBJECT_BACKENDS = {
    'stub': StubObjectDetector,
    'yolo': YoloObjectDetector,
}

TEXT_BACKENDS = {
    'stub': StubTextDetector,
    'paddle': PaddleTextDetector,
}


def _build(registry, name, setting_name):
    try:
        backend = registry[name]
    except KeyError:
        options = ', '.join(sorted(registry))
        raise ImproperlyConfigured(
            f"Unknown {setting_name} '{name}'. Available backends: {options}."
        ) from None
    return backend()


# Cached because constructing a backend loads model weights, which is far too
# slow to repeat per request.
@lru_cache(maxsize=None)
def get_object_detector():
    return _build(OBJECT_BACKENDS, settings.DETECTION_OBJECT_BACKEND, 'DETECTION_OBJECT_BACKEND')


@lru_cache(maxsize=None)
def get_text_detector():
    return _build(TEXT_BACKENDS, settings.DETECTION_TEXT_BACKEND, 'DETECTION_TEXT_BACKEND')
