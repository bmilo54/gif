"""
Detection backends.

A detector is any callable taking a preprocessed PIL image and returning a
list of `Detection`. Swapping engines is a settings change, so the rest of the
pipeline never imports ultralytics or paddleocr directly. Those libraries are
imported lazily inside each backend because they are multi-gigabyte optional
dependencies that the web tier does not need to load.
"""
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from ..choices import SOURCE_OCR, SOURCE_YOLO

logger = logging.getLogger(__name__)


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
        for result in model.predict(image, verbose=False, conf=conf_threshold, classes=[0]):
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


class YoloWorldCharacterDetector:
    """
    Open-vocabulary character detector backed by YOLO-World.

    Finds real people, cartoon characters, anime heroes, 3D-rendered avatars,
    mascots — anything described by the configurable CHARACTER_CONCEPTS list in
    settings — and returns them with source=SOURCE_YOLO and label='person' so
    the rest of the pipeline (SAM segmentation, inpainting) treats them as
    character layers.
    """

    def __init__(self, model_path=None):
        self.model_path = model_path or getattr(
            settings, 'YOLOWORLD_MODEL',
            'yolov8s-worldv2.pt',
        )
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from ultralytics import YOLOWorld
            except ImportError as exc:
                raise ImproperlyConfigured(
                    "YoloWorldCharacterDetector requires ultralytics with YOLOWorld support."
                ) from exc
            self._model = YOLOWorld(self.model_path)
        return self._model

    def __call__(self, image):
        concepts = list(getattr(settings, 'CHARACTER_CONCEPTS', [
            'person',
            'man',
            'woman',
            'character',
            'cartoon character',
            'anime character',
            '3D character',
            'mascot',
            'game character',
            'hero',
        ]))
        if not concepts:
            return []

        model = self._load()
        model.set_classes(concepts)

        width, height = image.size
        import numpy as np
        from PIL import Image as _PILImage
        array = np.asarray(image.convert('RGB'))

        min_conf = getattr(settings, 'CHARACTER_MIN_CONFIDENCE', 0.18)
        try:
            results = model.predict(
                array,
                verbose=False,
                conf=min_conf,
                iou=0.5,
            )
        except Exception:
            import logging as _logging
            _logging.getLogger(__name__).exception('YOLOWorld character detection failed')
            return []

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy
            confs = boxes.conf
            if hasattr(xyxy, 'cpu'):
                xyxy = xyxy.cpu().numpy()
                confs = confs.cpu().numpy()
            for coords, conf in zip(xyxy, confs):
                x1, y1, x2, y2 = (float(v) for v in coords)
                detections.append(Detection(
                    # Always label as 'person' so _is_person_region() recognises it
                    label='person',
                    confidence=float(conf),
                    source=SOURCE_YOLO,
                    x=x1 / width,
                    y=y1 / height,
                    width=(x2 - x1) / width,
                    height=(y2 - y1) / height,
                ))
        return detections



@lru_cache(maxsize=1)
def _get_paddle_engine(lang, enable_mkldnn, use_doc_orientation, use_doc_unwarping,
                       use_textline_orientation, det_limit_type, det_limit_side_len,
                       det_unclip_ratio):
    """
    Module-level singleton for the PaddleOCR engine.
    lru_cache ensures PaddleOCR is only initialised ONCE per process,
    no matter how many requests come in.
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise ImproperlyConfigured(
            "The 'paddle' detection backend requires paddleocr and paddlepaddle. "
            "Install them or set DETECTION_TEXT_BACKEND = 'stub'."
        ) from exc

    import logging as _logging
    _logging.getLogger(__name__).info('Initialising PaddleOCR engine (one-time setup)...')
    return PaddleOCR(
        lang=lang,
        enable_mkldnn=enable_mkldnn,
        use_doc_orientation_classify=use_doc_orientation,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
        text_det_limit_type=det_limit_type,
        text_det_limit_side_len=det_limit_side_len,
        text_det_unclip_ratio=det_unclip_ratio,
    )


class PaddleTextDetector:
    def __init__(self, lang=None, enable_mkldnn=None):
        self.lang = lang or settings.PADDLEOCR_LANG
        if enable_mkldnn is None:
            enable_mkldnn = settings.PADDLEOCR_ENABLE_MKLDNN
        self.enable_mkldnn = enable_mkldnn

    def _load(self):
        return _get_paddle_engine(
            lang=self.lang,
            enable_mkldnn=self.enable_mkldnn,
            use_doc_orientation=settings.PADDLEOCR_USE_DOC_ORIENTATION,
            use_doc_unwarping=settings.PADDLEOCR_USE_DOC_UNWARPING,
            use_textline_orientation=settings.PADDLEOCR_USE_TEXTLINE_ORIENTATION,
            det_limit_type=settings.PADDLEOCR_DET_LIMIT_TYPE,
            det_limit_side_len=settings.PADDLEOCR_DET_LIMIT_SIDE_LEN,
            det_unclip_ratio=settings.PADDLEOCR_DET_UNCLIP_RATIO,
        )


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


class GoogleVisionTextDetector:
    def __init__(self):
        self._client = None

    def _load(self):
        if self._client is None:
            try:
                from google.cloud import vision
            except ImportError as exc:
                raise ImproperlyConfigured("Install google-cloud-vision to use GoogleVisionTextDetector") from exc
            
            creds = getattr(settings, 'GOOGLE_APPLICATION_CREDENTIALS', '')
            if creds:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds
                
            self._client = vision.ImageAnnotatorClient()
            self._vision = vision
        return self._client, self._vision

    def __call__(self, image):
        import io
        client, vision = self._load()
        
        # Convert PIL to bytes
        img_byte_arr = io.BytesIO()
        image.convert('RGB').save(img_byte_arr, format='JPEG')
        content = img_byte_arr.getvalue()
        
        v_image = vision.Image(content=content)
        # Paragraph boxes, not words. Word hits on decorative gold type become
        # a tiled grid once layout clustering unions neighbours.
        response = client.document_text_detection(image=v_image)
        
        if response.error.message:
            raise Exception(f"{response.error.message}")

        detections = []
        width, height = image.size
        document = response.full_text_annotation
        if not document:
            return detections

        for page in document.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    text = " ".join(
                        "".join(symbol.text for symbol in word.symbols)
                        for word in paragraph.words
                    ).strip()
                    if len("".join(text.split())) < 2:
                        continue
                    vertices = paragraph.bounding_box.vertices
                    xs = [v.x for v in vertices]
                    ys = [v.y for v in vertices]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    box_w = max(x2 - x1, 1.0)
                    box_h = max(y2 - y1, 1.0)
                    pad_x = max(width * 0.003, box_w * 0.04)
                    pad_y = max(height * 0.003, box_h * 0.10)
                    x1 = max(0.0, x1 - pad_x)
                    y1 = max(0.0, y1 - pad_y)
                    x2 = min(float(width), x2 + pad_x)
                    y2 = min(float(height), y2 + pad_y)
                    detections.append(Detection(
                        label=text,
                        confidence=1.0,
                        source=SOURCE_OCR,
                        x=x1 / width,
                        y=y1 / height,
                        width=(x2 - x1) / width,
                        height=(y2 - y1) / height,
                        text_content=text,
                    ))
        return detections


class GPT4VisionCharacterDetector:
    def __init__(self):
        self._client = None

    def _load(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImproperlyConfigured("Install openai to use GPT4VisionCharacterDetector") from exc
            self._client = OpenAI(api_key=getattr(settings, 'OPENAI_API_KEY', None))
        return self._client

    def __call__(self, image):
        import base64
        import io
        import json
        from PIL import Image as _PILImage
        client = self._load()

        # Resize to max 1024px to save tokens and speed up API call
        MAX_SIDE = 1024
        orig_w, orig_h = image.size
        scale = min(MAX_SIDE / orig_w, MAX_SIDE / orig_h, 1.0)
        if scale < 1.0:
            canvas_w = int(orig_w * scale)
            canvas_h = int(orig_h * scale)
            canvas = image.convert('RGB').resize((canvas_w, canvas_h), _PILImage.LANCZOS)
        else:
            canvas = image.convert('RGB')

        img_byte_arr = io.BytesIO()
        canvas.save(img_byte_arr, format='JPEG', quality=90)
        base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        prompt = (
            "Identify ALL characters/people (including partially visible ones) and any props they hold "
            "(coins, gifts, weapons, cards).\n"
            "Return a JSON array. Each item:\n"
            "  - 'label': 'person' for human/character, or the exact prop name for props.\n"
            "  - 'box': [x_min, y_min, x_max, y_max] where each value is an integer from 0 to 1000 relative to the image dimensions (0 is top/left, 1000 is bottom/right).\n"
            "Output ONLY valid JSON, no markdown, no extra text."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high",   # use high-detail mode for accurate coords
                            }
                        }
                    ]
                }
            ],
            max_tokens=800
        )

        content = response.choices[0].message.content.strip()
        # Strip optional markdown code fence
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        try:
            results = json.loads(content)
        except json.JSONDecodeError:
            import logging
            logging.getLogger(__name__).error("GPT-4o Vision returned invalid JSON: %s", content)
            return []

        detections = []
        for res in results:
            label = res.get('label', 'person')
            box = res.get('box')
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]

            # Coords are 0-1000 → normalize to 0-1
            detections.append(Detection(
                label=label,
                confidence=0.99,
                source=SOURCE_YOLO,
                x=x1 / 1000.0,
                y=y1 / 1000.0,
                width=(x2 - x1) / 1000.0,
                height=(y2 - y1) / 1000.0,
            ))

        return detections


class OcrSpaceTextDetector:
    def __init__(self):
        self.api_key = getattr(settings, 'OCRSPACE_API_KEY', None)
        if not self.api_key:
            raise ImproperlyConfigured("OCRSPACE_API_KEY is not set in settings.")
            
    def __call__(self, image):
        import base64
        import io
        import requests
        
        img_byte_arr = io.BytesIO()
        # Resize if too large (OCR.Space free tier limit is 1MB)
        # Assuming original is around 1000px, JPEG 95% is fine
        image.convert('RGB').save(img_byte_arr, format='JPEG', quality=85)
        image_bytes = img_byte_arr.getvalue()
        base64_image = "data:image/jpg;base64," + base64.b64encode(image_bytes).decode('utf-8')
        
        response = requests.post(
            "https://api.ocr.space/parse/image",
            data={
                "base64Image": base64_image,
                "apikey": self.api_key,
                "isOverlayRequired": True,
                "language": "eng",
                "scale": False,
                "detectOrientation": False,
                "isTable": False,
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"OCR.Space API failed with status {response.status_code}")
            
        result = response.json()
        if result.get("IsErroredOnProcessing"):
            raise Exception(f"OCR.Space Error: {result.get('ErrorMessage')}")

        parsed_results = result.get("ParsedResults") or []
        width, height = image.size
        scale_x, scale_y = self._overlay_scale(parsed_results, width, height)

        # One box per text line. Word-level boxes on stylized posters become
        # a grid: ornaments get read as extra words, then layout clustering
        # tiles them into fake cards.
        detections = []
        for parsed_result in parsed_results:
            overlay = parsed_result.get("TextOverlay") or {}
            for line in overlay.get("Lines") or []:
                words = [
                    word for word in (line.get("Words") or [])
                    if (word.get("WordText") or "").strip()
                ]
                if not words:
                    continue
                text = " ".join(word["WordText"].strip() for word in words)
                if not self._usable_ocr_text(text):
                    continue
                x1 = min(float(word["Left"]) for word in words) * scale_x
                y1 = min(float(word["Top"]) for word in words) * scale_y
                x2 = max(float(word["Left"]) + float(word["Width"]) for word in words) * scale_x
                y2 = max(float(word["Top"]) + float(word["Height"]) for word in words) * scale_y
                box_w = max(x2 - x1, 1.0)
                box_h = max(y2 - y1, 1.0)
                pad_x = max(width * 0.003, box_w * 0.04)
                pad_y = max(height * 0.003, box_h * 0.10)
                x1 = max(0.0, x1 - pad_x)
                y1 = max(0.0, y1 - pad_y)
                x2 = min(float(width), x2 + pad_x)
                y2 = min(float(height), y2 + pad_y)
                detections.append(Detection(
                    label=text,
                    confidence=1.0,
                    source=SOURCE_OCR,
                    x=x1 / width,
                    y=y1 / height,
                    width=(x2 - x1) / width,
                    height=(y2 - y1) / height,
                    text_content=text,
                ))
        return detections

    @staticmethod
    def _usable_ocr_text(text):
        compact = "".join(text.split())
        if len(compact) < 2:
            return False
        return sum(ch.isalnum() for ch in compact) >= 2

    @staticmethod
    def _overlay_scale(parsed_results, width, height):
        max_x = max_y = 0.0
        for parsed_result in parsed_results:
            overlay = parsed_result.get("TextOverlay") or {}
            for line in overlay.get("Lines") or []:
                for word in line.get("Words") or []:
                    max_x = max(
                        max_x,
                        float(word.get("Left") or 0) + float(word.get("Width") or 0),
                    )
                    max_y = max(
                        max_y,
                        float(word.get("Top") or 0) + float(word.get("Height") or 0),
                    )
        scale_x = width / max_x if max_x > width * 1.05 else 1.0
        scale_y = height / max_y if max_y > height * 1.05 else 1.0
        return scale_x, scale_y


class ReplicateObjectDetector:
    def __init__(self):
        self.model_version = "ultralytics/yolov8s-worldv2:96a016a98290d3ff1f3ed8942c916379701c84da9b6d5b19a107b1f86cdc97f5"

    def __call__(self, image):
        import base64
        import io
        import json
        import os
        try:
            import replicate
        except ImportError as exc:
            raise ImproperlyConfigured("Install replicate to use ReplicateObjectDetector") from exc

        token = getattr(settings, 'REPLICATE_API_TOKEN', '')
        if not token:
            raise ImproperlyConfigured("REPLICATE_API_TOKEN is not set in settings.")
        os.environ['REPLICATE_API_TOKEN'] = token

        # Ultralytics on Replicate saves file inputs as /tmp/...download with
        # no extension, then refuses them. A JPEG data URI keeps the type.
        buf = io.BytesIO()
        image.convert('RGB').save(buf, format='JPEG', quality=95)
        image_uri = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')

        concepts = getattr(settings, 'CHARACTER_CONCEPTS', ['person'])
        output = replicate.run(
            self.model_version,
            input={
                "image": image_uri,
                "class_names": ",".join(concepts),
                "iou": 0.5,
                "conf": 0.15,
                "return_json": True,
            },
        )

        if hasattr(output, 'read'):
            output = output.read()
        if isinstance(output, (bytes, bytearray)):
            output = output.decode('utf-8', errors='replace')
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                logger.warning('Replicate YOLO returned non-JSON: %s', output[:300])
                return []

        detections = []
        width, height = image.size
        rows = output
        if isinstance(output, dict):
            rows = output.get('predictions') or output.get('results') or output.get('boxes') or []
        if not isinstance(rows, list):
            return detections

        for res in rows:
            if not isinstance(res, dict):
                continue
            box = res.get('box_2d') or res.get('bbox') or res.get('xyxy')
            if not box or len(box) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            if max(x2, y2) <= 1.5:
                x1, x2 = x1 * width, x2 * width
                y1, y2 = y1 * height, y2 * height
            conf = res.get('score') or res.get('confidence') or 0.99
            detections.append(Detection(
                label='person',
                confidence=float(conf),
                source=SOURCE_YOLO,
                x=x1 / width,
                y=y1 / height,
                width=(x2 - x1) / width,
                height=(y2 - y1) / height,
            ))
        return detections


OBJECT_BACKENDS = {
    'stub': StubObjectDetector,
    'yolo': YoloObjectDetector,
    'yolo-world': YoloWorldCharacterDetector,
    'gpt4o': GPT4VisionCharacterDetector,
    'replicate': ReplicateObjectDetector,
}

TEXT_BACKENDS = {
    'stub': StubTextDetector,
    'paddle': PaddleTextDetector,
    'google': GoogleVisionTextDetector,
    'ocrspace': OcrSpaceTextDetector,
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
def get_character_detector():
    """YOLO-World open-vocab character detector (always active when model is present)."""
    if not getattr(settings, 'CHARACTER_DETECTION_ENABLED', True):
        return None
    try:
        return YoloWorldCharacterDetector()
    except Exception:
        import logging as _log
        _log.getLogger(__name__).warning(
            'YOLOWorld character detector could not be loaded; '
            'cartoon/3D characters will not be auto-detected.'
        )
        return None

@lru_cache(maxsize=None)
def get_text_detector():
    return _build(TEXT_BACKENDS, settings.DETECTION_TEXT_BACKEND, 'DETECTION_TEXT_BACKEND')
