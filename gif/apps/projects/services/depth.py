"""
Depth-map generation for the GIF animation pipeline.

Uses Depth-Anything-Small (HuggingFace transformers) if available,
otherwise falls back to a fast luminance-inversion approximation so
the rest of the pipeline can always continue.
"""
import logging
import numpy as np

from PIL import Image

logger = logging.getLogger(__name__)

_pipeline = None
_pipeline_tried = False


def _load_pipeline():
    global _pipeline, _pipeline_tried
    if _pipeline_tried:
        return _pipeline
    _pipeline_tried = True
    try:
        from transformers import pipeline as hf_pipeline
        logger.info('Loading Depth-Anything model...')
        _pipeline = hf_pipeline(
            task='depth-estimation',
            model='LiheYoung/depth-anything-small-hf',
        )
        logger.info('Depth-Anything model loaded.')
    except Exception as exc:
        logger.warning('Could not load Depth-Anything model: %s', exc)
        _pipeline = None
    return _pipeline


def _luminance_depth(image: Image.Image) -> Image.Image:
    """
    Simple luminance-inversion fallback.
    Brighter pixels → closer (higher depth value).
    Returns an 'L' mode PIL image.
    """
    grey = np.asarray(image.convert('L'), dtype=np.float32)
    # Invert so bright areas appear close
    depth = 255.0 - grey
    return Image.fromarray(depth.astype(np.uint8), mode='L')


def get_depth_map(image: Image.Image) -> Image.Image:
    """
    Generate a greyscale depth map for *image*.

    Returns a PIL Image in 'L' mode (0=far, 255=close), same size
    as the input, or None if generation fails entirely.
    """
    pipe = _load_pipeline()
    if pipe is not None:
        try:
            result = pipe(image)
            depth_pil = result['depth']            # PIL Image
            # Resize to match source in case the model downscaled
            if depth_pil.size != image.size:
                depth_pil = depth_pil.resize(image.size, Image.BILINEAR)
            depth_arr = np.asarray(depth_pil.convert('L'), dtype=np.float32)
            # Normalise to 0-255
            mn, mx = depth_arr.min(), depth_arr.max()
            if mx > mn:
                depth_arr = (depth_arr - mn) / (mx - mn) * 255.0
            return Image.fromarray(depth_arr.astype(np.uint8), mode='L')
        except Exception as exc:
            logger.warning('Depth-Anything inference failed: %s — using luminance fallback', exc)

    logger.info('Using luminance-inversion depth fallback.')
    return _luminance_depth(image)
