"""
Fal.ai LivePortrait: crop a detected person, animate the face, return a clip.

The still poster and Lottie UI overlays stay in Remotion. This module only
produces the character layer.
"""

import logging
import os

import requests
from django.conf import settings
from PIL import Image

logger = logging.getLogger(__name__)

SUBJECT_SOURCES = {'yolo'}
DEFAULT_DRIVING = (
    'https://storage.googleapis.com/falserverless/model_tests/'
    'live-portrait/liveportrait-example.mp4'
)


def fal_configured():
    return bool(getattr(settings, 'FAL_KEY', ''))


def _as_region(item):
    if isinstance(item, dict):
        return {
            'source': (item.get('source') or 'manual').lower(),
            'label': item.get('label') or '',
            'x': float(item['x']),
            'y': float(item['y']),
            'width': float(item['width']),
            'height': float(item['height']),
        }
    return {
        'source': (getattr(item, 'source', None) or 'manual').lower(),
        'label': getattr(item, 'label', None) or '',
        'x': float(item.x),
        'y': float(item.y),
        'width': float(item.width),
        'height': float(item.height),
    }


def _is_subject(region):
    source = (region.get('source') or '').lower()
    label = (region.get('label') or '').lower()
    return source in SUBJECT_SOURCES or 'person' in label


def primary_person(regions):
    subjects = [_as_region(item) for item in (regions or []) if _is_subject(_as_region(item))]
    if not subjects:
        return None
    return max(subjects, key=lambda item: item['width'] * item['height'])


def should_use_fal(regions, effects=None):
    return fal_configured() and primary_person(regions) is not None


def _pixel_box(region, size, *, face=True, pad=0.08):
    width, height = size
    left = region['x'] * width
    top = region['y'] * height
    box_w = region['width'] * width
    box_h = region['height'] * height
    if face:
        box_h = min(box_h, max(box_w * 1.15, box_h * 0.58))
    pad_x = box_w * pad
    pad_y = box_h * pad
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(width, left + box_w + 2 * pad_x)
    bottom = min(height, top + box_h + 2 * pad_y)
    left_i = int(round(left))
    top_i = int(round(top))
    width_i = max(32, int(round(right - left)))
    height_i = max(32, int(round(bottom - top)))
    width_i = min(width_i, width - left_i)
    height_i = min(height_i, height - top_i)
    return left_i, top_i, width_i, height_i


def _region_norm(box, size):
    left, top, width, height = box
    img_w, img_h = size
    return {
        'source': 'yolo',
        'label': 'person',
        'x': left / img_w,
        'y': top / img_h,
        'width': width / img_w,
        'height': height / img_h,
    }


def _liveportrait_args(effects):
    chosen = set(effects or [])
    args = {
        'flag_lip_zero': True,
        'flag_stitching': True,
        'flag_relative': True,
        'flag_pasteback': True,
        'flag_do_crop': True,
        'flag_do_rot': True,
        'dsize': 512,
        'scale': 2.3,
        'vy_ratio': -0.125,
    }
    if 'breathe' in chosen:
        args['blink'] = 0.4
        args['smile'] = 0.12
    if 'float' in chosen:
        args['rotate_pitch'] = 5
        args['rotate_yaw'] = 4
    if 'zoom' in chosen:
        args['scale'] = 2.6
    return args


def _video_url(result):
    data = result
    if hasattr(result, 'get'):
        data = result.get('data', result)
    video = data.get('video') if isinstance(data, dict) else None
    if isinstance(video, dict):
        return video.get('url')
    if isinstance(video, str):
        return video
    raise RuntimeError(f'Fal LivePortrait returned no video URL: {result!r}')


def render_liveportrait(image, regions, effects, output_path):
    """
    Crop the main person, run Fal LivePortrait, download the face clip.

    Returns (output_path, person_region_norm, request_id).
    """
    person = primary_person(regions)
    if person is None:
        raise ValueError('LivePortrait needs a person region.')
    key = (getattr(settings, 'FAL_KEY', '') or '').strip().strip("'\"")
    if not key:
        raise RuntimeError('Set FAL_KEY in gif/core/settings/base.py')

    os.environ['FAL_KEY'] = key
    import fal_client

    rgb = image.convert('RGB')
    box = _pixel_box(person, rgb.size)
    crop = rgb.crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
    # Data URL avoids Fal CDN token minting, which 403s for some API keys.
    image_url = fal_client.encode_image(crop, format='jpeg')

    driving = getattr(settings, 'FAL_DRIVING_VIDEO', '') or DEFAULT_DRIVING
    arguments = {
        'image_url': image_url,
        'video_url': driving,
        **_liveportrait_args(effects),
    }
    logger.info('Submitting Fal LivePortrait for person box %s', box)
    result = fal_client.subscribe(
        'fal-ai/live-portrait',
        arguments=arguments,
        with_logs=True,
    )
    video_url = _video_url(result)
    response = requests.get(video_url, timeout=120)
    response.raise_for_status()
    with open(output_path, 'wb') as handle:
        handle.write(response.content)
    if not os.path.getsize(output_path):
        raise RuntimeError('Downloaded LivePortrait video was empty')
    request_id = ''
    if isinstance(result, dict):
        request_id = str(result.get('request_id') or result.get('requestId') or '')
    logger.info('LivePortrait wrote %s', output_path)
    return output_path, _region_norm(box, rgb.size), request_id
