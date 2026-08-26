"""
Kling image-to-video with motion brush.

Selected person boxes become a dynamic mask (they move). Cards, buttons,
and title become a static mask (they stay still). Remotion then overlays
shine and the other UI effects on the returned clip.

Motion brush is only on kling-v1 (std, 5s) and kling-v1-5 (pro, 5s).
"""

import base64
import io
import logging
import math
import os
import time

import requests
from django.conf import settings
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)

SUBJECT_SOURCES = {'yolo'}
UI_SOURCES = {'card', 'button', 'title', 'ocr'}
MOTION_EFFECTS = {'zoom', 'float', 'breathe'}


def kling_configured():
    return bool(getattr(settings, 'KLING_API_KEY', ''))


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


def _is_ui(region):
    return (region.get('source') or '').lower() in UI_SOURCES


def should_use_kling(regions, effects):
    if not kling_configured():
        return False
    boxes = [_as_region(item) for item in (regions or [])]
    chosen = set(effects or [])
    return any(_is_subject(item) for item in boxes) and bool(chosen & MOTION_EFFECTS)


def _headers():
    api_key = getattr(settings, 'KLING_API_KEY', '')
    if not api_key:
        raise RuntimeError('Set KLING_API_KEY in gif/core/settings/base.py')
    return {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }


def _base_url():
    return getattr(settings, 'KLING_API_BASE', 'https://api.klingai.com').rstrip('/')


def _png_b64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', optimize=True)
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def _jpeg_b64(image, quality=90):
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, format='JPEG', quality=quality)
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def _pixel_box(region, size):
    width, height = size
    left = int(round(region['x'] * width))
    top = int(round(region['y'] * height))
    box_w = max(1, int(round(region['width'] * width)))
    box_h = max(1, int(round(region['height'] * height)))
    left = min(max(left, 0), width - 1)
    top = min(max(top, 0), height - 1)
    box_w = min(box_w, width - left)
    box_h = min(box_h, height - top)
    return left, top, box_w, box_h


def _paint_regions(size, regions, *, ellipse=False):
    mask = Image.new('L', size, 0)
    if not regions:
        return mask
    draw = ImageDraw.Draw(mask)
    for region in regions:
        left, top, box_w, box_h = _pixel_box(region, size)
        box = [left, top, left + box_w, top + box_h]
        if ellipse:
            draw.ellipse(box, fill=255)
        else:
            radius = max(8, min(box_w, box_h) // 8)
            draw.rounded_rectangle(box, radius=radius, fill=255)
    return mask.filter(ImageFilter.GaussianBlur(2))


def _kling_xy(cx, cy, width, height):
    """Kling trajectories use origin at the bottom-left."""
    x = min(max(int(round(cx)), 0), width - 1)
    y = min(max(int(round(height - 1 - cy)), 0), height - 1)
    return x, y


def _trajectories(region, size, effects):
    width, height = size
    left, top, box_w, box_h = _pixel_box(region, size)
    cx = left + box_w / 2.0
    cy = top + box_h / 2.0
    amp = box_h * (0.04 if 'float' in effects or 'breathe' in effects else 0.018)
    if 'zoom' in effects:
        amp = max(amp, box_h * 0.025)
    points = []
    for index in range(8):
        t = index / 7.0
        dy = amp * math.sin(t * math.pi * 2.0)
        dx = 0.0
        if 'zoom' in effects:
            dx = (box_w * 0.012) * math.sin(t * math.pi)
        kx, ky = _kling_xy(cx + dx, cy + dy, width, height)
        points.append({'x': kx, 'y': ky})
    return points


def _prompt(effects):
    parts = []
    chosen = set(effects or [])
    if 'zoom' in chosen:
        parts.append('the character slowly leans a little closer')
    if 'float' in chosen:
        parts.append('the character gently floats up and down')
    if 'breathe' in chosen:
        parts.append('subtle idle breathing, glowing eyes stay on the face')
    if not parts:
        parts.append('subtle idle character motion')
    parts.append('keep all text, numbers, logos, bonus cards, and UI completely still and sharp')
    return ', '.join(parts)


def _raise_api(payload, context):
    code = payload.get('code')
    if code in (0, '0', None):
        return
    message = payload.get('message') or payload.get('msg') or 'unknown error'
    raise RuntimeError(f'Kling {context} failed ({code}): {message}')


def _create_task(body):
    url = f'{_base_url()}/v1/videos/image2video'
    response = requests.post(url, headers=_headers(), json=body, timeout=60)
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f'Kling create returned non-JSON ({response.status_code}): {response.text[:500]}') from exc
    if response.status_code >= 400:
        raise RuntimeError(f'Kling create HTTP {response.status_code}: {payload}')
    _raise_api(payload, 'create')
    task_id = (payload.get('data') or {}).get('task_id')
    if not task_id:
        raise RuntimeError(f'Kling create had no task_id: {payload}')
    return task_id


def _poll_task(task_id, timeout_s=600):
    url = f'{_base_url()}/v1/videos/image2video/{task_id}'
    deadline = time.time() + timeout_s
    delay = 4
    while time.time() < deadline:
        response = requests.get(url, headers=_headers(), timeout=60)
        payload = response.json()
        if response.status_code >= 400:
            raise RuntimeError(f'Kling poll HTTP {response.status_code}: {payload}')
        _raise_api(payload, 'poll')
        data = payload.get('data') or {}
        status = (data.get('task_status') or '').lower()
        if status == 'succeed':
            videos = (data.get('task_result') or {}).get('videos') or []
            if not videos or not videos[0].get('url'):
                raise RuntimeError(f'Kling succeeded without a video URL: {payload}')
            return videos[0]['url']
        if status == 'failed':
            raise RuntimeError(data.get('task_status_msg') or 'Kling task failed')
        logger.info('Kling task %s is %s', task_id, status or 'pending')
        time.sleep(delay)
        delay = min(delay + 2, 12)
    raise RuntimeError(f'Kling task {task_id} timed out')


def _download(url, dest_path):
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with open(dest_path, 'wb') as handle:
        handle.write(response.content)
    if not dest_path or not os.path.getsize(dest_path):
        raise RuntimeError('Downloaded Kling video was empty')
    return dest_path


def render_kling_video(image, regions, effects, output_path):
    """
    Run Kling image-to-video for the selected person boxes.

    Returns (output_path, task_id).
    """
    boxes = [_as_region(item) for item in regions]
    subjects = [item for item in boxes if _is_subject(item)]
    ui = [item for item in boxes if _is_ui(item)]
    if not subjects:
        raise ValueError('Kling needs a person region selected.')

    rgb = image.convert('RGB')
    size = rgb.size
    poster_b64 = _jpeg_b64(rgb)
    dynamic_mask = _paint_regions(size, subjects, ellipse=True)
    static_mask = _paint_regions(size, ui, ellipse=False) if ui else None
    chosen = list(effects or [])

    primary = max(subjects, key=lambda item: item['width'] * item['height'])
    body = {
        'model_name': getattr(settings, 'KLING_MODEL', 'kling-v1'),
        'mode': getattr(settings, 'KLING_MODE', 'std'),
        'duration': str(getattr(settings, 'KLING_DURATION', '5')),
        'aspect_ratio': '1:1' if abs(size[0] - size[1]) < 8 else '16:9',
        'image': poster_b64,
        'prompt': _prompt(chosen),
        'negative_prompt': (
            'warped text, moving numbers, morphing logo, extra heads, '
            'blurry gold UI, changing bonus amounts'
        ),
        'cfg_scale': 0.6,
        'dynamic_masks': [
            {
                'mask': _png_b64(dynamic_mask),
                'trajectories': _trajectories(primary, size, chosen),
            }
        ],
    }
    if static_mask is not None and static_mask.getextrema()[1] > 0:
        body['static_mask'] = _png_b64(static_mask)

    logger.info('Submitting Kling image2video (%s subjects, %s UI freeze boxes)', len(subjects), len(ui))
    task_id = _create_task(body)
    logger.info('Kling task %s submitted', task_id)
    video_url = _poll_task(task_id)
    _download(video_url, output_path)
    logger.info('Kling wrote %s', output_path)
    return output_path, task_id
