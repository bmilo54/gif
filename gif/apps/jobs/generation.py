import io
import logging
import math
from types import SimpleNamespace

import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

from apps.projects.services.preprocessing import load_preprocessed_image

from .choices import DEFAULT_ANIMATION_TYPES, EFFECT_APPLY_ORDER
from .models import AnimationJob

logger = logging.getLogger(__name__)

STATUS_PROCESSING = 'processing'
STATUS_COMPLETED = 'completed'
STATUS_FAILED = 'failed'

EXPAND_FRAC = 0.16


def _pixel_box(detection, size):
    width, height = size
    left = int(round(detection.x * width))
    top = int(round(detection.y * height))
    box_width = max(1, int(round(detection.width * width)))
    box_height = max(1, int(round(detection.height * height)))

    left = min(max(left, 0), width - 1)
    top = min(max(top, 0), height - 1)
    box_width = min(box_width, width - left)
    box_height = min(box_height, height - top)
    return left, top, box_width, box_height


def _expanded_box(left, top, box_width, box_height, image_size):
    img_w, img_h = image_size
    pad_x = max(14, int(box_width * EXPAND_FRAC))
    pad_y = max(14, int(box_height * EXPAND_FRAC))
    el = max(0, left - pad_x)
    et = max(0, top - pad_y)
    er = min(img_w, left + box_width + pad_x)
    eb = min(img_h, top + box_height + pad_y)
    return el, et, er - el, eb - et, left - el, top - et


def _region_mask(width, height, source):
    """
    Mostly-opaque rounded mask. A tiny inset keeps the effect on the region
    without feathering so much that zoom/fade disappear.
    """
    inset = max(1, int(min(width, height) * 0.02))
    if source == 'button':
        radius = max(4, (height - 2 * inset) // 2)
    elif source == 'card':
        radius = max(8, int(min(width, height) * 0.08))
    else:
        radius = max(4, int(min(width, height) * 0.06))

    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        [inset, inset, width - 1 - inset, height - 1 - inset],
        radius=radius,
        fill=255,
    )
    return mask.filter(ImageFilter.GaussianBlur(1))


def _shine_mask(width, height, progress, band_frac, opacity=1.0):
    band = max(4, int(max(width, height) * band_frac))
    span = width + height + 2 * band
    center = progress * span - band
    rows, cols = np.mgrid[0:height, 0:width]
    dist = np.abs((cols + rows).astype(float) - center)
    falloff = np.clip(1.0 - dist / band, 0.0, 1.0) ** 2
    return (falloff * opacity).astype(np.float32)


def _leading_edge_mask(width, height, progress, edge_frac=0.04):
    edge = max(2, int(max(width, height) * edge_frac))
    band = max(4, int(max(width, height) * 0.20))
    span = width + height + 2 * band
    center = progress * span - band
    rows, cols = np.mgrid[0:height, 0:width]
    diag = (cols + rows).astype(float)
    lead = np.clip(1.0 - np.abs(diag - center) / edge, 0.0, 1.0) ** 3
    return lead.astype(np.float32)


def _screen_color(base, mask_f, color):
    overlay = (mask_f[:, :, np.newaxis] * np.array(color, dtype=np.float32)).astype(np.uint8)
    return ImageChops.screen(base, Image.fromarray(overlay, mode='RGB'))


def _center_crop(src, tw, th, ox=0, oy=0):
    left = max(0, min((src.width - tw) // 2 + ox, src.width - tw))
    top = max(0, min((src.height - th) // 2 + oy, src.height - th))
    return src.crop((left, top, left + tw, top + th))


def effect_shine(crop, wave, progress, **_kwargs):
    lit = ImageEnhance.Brightness(crop).enhance(1.0 + 0.18 * wave)
    w, h = crop.size
    result = _screen_color(lit, _shine_mask(w, h, progress, 0.20, 0.95), (255, 200, 80))
    result = _screen_color(result, _shine_mask(w, h, (progress + 0.35) % 1.0, 0.14, 0.5), (255, 240, 150))
    return _screen_color(result, _leading_edge_mask(w, h, progress), (255, 255, 230))


def effect_glow(crop, wave, progress, **_kwargs):
    return ImageEnhance.Brightness(crop).enhance(1.0 + 0.55 * wave)


def effect_gold_pulse(crop, wave, progress, **_kwargs):
    gold = Image.new('RGB', crop.size, (255, 196, 64))
    tinted = Image.blend(crop, gold, 0.18 + 0.32 * wave)
    return ImageEnhance.Brightness(tinted).enhance(1.0 + 0.18 * wave)


def effect_breathe(crop, wave, progress, expanded=None, box_size=None, **_kwargs):
    src = expanded if expanded is not None else crop
    tw, th = box_size or crop.size
    scale = 1.0 + 0.20 * wave
    nw = max(tw + 1, int(round(src.width * scale)))
    nh = max(th + 1, int(round(src.height * scale)))
    grown = src.resize((nw, nh), Image.Resampling.LANCZOS)
    return _center_crop(grown, tw, th)


def effect_float(crop, wave, progress, expanded=None, box_size=None, **_kwargs):
    src = expanded if expanded is not None else crop
    tw, th = box_size or crop.size
    shift = int(round(max(8, th * 0.06) * math.sin(2.0 * math.pi * progress)))
    return _center_crop(src, tw, th, oy=shift)


def effect_sparkle(crop, wave, progress, sparkles=None, **_kwargs):
    result = ImageEnhance.Brightness(crop).enhance(1.0 + 0.12 * wave)
    arr = np.array(result).astype(np.float32)
    h, w = arr.shape[:2]
    for sx, sy, phase, radius in sparkles or ():
        twinkle = 0.5 * (1.0 + math.sin(2.0 * math.pi * (progress + phase)))
        if twinkle < 0.25:
            continue
        radius = max(radius, 3)
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        blob = np.clip(1.0 - np.sqrt(xx * xx + yy * yy) / max(radius, 1), 0, 1) ** 2
        x0, y0 = int(sx * w), int(sy * h)
        y1, y2 = max(0, y0 - radius), min(h, y0 + radius + 1)
        x1, x2 = max(0, x0 - radius), min(w, x0 + radius + 1)
        by1, by2 = y1 - (y0 - radius), y2 - (y0 - radius)
        bx1, bx2 = x1 - (x0 - radius), x2 - (x0 - radius)
        glow = blob[by1:by2, bx1:bx2, np.newaxis] * twinkle * 220
        arr[y1:y2, x1:x2] = np.clip(arr[y1:y2, x1:x2] + glow, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode='RGB')


def effect_fade(crop, wave, progress, **_kwargs):
    black = Image.new('RGB', crop.size, (0, 0, 0))
    dimmed = Image.blend(crop, black, 0.45 * (1.0 - wave))
    return ImageEnhance.Brightness(dimmed).enhance(0.75 + 0.55 * wave)


def effect_zoom(crop, wave, progress, expanded=None, box_size=None, **_kwargs):
    src = expanded if expanded is not None else crop
    tw, th = box_size or crop.size
    scale = 1.0 + 0.36 * wave
    nw = max(tw + 1, int(round(src.width * scale)))
    nh = max(th + 1, int(round(src.height * scale)))
    grown = src.resize((nw, nh), Image.Resampling.LANCZOS)
    return _center_crop(grown, tw, th)


def effect_rim(crop, wave, progress, **_kwargs):
    w, h = crop.size
    yy, xx = np.mgrid[0:h, 0:w]
    nx = np.minimum(xx, w - 1 - xx) / max(w / 2.0, 1)
    ny = np.minimum(yy, h - 1 - yy) / max(h / 2.0, 1)
    rim = np.clip(1.0 - np.minimum(nx, ny) / 0.22, 0.0, 1.0) ** 2
    rim = rim * (0.45 + 0.55 * wave)
    lit = ImageEnhance.Brightness(crop).enhance(1.0 + 0.12 * wave)
    return _screen_color(lit, rim.astype(np.float32), (255, 214, 96))


def effect_flicker(crop, wave, progress, **_kwargs):
    flicker = 0.5 + 0.5 * (0.5 + 0.5 * math.sin(progress * 22))
    strength = 0.12 + 0.28 * wave * flicker
    gold = Image.new('RGB', crop.size, (255, 210, 90))
    return Image.blend(crop, Image.blend(crop, gold, 0.45), strength)


EFFECTS = {
    'shine': effect_shine,
    'glow': effect_glow,
    'gold_pulse': effect_gold_pulse,
    'breathe': effect_breathe,
    'float': effect_float,
    'sparkle': effect_sparkle,
    'fade': effect_fade,
    'zoom': effect_zoom,
    'rim': effect_rim,
    'flicker': effect_flicker,
}


def _sparkles_for(box_width, box_height, seed):
    rng = np.random.default_rng(abs(hash(seed)) % (2 ** 32))
    count = max(8, int(12 * (box_width * box_height) ** 0.5 / 80))
    points = []
    for _ in range(min(count, 16)):
        points.append((
            float(rng.uniform(0.12, 0.88)),
            float(rng.uniform(0.12, 0.88)),
            float(rng.uniform(0.0, 1.0)),
            int(rng.integers(3, 6)),
        ))
    return points


def _as_region(item):
    if isinstance(item, dict):
        return SimpleNamespace(
            x=float(item['x']),
            y=float(item['y']),
            width=float(item['width']),
            height=float(item['height']),
            source=item.get('source') or 'manual',
        )
    return item


def _ordered_effects(effect_names):
    chosen = [name for name in (effect_names or []) if name in EFFECTS]
    if not chosen:
        chosen = list(DEFAULT_ANIMATION_TYPES)
    return [name for name in EFFECT_APPLY_ORDER if name in chosen]


MOTION_EFFECTS = {'zoom', 'breathe', 'float'}


def apply_effects(crop, expanded, wave, progress, effect_names, sparkles):
    ordered = _ordered_effects(effect_names)
    tw, th = crop.size
    has_motion = any(name in MOTION_EFFECTS for name in ordered)

    if has_motion:
        src = expanded if expanded is not None else crop
        scale = 1.0
        if 'zoom' in ordered:
            scale += 0.36 * wave
        if 'breathe' in ordered:
            scale += 0.18 * wave
        if scale > 1.001:
            nw = max(tw + 1, int(round(src.width * scale)))
            nh = max(th + 1, int(round(src.height * scale)))
            src = src.resize((nw, nh), Image.Resampling.LANCZOS)
        shift = 0
        if 'float' in ordered:
            shift = int(round(max(8, th * 0.06) * math.sin(2.0 * math.pi * progress)))
        result = _center_crop(src, tw, th, oy=shift)
    else:
        result = crop

    for name in ordered:
        if name in MOTION_EFFECTS:
            continue
        result = EFFECTS[name](
            result,
            wave,
            progress,
            sparkles=sparkles,
        )
    return result


def render_gif_bytes(image, detections, *, effects=None, frame_count=None, duration_ms=None):
    """
    Build a looping GIF. Multiple effects are stacked. Motion uses a padded
    crop so zoom/float/breathe have pixels to pull from and stay visible.
    """
    frame_count = frame_count or settings.GIF_FRAME_COUNT
    duration_ms = duration_ms or settings.GIF_DURATION_MS
    effect_names = effects or list(DEFAULT_ANIMATION_TYPES)

    regions = []
    for detection in detections:
        detection = _as_region(detection)
        left, top, box_width, box_height = _pixel_box(detection, image.size)
        el, et, ew, eh, ox, oy = _expanded_box(left, top, box_width, box_height, image.size)
        expanded = image.crop((el, et, el + ew, et + eh)).convert('RGB')
        crop = expanded.crop((ox, oy, ox + box_width, oy + box_height))
        regions.append({
            'crop': crop,
            'expanded': expanded,
            'left': left,
            'top': top,
            'mask': _region_mask(box_width, box_height, detection.source),
            'sparkles': _sparkles_for(box_width, box_height, (detection.x, detection.y, detection.width)),
        })

    frames = []
    for index in range(frame_count):
        wave = 0.5 * (1.0 - math.cos(2.0 * math.pi * index / frame_count))
        progress = index / float(frame_count)
        frame = image.convert('RGB')
        for region in regions:
            animated = apply_effects(
                region['crop'],
                region['expanded'],
                wave,
                progress,
                effect_names,
                region['sparkles'],
            )
            frame.paste(animated, (region['left'], region['top']), region['mask'])
        frames.append(frame)

    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return buffer.getvalue(), frame_count


def generate_gif(job):
    """
    Render the GIF for an AnimationJob and store it on gif_file.

    Uses the project image, adjusted regions, and the job's animation_types.
    """
    if not isinstance(job, AnimationJob):
        job = AnimationJob.objects.select_related('project').prefetch_related('selected_objects').get(pk=job)

    job.status = STATUS_PROCESSING
    job.save(update_fields=['status'])

    try:
        project = job.project
        if not project.image:
            raise ValueError('Project has no image to animate.')

        detections = job.get_regions()
        if not detections:
            raise ValueError('Animation job has no selected regions.')

        image = load_preprocessed_image(project.image, max_side=settings.GIF_MAX_SIDE)
        payload, frame_count = render_gif_bytes(
            image,
            detections,
            effects=job.get_animation_types(),
        )
        filename = f"{project.project_id or 'project'}_v{job.version}.gif"
        if job.gif_file:
            job.gif_file.delete(save=False)
        job.gif_file.save(filename, ContentFile(payload), save=False)
        job.frame_count = frame_count
        job.file_size = len(payload)
        job.status = STATUS_COMPLETED
        job.save(update_fields=['gif_file', 'frame_count', 'file_size', 'status'])
        logger.info(
            "GIF generated for %s v%s (%s, %s frames, %s bytes)",
            project.project_id,
            job.version,
            ','.join(job.get_animation_types()),
            frame_count,
            len(payload),
        )
        return job
    except Exception:
        job.status = STATUS_FAILED
        job.save(update_fields=['status'])
        logger.exception("GIF generation failed for job %s", job.pk)
        raise
