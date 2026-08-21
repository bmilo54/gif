import logging
import math
from types import SimpleNamespace

import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

from apps.projects.services.preprocessing import load_preprocessed_image

from .choices import DEFAULT_ANIMATION_TYPES, EFFECT_APPLY_ORDER
from .encoding import encode_animation
from .models import AnimationJob

logger = logging.getLogger(__name__)

STATUS_PROCESSING = 'processing'
STATUS_COMPLETED = 'completed'
STATUS_FAILED = 'failed'

EXPAND_FRAC = 0.16
UI_SOURCES = {'card', 'button', 'title', 'ocr'}
SUBJECT_SOURCES = {'yolo'}
PASTE_ORDER = {
    'yolo': 0,
    'manual': 1,
    'title': 2,
    'ocr': 3,
    'card': 4,
    'button': 5,
}


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


def _is_subject_region(detection):
    source = (getattr(detection, 'source', None) or '').lower()
    label = (getattr(detection, 'label', None) or '').lower()
    return source in SUBJECT_SOURCES or 'person' in label


def _occluder_holes(box, occluders, image_size):
    left, top, width, height = box
    holes = Image.new('L', (width, height), 0)
    if not occluders:
        return holes
    draw = ImageDraw.Draw(holes)
    pad = max(8, int(min(width, height) * 0.025))
    for occluder in occluders:
        occ = _as_region(occluder)
        if _is_subject_region(occ):
            continue
        if (occ.source or '').lower() not in UI_SOURCES:
            continue
        ol, ot, ow, oh = _pixel_box(occ, image_size)
        x1 = ol - left - pad
        y1 = ot - top - pad
        x2 = ol + ow - left + pad
        y2 = ot + oh - top + pad
        if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
            continue
        radius = max(8, oh // 5)
        draw.rounded_rectangle(
            [int(x1), int(y1), int(x2), int(y2)],
            radius=radius,
            fill=255,
        )
    return holes


def _punch_occluders(mask, box, occluders, image_size):
    holes = _occluder_holes(box, occluders, image_size).filter(ImageFilter.GaussianBlur(3))
    keep = np.clip(
        np.array(mask, dtype=np.float32) * (1.0 - np.array(holes, dtype=np.float32) / 255.0),
        0,
        255,
    ).astype(np.uint8)
    return Image.fromarray(keep, mode='L')


def _largest_blob(binary):
    try:
        import cv2
    except ImportError:
        return binary
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _grabcut_silhouette(crop_rgb, hole_arr):
    try:
        import cv2
    except ImportError:
        return None

    h, w = crop_rgb.shape[:2]
    if w < 24 or h < 24:
        return None

    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    inset_x = max(6, int(w * 0.10))
    inset_y = max(6, int(h * 0.06))
    mask[inset_y:h - inset_y, inset_x:w - inset_x] = cv2.GC_PR_FGD
    border_x = max(3, inset_x // 2)
    border_y = max(3, inset_y // 2)
    mask[:border_y, :] = cv2.GC_BGD
    mask[h - border_y:, :] = cv2.GC_BGD
    mask[:, :border_x] = cv2.GC_BGD
    mask[:, w - border_x:] = cv2.GC_BGD
    if hole_arr is not None:
        mask[hole_arr > 80] = cv2.GC_BGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop_rgb, mask, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    result = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    result = _largest_blob(result)
    if result.mean() < 12:
        return None
    result = cv2.GaussianBlur(result, (0, 0), 3.5)
    return result


def _ellipse_mask(width, height):
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    inset_x = max(4, int(width * 0.08))
    inset_y = max(4, int(height * 0.04))
    draw.ellipse([inset_x, inset_y, width - 1 - inset_x, height - 1 - inset_y], fill=255)
    blur = max(4, min(width, height) // 28)
    return mask.filter(ImageFilter.GaussianBlur(blur))


def _subject_mask(image, box, occluders):
    """
    Follow the character silhouette instead of the YOLO rectangle.

    A rectangular glow with card-shaped holes looks like a broken sticker.
    GrabCut (with UI treated as background) plus a soft edge keeps the
    effect on the person.
    """
    left, top, width, height = box
    holes = _occluder_holes(box, occluders, image.size)
    hole_arr = np.array(holes)
    crop = np.array(image.crop((left, top, left + width, top + height)).convert('RGB'))
    silhouette = _grabcut_silhouette(crop, hole_arr)
    if silhouette is None:
        silhouette = np.array(_ellipse_mask(width, height))
    mask = Image.fromarray(silhouette, mode='L')
    mask = _punch_occluders(mask, box, occluders, image.size)
    return mask.filter(ImageFilter.GaussianBlur(2))


def _collect_occluders(job, selected):
    occluders = []
    project = getattr(job, 'project', None)
    if project is not None:
        for detection in project.detections.all():
            if (detection.source or '') in UI_SOURCES:
                occluders.append(detection)
    for region in selected:
        occ = _as_region(region)
        if (occ.source or '').lower() in UI_SOURCES:
            occluders.append(region)
    return occluders


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


def _keep_frame(original, inner, frame_frac=0.20):
    """
    Keep the gold chrome around the edge still. Only the inner art moves,
    so a card does not look like a zooming rectangle.
    """
    w, h = original.size
    fx = max(6, int(w * frame_frac))
    fy = max(6, int(h * frame_frac))
    mask = Image.new('L', (w, h), 0)
    inset_w = max(1, w - 2 * fx)
    inset_h = max(1, h - 2 * fy)
    mask.paste(Image.new('L', (inset_w, inset_h), 255), (fx, fy))
    blur = max(2, min(6, int(min(fx, fy) * 0.12)))
    mask = mask.filter(ImageFilter.GaussianBlur(blur))
    return Image.composite(inner, original, mask)


def _scale_centered(crop, scale):
    if scale <= 1.002:
        return crop
    w, h = crop.size
    nw = max(w + 1, int(round(w * scale)))
    nh = max(h + 1, int(round(h * scale)))
    grown = crop.resize((nw, nh), Image.Resampling.LANCZOS)
    return _center_crop(grown, w, h)


def effect_shine(crop, wave, progress, **_kwargs):
    lit = ImageEnhance.Brightness(crop).enhance(1.0 + 0.14 * wave)
    w, h = crop.size
    result = _screen_color(lit, _shine_mask(w, h, progress, 0.18, 0.82), (255, 206, 96))
    result = _screen_color(result, _shine_mask(w, h, (progress + 0.38) % 1.0, 0.11, 0.40), (255, 236, 170))
    result = _screen_color(result, _leading_edge_mask(w, h, progress) * 0.85, (255, 255, 235))
    return result


def effect_glow(crop, wave, progress, **_kwargs):
    bright = ImageEnhance.Brightness(crop).enhance(1.0 + 0.28 * wave)
    return ImageEnhance.Contrast(bright).enhance(1.0 + 0.10 * wave)


def effect_gold_pulse(crop, wave, progress, **_kwargs):
    gold = Image.new('RGB', crop.size, (255, 198, 72))
    tinted = Image.blend(crop, gold, 0.08 + 0.16 * wave)
    return ImageEnhance.Brightness(tinted).enhance(1.0 + 0.12 * wave)


def effect_breathe(crop, wave, progress, **_kwargs):
    """
    Lighting-only pulse. Gold numbers and chrome brighten in place.

    Scaling or a wide blend mask made 18 / 90% / 60 look soft; this never
    resizes, so the type stays sharp.
    """
    arr = np.asarray(crop, dtype=np.float32)
    luma = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    highlight = np.clip((luma - 50.0) / 150.0, 0.0, 1.0)
    highlight *= highlight
    gain = 1.0 + 0.32 * wave * highlight
    out = np.clip(arr * gain[:, :, np.newaxis], 0, 255)
    gold = np.array([255.0, 214.0, 96.0], dtype=np.float32)
    tint = (0.14 * wave) * highlight[:, :, np.newaxis]
    out = np.clip(out * (1.0 - tint) + gold * tint, 0, 255)
    return Image.fromarray(out.astype(np.uint8), mode='RGB')


def effect_float(crop, wave, progress, **_kwargs):
    w, h = crop.size
    shift = int(round(5 * math.sin(2.0 * math.pi * progress)))
    lit = ImageEnhance.Brightness(crop).enhance(1.0 + 0.10 * wave)
    if shift == 0:
        return lit
    moved = Image.new('RGB', crop.size)
    if shift < 0:
        moved.paste(lit.crop((0, -shift, w, h)), (0, 0))
        moved.paste(lit.crop((0, 0, w, -shift)), (0, h + shift))
    else:
        moved.paste(lit.crop((0, 0, w, h - shift)), (0, shift))
        moved.paste(lit.crop((0, h - shift, w, h)), (0, 0))
    return _keep_frame(lit, moved, frame_frac=0.22)


def effect_zoom(crop, wave, progress, **_kwargs):
    """Zoom is applied as a full-box layer scale in compose_frames, not here."""
    return crop


def effect_sparkle(crop, wave, progress, sparkles=None, **_kwargs):
    result = ImageEnhance.Brightness(crop).enhance(1.0 + 0.08 * wave)
    arr = np.array(result).astype(np.float32)
    h, w = arr.shape[:2]
    for sx, sy, phase, radius in sparkles or ():
        twinkle = 0.5 * (1.0 + math.sin(2.0 * math.pi * (progress + phase)))
        if twinkle < 0.32:
            continue
        radius = max(2, min(radius, 5))
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        blob = np.clip(1.0 - np.sqrt(xx * xx + yy * yy) / max(radius, 1), 0, 1) ** 2
        x0, y0 = int(sx * w), int(sy * h)
        y1, y2 = max(0, y0 - radius), min(h, y0 + radius + 1)
        x1, x2 = max(0, x0 - radius), min(w, x0 + radius + 1)
        by1, by2 = y1 - (y0 - radius), y2 - (y0 - radius)
        bx1, bx2 = x1 - (x0 - radius), x2 - (x0 - radius)
        glow = blob[by1:by2, bx1:bx2, np.newaxis] * twinkle * 190
        arr[y1:y2, x1:x2] = np.clip(arr[y1:y2, x1:x2] + glow, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode='RGB')


def effect_fade(crop, wave, progress, **_kwargs):
    return ImageEnhance.Brightness(crop).enhance(0.86 + 0.26 * wave)


def effect_rim(crop, wave, progress, **_kwargs):
    w, h = crop.size
    yy, xx = np.mgrid[0:h, 0:w]
    nx = np.minimum(xx, w - 1 - xx) / max(w / 2.0, 1)
    ny = np.minimum(yy, h - 1 - yy) / max(h / 2.0, 1)
    dist = np.minimum(nx, ny)
    rim = np.clip(1.0 - np.abs(dist - 0.10) / 0.10, 0.0, 1.0) ** 2
    rim = rim * (0.35 + 0.55 * wave)
    lit = ImageEnhance.Brightness(crop).enhance(1.0 + 0.08 * wave)
    return _screen_color(lit, rim.astype(np.float32), (255, 214, 110))


def effect_flicker(crop, wave, progress, **_kwargs):
    shimmer = 0.5 + 0.5 * math.sin(progress * 2.0 * math.pi * 3.0)
    strength = 0.06 + 0.12 * wave * shimmer
    gold = Image.new('RGB', crop.size, (255, 214, 110))
    return Image.blend(crop, Image.blend(crop, gold, 0.36), strength)


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
    count = max(6, int(10 * (box_width * box_height) ** 0.5 / 90))
    points = []
    for _ in range(min(count, 14)):
        points.append((
            float(rng.uniform(0.18, 0.82)),
            float(rng.uniform(0.18, 0.82)),
            float(rng.uniform(0.0, 1.0)),
            int(rng.integers(2, 4)),
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
            label=item.get('label') or '',
        )
    return item


def _ordered_effects(effect_names):
    chosen = [name for name in (effect_names or []) if name in EFFECTS]
    if not chosen:
        chosen = list(DEFAULT_ANIMATION_TYPES)
    return [name for name in EFFECT_APPLY_ORDER if name in chosen]


MOTION_EFFECTS = {'zoom', 'float'}
SUBJECT_SKIP_EFFECTS = {'float', 'rim'}
LAYER_EFFECTS = {'zoom'}
ZOOM_GROW = 0.30


def apply_effects(crop, expanded, wave, progress, effect_names, sparkles, subject=False):
    ordered = [name for name in _ordered_effects(effect_names) if name not in LAYER_EFFECTS]
    if subject:
        ordered = [name for name in ordered if name not in SUBJECT_SKIP_EFFECTS]
        if not ordered:
            ordered = ['glow']

    result = crop
    for name in ordered:
        result = EFFECTS[name](
            result,
            wave,
            progress,
            sparkles=sparkles,
            expanded=expanded,
            box_size=crop.size,
            subject=subject,
        )
    return result


def _layer_zoom_scale(progress, effect_names):
    if 'zoom' not in _ordered_effects(effect_names):
        return 1.0
    # Triangle 0 → 1 → 0. Cosine easing sat near 1.0 for most of the loop.
    ping = 1.0 - abs(2.0 * progress - 1.0)
    return 1.0 + ZOOM_GROW * ping


def _scale_layer(patch, mask, scale):
    """Grow the whole box (chrome + content) together, centred on itself."""
    if scale <= 1.002:
        return patch, mask, 0, 0
    width, height = patch.size
    new_w = max(width + 1, int(round(width * scale)))
    new_h = max(height + 1, int(round(height * scale)))
    grown = patch.resize((new_w, new_h), Image.Resampling.LANCZOS)
    # Keep the mask binary so unzoomed pixels cannot show through and cancel the motion.
    grown_mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)
    return grown, grown_mask, (new_w - width) // 2, (new_h - height) // 2


def _paste_layer(frame, patch, xy, mask):
    """Paste a possibly oversized layer, clipping to the frame."""
    left, top = xy
    src_x = max(0, -left)
    src_y = max(0, -top)
    dst_x = max(0, left)
    dst_y = max(0, top)
    copy_w = min(patch.width - src_x, frame.width - dst_x)
    copy_h = min(patch.height - src_y, frame.height - dst_y)
    if copy_w <= 0 or copy_h <= 0:
        return
    frame.paste(
        patch.crop((src_x, src_y, src_x + copy_w, src_y + copy_h)),
        (dst_x, dst_y),
        mask.crop((src_x, src_y, src_x + copy_w, src_y + copy_h)),
    )


def compose_frames(image, detections, *, effects=None, occluders=None, frame_count=None):
    """
    Compose looping RGB frames with Pillow.

    Multiple effects are stacked. YOLO / person boxes use a silhouette mask
    so the effect follows the character instead of a rectangle with holes.
    """
    frame_count = frame_count or settings.GIF_FRAME_COUNT
    effect_names = effects or list(DEFAULT_ANIMATION_TYPES)
    occluders = list(occluders or [])
    zoom_selected = 'zoom' in _ordered_effects(effect_names)

    ordered_detections = sorted(
        detections,
        key=lambda item: PASTE_ORDER.get((_as_region(item).source or '').lower(), 6),
    )

    regions = []
    for detection in ordered_detections:
        detection = _as_region(detection)
        left, top, box_width, box_height = _pixel_box(detection, image.size)
        el, et, ew, eh, ox, oy = _expanded_box(left, top, box_width, box_height, image.size)
        expanded = image.crop((el, et, el + ew, et + eh)).convert('RGB')
        crop = expanded.crop((ox, oy, ox + box_width, oy + box_height))
        is_subject = _is_subject_region(detection)
        if is_subject:
            mask = _subject_mask(image, (left, top, box_width, box_height), occluders)
        elif zoom_selected:
            # Opaque rectangle so chrome at the edges zooms with the rest.
            mask = Image.new('L', (box_width, box_height), 255)
        else:
            mask = _region_mask(box_width, box_height, detection.source)
        regions.append({
            'crop': crop,
            'expanded': expanded,
            'left': left,
            'top': top,
            'mask': mask,
            'subject': is_subject,
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
                subject=region['subject'],
            )
            mask = region['mask']
            left, top = region['left'], region['top']
            scale = _layer_zoom_scale(progress, effect_names)
            animated, mask, dx, dy = _scale_layer(animated, mask, scale)
            _paste_layer(frame, animated, (left - dx, top - dy), mask)
        frames.append(frame)
    return frames


def render_gif_bytes(image, detections, *, effects=None, occluders=None, frame_count=None, duration_ms=None):
    """Compose frames then encode a looping GIF (FFmpeg palette, Pillow fallback)."""
    duration_ms = duration_ms or settings.GIF_DURATION_MS
    frames = compose_frames(
        image,
        detections,
        effects=effects,
        occluders=occluders,
        frame_count=frame_count,
    )
    gif_bytes, _video_bytes = encode_animation(frames, duration_ms)
    return gif_bytes, len(frames)


def generate_gif(job):
    """
    Render animation for an AnimationJob.

    Pillow composes frames. PyAV writes an MP4 preview; FFmpeg writes the GIF.
    """
    if not isinstance(job, AnimationJob):
        job = AnimationJob.objects.select_related('project').prefetch_related(
            'selected_objects',
            'project__detections',
        ).get(pk=job)

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
        duration_ms = settings.GIF_DURATION_MS
        frames = compose_frames(
            image,
            detections,
            effects=job.get_animation_types(),
            occluders=_collect_occluders(job, detections),
        )
        gif_bytes, video_bytes = encode_animation(frames, duration_ms)
        stem = f"{project.project_id or 'project'}_v{job.version}"
        if job.gif_file:
            job.gif_file.delete(save=False)
        if job.video_file:
            job.video_file.delete(save=False)
        job.gif_file.save(f'{stem}.gif', ContentFile(gif_bytes), save=False)
        if video_bytes:
            job.video_file.save(f'{stem}.mp4', ContentFile(video_bytes), save=False)
        else:
            job.video_file = None
        job.frame_count = len(frames)
        job.file_size = len(gif_bytes)
        job.status = STATUS_COMPLETED
        update_fields = ['gif_file', 'video_file', 'frame_count', 'file_size', 'status']
        job.save(update_fields=update_fields)
        logger.info(
            "Animation generated for %s v%s (%s, %s frames, gif=%s bytes, mp4=%s bytes)",
            project.project_id,
            job.version,
            ','.join(job.get_animation_types()),
            len(frames),
            len(gif_bytes),
            len(video_bytes) if video_bytes else 0,
        )
        return job
    except Exception:
        job.status = STATUS_FAILED
        job.save(update_fields=['status'])
        logger.exception("GIF generation failed for job %s", job.pk)
        raise
