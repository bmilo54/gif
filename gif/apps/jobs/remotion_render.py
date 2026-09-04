import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from django.conf import settings
from imageio_ffmpeg import get_ffmpeg_exe

logger = logging.getLogger(__name__)


def remotion_dir():
    configured = getattr(settings, 'REMOTION_DIR', None)
    if configured:
        return Path(configured)
    return Path(settings.BASE_DIR).parent / 'remotion'


def _npx_bin():
    found = shutil.which('npx') or shutil.which('npx.cmd')
    if found:
        return found
    for candidate in (
        Path(r'C:\Program Files\nodejs\npx.cmd'),
        Path(r'C:\Program Files\nodejs\npx'),
    ):
        if candidate.exists():
            return str(candidate)
    return None


def remotion_available():
    if _npx_bin() is None:
        return False
    project = remotion_dir()
    return (project / 'package.json').exists() and (project / 'node_modules' / 'remotion').exists()


def render_promo_video(
    poster_path,
    *,
    regions,
    width,
    height,
    fps,
    frame_count,
    output_mp4,
    characters=None,
    cutouts=None,
    depth_map_path=None,
):
    """
    Render the Promo composition with local Remotion (no AWS).

    Args:
        poster_path:    Absolute path to the background PNG (inpainted if SAM ran).
        regions:        List of UI region dicts (card, button, ocr, title ...).
        characters:     List of character dicts with keys:
                          src, index, bbox {x,y,width,height}, effects [...]
                        Each src is a basename like 'char_0.png' that must sit
                        next to poster_path in the same tmp directory.
        depth_map_path: Absolute path to depth.png, or None.
        width/height:   Canvas dimensions in pixels.
        fps:            Frames per second.
        frame_count:    Total frames to render.
        output_mp4:     Destination path for the rendered MP4.
    """
    if not remotion_available():
        raise RuntimeError(
            'Remotion is not available. Install Node.js and run npm install in the remotion/ folder.'
        )

    project = remotion_dir()
    output_mp4 = Path(output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    # ── Copy assets into Remotion public/ ─────────────────────────────────────
    public_id = f'renders/{uuid.uuid4().hex}'
    public_dir = project / 'public' / public_id.replace('/', os.sep)
    public_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(poster_path, public_dir / 'poster.png')

    characters_payload = []
    for char in (characters or []):
        src_name = char.get('src', '')
        src_path = Path(poster_path).parent / src_name
        if src_path.exists():
            shutil.copyfile(src_path, public_dir / src_name)
            characters_payload.append({
                'src': f'{public_id}/{src_name}',
                'index': char.get('index', 0),
                'bbox': char.get('bbox', {}),
                'effects': char.get('effects', []),
                'color': char.get('color'),
            })
        else:
            logger.warning('Character asset not found, skipping: %s', src_path)

    cutouts_payload = []
    for item in (cutouts or []):
        src_name = item.get('src', '')
        src_path = Path(poster_path).parent / src_name
        if not src_path.exists():
            logger.warning('UI cut-out asset not found, skipping: %s', src_path)
            continue
        shutil.copyfile(src_path, public_dir / src_name)
        cutouts_payload.append({
            'src': f'{public_id}/{src_name}',
            'index': item.get('index', 0),
            'bbox': item.get('bbox', {}),
            'effects': item.get('effects', []),
            'color': item.get('color'),
            'source': item.get('source') or 'card',
            'label': item.get('label') or '',
        })

    # Depth map
    depth_map_prop = None
    if depth_map_path and Path(depth_map_path).exists():
        shutil.copyfile(depth_map_path, public_dir / 'depth.png')
        depth_map_prop = f'{public_id}/depth.png'

    # ── Build props JSON ───────────────────────────────────────────────────────
    props = {
        'poster': f'{public_id}/poster.png',
        'depthMap': depth_map_prop,
        'regions': list(regions or []),
        'characters': characters_payload,
        'cutouts': cutouts_payload,
        'width': int(width),
        'height': int(height),
        'fps': int(round(fps)),
        'durationInFrames': int(frame_count),
    }

    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8') as handle:
        json.dump(props, handle)
        props_path = handle.name

    npx = _npx_bin()
    command = [
        npx,
        '--yes',
        'remotion',
        'render',
        'src/index.js',
        'Promo',
        str(output_mp4),
        f'--props={props_path}',
    ]
    env = os.environ.copy()
    env['FFMPEG_PATH'] = get_ffmpeg_exe()
    env['REMOTION_FFMPEG_PATH'] = env['FFMPEG_PATH']
    logger.info('Rendering Remotion composition %s', command)
    try:
        result = subprocess.run(
            command,
            cwd=str(project),
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=False,
        )
    finally:
        try:
            os.remove(props_path)
        except OSError:
            pass
        shutil.rmtree(public_dir, ignore_errors=True)

    if result.returncode != 0 or not output_mp4.exists():
        stderr = (result.stderr or '')[-4000:]
        stdout = (result.stdout or '')[-2000:]
        raise RuntimeError(f'Remotion render failed: {stderr or stdout}')

    logger.info('Remotion wrote %s (%s bytes)', output_mp4, output_mp4.stat().st_size)
    return output_mp4
