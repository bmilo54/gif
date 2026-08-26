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
    effects,
    regions,
    width,
    height,
    fps,
    frame_count,
    output_mp4,
    person_path=None,
    person_region=None,
):
    """
    Render the Promo composition with local Remotion (no AWS).

    Returns the output MP4 path.
    """
    if not remotion_available():
        raise RuntimeError(
            'Remotion is not available. Install Node.js and run npm install in the remotion/ folder.'
        )

    project = remotion_dir()
    output_mp4 = Path(output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    public_id = f'renders/{uuid.uuid4().hex}'
    public_dir = project / 'public' / public_id.replace('/', os.sep)
    public_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(poster_path, public_dir / 'poster.png')
    person_prop = ''
    if person_path and os.path.exists(person_path):
        shutil.copyfile(person_path, public_dir / 'person.mp4')
        person_prop = f'{public_id}/person.mp4'

    props = {
        'poster': f'{public_id}/poster.png',
        'person': person_prop,
        'personRegion': person_region or {},
        'effects': list(effects or []),
        'regions': list(regions or []),
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
