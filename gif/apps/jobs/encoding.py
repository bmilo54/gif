import io
import logging
import os
import shutil
import subprocess
import tempfile
from fractions import Fraction

import av
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

logger = logging.getLogger(__name__)

PALETTE_FILTER = (
    'split[s0][s1];'
    '[s0]palettegen=stats_mode=diff:max_colors=256[p];'
    '[s1][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle'
)


def frames_to_arrays(frames):
    """Convert Pillow frames to RGB numpy arrays for imageio/PyAV."""
    return [np.asarray(frame.convert('RGB'), dtype=np.uint8) for frame in frames]


def _even_frame(arr):
    """H.264 yuv420p needs even width and height."""
    height, width = arr.shape[:2]
    even_h = height + (height % 2)
    even_w = width + (width % 2)
    if even_h == height and even_w == width:
        return arr
    padded = np.empty((even_h, even_w, 3), dtype=np.uint8)
    padded[:height, :width] = arr
    if even_w > width:
        padded[:height, width:] = arr[:height, width - 1:width]
    if even_h > height:
        padded[height:, :width] = arr[height - 1:height, :width]
        if even_w > width:
            padded[height:, width:] = arr[height - 1, width - 1]
    return padded


def encode_mp4_bytes(arrays, fps):
    """Encode a looping H.264 MP4 with PyAV so colour pulses stay intact."""
    if not arrays:
        raise ValueError('No frames to encode.')
    even = [_even_frame(arr) for arr in arrays]
    height, width = even[0].shape[:2]
    rate = fps if isinstance(fps, Fraction) else Fraction(fps).limit_denominator(1000)
    tmpdir = tempfile.mkdtemp(prefix='gif-mp4-')
    path = os.path.join(tmpdir, 'preview.mp4')
    try:
        container = av.open(path, mode='w', format='mp4', options={'movflags': 'faststart'})
        stream = container.add_stream('libx264', rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        stream.options = {
            'preset': 'veryfast',
            'crf': '17',
            'tune': 'animation',
            'profile': 'high',
        }
        for arr in even:
            video_frame = av.VideoFrame.from_ndarray(arr, format='rgb24')
            video_frame = video_frame.reformat(format='yuv420p')
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        with open(path, 'rb') as handle:
            return handle.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


def encode_gif_bytes(arrays, fps):
    """
    Encode a looping GIF with FFmpeg's two-pass palette (via imageio-ffmpeg).

    This keeps more of the composed colour than Pillow's single-frame quantize.
    """
    if not arrays:
        raise ValueError('No frames to encode.')
    height, width = arrays[0].shape[:2]
    ffmpeg = get_ffmpeg_exe()
    tmpdir = tempfile.mkdtemp(prefix='gif-enc-')
    path = os.path.join(tmpdir, 'out.gif')
    command = [
        ffmpeg,
        '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', 'pipe:0',
        '-vf', PALETTE_FILTER,
        '-loop', '0',
        path,
    ]
    payload = b''.join(arr.tobytes() for arr in arrays)
    try:
        result = subprocess.run(
            command,
            input=payload,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not os.path.exists(path) or os.path.getsize(path) == 0:
            stderr = (result.stderr or b'').decode('utf-8', errors='replace')[-1500:]
            raise RuntimeError(f'FFmpeg GIF encode failed: {stderr}')
        with open(path, 'rb') as handle:
            return handle.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


def encode_gif_from_video(video_path, fps):
    """Make a looping GIF from a rendered MP4 using the FFmpeg palette."""
    ffmpeg = get_ffmpeg_exe()
    tmpdir = tempfile.mkdtemp(prefix='gif-from-mp4-')
    path = os.path.join(tmpdir, 'out.gif')
    command = [
        ffmpeg,
        '-y',
        '-i', str(video_path),
        '-vf', f'fps={fps},{PALETTE_FILTER}',
        '-loop', '0',
        path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode != 0 or not os.path.exists(path) or os.path.getsize(path) == 0:
            stderr = (result.stderr or b'').decode('utf-8', errors='replace')[-1500:]
            raise RuntimeError(f'FFmpeg GIF from video failed: {stderr}')
        with open(path, 'rb') as handle:
            return handle.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


def encode_gif_pillow(frames, duration_ms):
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
    return buffer.getvalue()


def encode_animation(frames, duration_ms):
    """
    Encode Pillow-composed frames to GIF + MP4.

    Returns (gif_bytes, mp4_bytes_or_none).
    """
    if not frames:
        raise ValueError('No frames to encode.')
    fps = Fraction(1000, int(duration_ms))
    arrays = frames_to_arrays(frames)

    try:
        gif_bytes = encode_gif_bytes(arrays, float(fps))
    except Exception:
        logger.exception('FFmpeg palette GIF failed; falling back to Pillow.')
        gif_bytes = encode_gif_pillow(frames, duration_ms)

    mp4_bytes = None
    try:
        mp4_bytes = encode_mp4_bytes(arrays, fps)
    except Exception:
        logger.exception('PyAV MP4 encode failed; preview will use the GIF.')

    return gif_bytes, mp4_bytes


def compress_gif_bytes(gif_bytes, *, max_colors=64, fps=12):
    """Same width/height, much smaller file: drop fps + fewer colours.

    Photoreal 720 GIF at 30fps stays ~20–30MB even after a mild re-encode.
    Keep the pixel size; thin the timeline and palette so download is small.
    """
    if not gif_bytes:
        raise ValueError('No GIF to compress.')
    tmpdir = tempfile.mkdtemp(prefix='gif-compress-')
    src = os.path.join(tmpdir, 'in.gif')
    dest = os.path.join(tmpdir, 'out.gif')
    opt = os.path.join(tmpdir, 'opt.gif')
    try:
        with open(src, 'wb') as handle:
            handle.write(gif_bytes)
        vf = (
            f'fps={int(fps)},'
            f'split[s0][s1];'
            f'[s0]palettegen=stats_mode=diff:max_colors={int(max_colors)}[p];'
            '[s1][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle'
        )
        result = subprocess.run(
            [
                get_ffmpeg_exe(), '-y', '-i', src,
                '-vf', vf,
                '-loop', '0',
                dest,
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not os.path.exists(dest) or os.path.getsize(dest) == 0:
            stderr = (result.stderr or b'').decode('utf-8', errors='replace')[-1500:]
            raise RuntimeError(f'GIF compress failed: {stderr}')
        out_path = dest
        gifsicle = shutil.which('gifsicle') or shutil.which('gifsicle.exe')
        if gifsicle:
            extra = subprocess.run(
                [
                    gifsicle, '-O3', '--lossy=80',
                    '--no-warnings', '-o', opt, dest,
                ],
                capture_output=True,
                check=False,
            )
            if extra.returncode == 0 and os.path.exists(opt) and os.path.getsize(opt) > 0:
                out_path = opt
        with open(out_path, 'rb') as handle:
            out = handle.read()
        if len(out) >= len(gif_bytes):
            logger.info('Compress did not shrink GIF (%s bytes), keeping original', len(gif_bytes))
            return gif_bytes
        logger.info('Compressed GIF %s → %s bytes (same dimensions)', len(gif_bytes), len(out))
        return out
    finally:
        for path in (src, dest, opt):
            try:
                os.remove(path)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
