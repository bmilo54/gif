from PIL import Image, ImageOps

# Detection runs on a bounded copy so large phone photos do not blow up
# inference time. Bounding boxes are stored normalised (0-1), so downscaling
# here never invalidates coordinates saved against the original upload.
MAX_SIDE = 1280


def load_preprocessed_image(image_field, max_side=MAX_SIDE):
    """
    Open a stored upload and normalise it for detection.

    Applies EXIF rotation, converts to RGB and downscales so the longest side
    is at most `max_side`. Returns an in-memory PIL image; the stored original
    is left untouched.
    """
    with image_field.open('rb') as f:
        image = Image.open(f)
        image.load()

    image = ImageOps.exif_transpose(image)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    width, height = image.size
    longest = max(width, height)
    if longest > max_side:
        ratio = max_side / float(longest)
        image = image.resize(
            (max(1, int(width * ratio)), max(1, int(height * ratio))),
            Image.Resampling.LANCZOS,
        )

    return image
