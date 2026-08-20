# -*- coding: utf-8 -*-

import base64
import io
import mimetypes
import string
import random
import os
import re
import nh3

from bs4 import BeautifulSoup

from django.db import models
from django.core import validators

from django.conf import settings
from django.contrib.staticfiles import finders
from django.contrib.sessions.models import Session
from django.utils.timezone import now
from django.http import JsonResponse
from django.utils.http import url_has_allowed_host_and_scheme
from django.shortcuts import resolve_url
from django.utils.safestring import mark_safe
from django.core.exceptions import ValidationError

from PIL import Image, UnidentifiedImageError
from sorl.thumbnail.shortcuts import get_thumbnail

try:
    from hashlib import sha1 as sha_constructor
except ImportError:
    from django.utils.hashcompat import sha_constructor


DISPLAY_EMPTY_VALUE = "---"


def validate_password(value):
    if len(value) < 8:
        raise ValidationError("Password must be at least 8 characters long.")
    if not re.search(r"[a-z]", value):
        raise ValidationError("Password must contain at least one lowercase letter.")
    if not re.search(r"[A-Z]", value):
        raise ValidationError("Password must contain at least one uppercase letter.")
    if not re.search(r"[0-9]", value):
        raise ValidationError("Password must contain at least one number.")
    if not re.search(r"[^a-zA-Z0-9]", value):  # Includes all non-alphanumeric characters
        raise ValidationError("Password must contain at least one special character.")


def generate_sha1(string, salt=None):
    if not salt:
        salt = sha_constructor(str(random.random()).encode('utf-8')).hexdigest()[:5]
    hash_key = sha_constructor("{0}{1}".format(str(salt), str(string)).encode('utf-8')).hexdigest()

    return (salt, hash_key)


def safe_referrer(request, default):
    """
    Takes the request and a default URL. Returns HTTP_REFERER if it's safe
    to use and set, and the default URL otherwise.

    The default URL can be a model with get_absolute_url defined, a urlname
    or a regular URL
    """
    referrer = request.META.get('HTTP_REFERER')
    if referrer and url_has_allowed_host_and_scheme(referrer, request.get_host()):
        return referrer
    if default:
        # Try to resolve. Can take a model instance, Django URL name or URL.
        return resolve_url(default)
    else:
        # Allow passing in '' and None as default
        return default


def random_string_generator(size, additional=None, chars=string.ascii_uppercase + string.digits + string.ascii_lowercase):
    return ''.join(random.choice(chars + str(additional)) for _ in range(size))


def random_string_generator_v2(size, additional=None):
    # Build allowed characters
    exclude = {'O', 'o', 'l', 'I'}
    chars = ''.join(c for c in (string.ascii_uppercase + string.ascii_lowercase + string.digits) if c not in exclude)

    # Add any extra characters
    if additional:
        chars += str(additional)

    return ''.join(random.choice(chars) for _ in range(size))


def get_protocol():
    """
    Returns a string with the current protocol.

    This can be either 'http' or 'https' depending on setting.
    """
    protocol = 'http'
    if settings.USE_HTTPS:
        protocol = 'https'
    return protocol


def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


def get_img_extension(img):
    """Return ext based on given image."""
    ext = 'JPEG'
    try:
        aux_ext = str(img).split('.')
        if aux_ext[len(aux_ext) - 1].lower() == 'png':
            ext = 'PNG'
        elif aux_ext[len(aux_ext) - 1].lower() == 'gif':
            ext = 'GIF'
    except Exception:  # pragma: no cover
        pass

    return ext


def _image_bytes_to_pdf_data_uri(raw, *, name_for_mime, max_side=800, jpeg_quality=85):
    if not raw:
        return None
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except (UnidentifiedImageError, OSError, ValueError):
        mime, _ = mimetypes.guess_type(name_for_mime or "")
        if not mime or not mime.startswith("image/"):
            return None
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

    if im.mode == "CMYK":
        im = im.convert("RGB")
    elif im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
    if im.mode == "RGBA":
        background = Image.new("RGB", im.size, (255, 255, 255))
        background.paste(im, mask=im.split()[3])
        im = background
    elif im.mode != "RGB":
        im = im.convert("RGB")

    w, h = im.size
    if max(w, h) > max_side:
        ratio = max_side / float(max(w, h))
        im = im.resize(
            (max(1, int(w * ratio)), max(1, int(h * ratio))),
            Image.Resampling.LANCZOS,
        )
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def image_field_to_pdf_data_uri(field_file, *, max_side=800, jpeg_quality=85):
    """
    Build a data URI for embedding in HTML→PDF (e.g. WeasyPrint).

    Reads from Django file storage so PDF generation does not depend on HTTP
    fetches to MEDIA_URL (which often break behind proxies or for self-URLs).
    Downscales and JPEG-encodes to keep PDF size reasonable.
    """
    if not field_file:
        return None
    try:
        with field_file.open("rb") as f:
            raw = f.read()
    except OSError:
        return None
    return _image_bytes_to_pdf_data_uri(
        raw,
        name_for_mime=field_file.name,
        max_side=max_side,
        jpeg_quality=jpeg_quality,
    )


def static_relative_path_to_pdf_data_uri(relative_static_path, *, max_side=800, jpeg_quality=85):
    """
    Same as image_field_to_pdf_data_uri but for a path discoverable by Django
    staticfiles finders (e.g. 'view_exhibitors/assets/img/placeholder/800x800.png').
    """
    fs_path = finders.find(relative_static_path)
    if not fs_path:
        return None
    try:
        with open(fs_path, "rb") as f:
            raw = f.read()
    except OSError:
        return None
    return _image_bytes_to_pdf_data_uri(
        raw,
        name_for_mime=fs_path,
        max_side=max_side,
        jpeg_quality=jpeg_quality,
    )


def generate_thumbnail(img, img_size='x36'):
    """
    Generate image thumbnail based on given image.

    Return mark_safe string.
    """
    if img and hasattr(img, 'url'):
        ext = get_img_extension(img)
        thumb = get_thumbnail(img, img_size, upscale=False, format=ext)
        filename = os.path.basename(img.name)

        return mark_safe(
            f'<a href="{img.url}" data-fancybox="gallery" data-caption="{filename}" style="display: inline-block;">'
            f'<img width="{thumb.width}" height="{thumb.height}" src="{thumb.url}" '
            'style="border: 1px solid #CCC; padding: 2px;" />'
            "</a>"
        )
    else:
        return DISPLAY_EMPTY_VALUE


def running_number_generator(latest_code, prefix_length=None):
    running_number = latest_code
    number = running_number

    new_number = int(number) + 1
    if new_number < 100000:
        alphabet = running_number[:0]
    else:
        ascii_code = ord(running_number[:1])  # get alphabet int value
        alphabet = chr(ascii_code + 1)  # convert int value to alphabet
        new_number = 0
    zeros = len(str(new_number))

    for x in range(6 - zeros):  # loop for numbers of zeros
        alphabet += "0"

    new_code = alphabet + str(int(number) + 1)  # new running number

    return new_code


def running_number_generator_v2(latest_code):
    """
    Generate next running number with alphabetic prefix rollover.
    Examples: 
        001 → 002
        999 → A001 (rollover: add prefix A)
        A001 → A002
        A999 → B001 (rollover: increment prefix to B)
    """
    # Find where the number starts
    prefix = ''
    for i, char in enumerate(latest_code):
        if char.isdigit():
            prefix = latest_code[:i]
            number = int(latest_code[i:])
            break
    
    # Increment number
    new_number = number + 1
    
    # Check for rollover (number would become 1000)
    if new_number > 999:
        if prefix == '':
            # No prefix yet, add 'A'
            prefix = 'A'
        else:
            # Increment last character of prefix
            prefix = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        new_number = 1  # Reset to 001
    
    # Format with leading zeros (always 3 digits)
    return prefix + str(new_number).zfill(3)


def validate_pdf(value):
    if not value.name.endswith('.pdf'):
        raise ValidationError('Only PDF files are allowed.')


def validate_image(value):
    if not value.name.endswith(('.jpg', '.jpeg', '.png')):
        raise ValidationError('Only image files are allowed.')


def validate_image_pdf(value):
    if not value.name.endswith(('.jpg', '.jpeg', '.png', '.pdf')):
        raise ValidationError('Only JPEG, PNG, and PDF files are allowed.')  

    if value.size > 1 * 1024 * 1024:
        raise ValidationError('File size must be less than 1MB.') 


class JSONResponseMixin(object):
    """
    A mixin that can be used to render a JSON response.
    """

    def render_to_json_response(self, context, **response_kwargs):
        """
        Returns a JSON response, transforming 'context' to make the payload.
        """
        return JsonResponse(
            self.get_data(context),
            **response_kwargs
        )

    def get_data(self, context):
        """
        Returns an object that will be serialized as JSON by json.dumps().
        """
        # Note: This is *EXTREMELY* naive; in reality, you'll need
        # to do much more complex handling to ensure that arbitrary
        # objects -- such as Django model instances or querysets
        # -- can be serialized as JSON.
        return context


def logout_user_everywhere(user):
    """
    Deletes all active sessions for the given user.
    """
    # Get only active (non-expired) sessions
    active_sessions = Session.objects.filter(expire_date__gte=now())

    # Bulk scan
    for session in active_sessions:
        data = session.get_decoded()
        if str(user.id) == str(data.get('_auth_user_id')):
            session.delete()


# Helper functions for CKEditor to email WYSIWYG conversion
def merge_styles(existing: str, new_styles: dict) -> str:
    """Merge CSS styles, overriding existing ones."""
    style_dict = {}
    for item in existing.split(";"):
        if ":" in item:
            key, value = item.split(":", 1)
            style_dict[key.strip()] = value.strip()
    for key, value in new_styles.items():
        style_dict[key] = value
    return ";".join(f"{k}:{v}" for k, v in style_dict.items() if v) + ";"


# Helper functions to read styles safely
def get_style_property(cell, prop, ckeditor_attrs=None, html_attrs=None, default=None):
    """Get a style property from inline style, CKEditor attribute, or HTML attribute."""
    # 1️⃣ Inline style
    style = cell.get("style", "")
    style_dict = {k.strip(): v.strip() for k, v in (item.split(":", 1) for item in style.split(";") if ":" in item)}
    if prop in style_dict:
        return style_dict[prop]

    # 2️⃣ CKEditor attributes
    if ckeditor_attrs:
        for attr in ckeditor_attrs:
            if cell.get(attr):
                return cell[attr]

    # 3️⃣ HTML attributes
    if html_attrs:
        for attr in html_attrs:
            if cell.get(attr):
                return cell[attr]

    # 4️⃣ fallback default
    return default