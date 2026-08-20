from django import forms
from django.template.defaultfilters import filesizeformat

from .models import Project

MAX_IMAGE_SIZE = 10 * 1024 * 1024
ALLOWED_FORMATS = ('JPEG', 'PNG', 'WEBP')


class ProjectUploadForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ['image']
        widgets = {
            'image': forms.ClearableFileInput(attrs={
                'accept': 'image/jpeg,image/png,image/webp',
                'class': 'file-input',
            }),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['image'].required = True
        self.fields['image'].label = "Source image"

    def clean_image(self):
        image = self.cleaned_data.get('image')
        if not image:
            return image

        if image.size > MAX_IMAGE_SIZE:
            raise forms.ValidationError(
                f"Image must be smaller than {filesizeformat(MAX_IMAGE_SIZE)}."
            )

        # forms.ImageField has already verified the file with Pillow and
        # attached the parsed image, so the real format is trustworthy here
        # even if the extension lies.
        image_format = getattr(getattr(image, 'image', None), 'format', None)
        if image_format and image_format not in ALLOWED_FORMATS:
            allowed = ', '.join(ALLOWED_FORMATS)
            raise forms.ValidationError(f"Unsupported image format. Allowed: {allowed}.")

        return image
