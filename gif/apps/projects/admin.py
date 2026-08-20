from django.contrib import admin
from .models import Project, DetectionObject
from django.utils.html import format_html

@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ["project_id", "image_thumbnail"]
    search_fields = ["project_id"]

    @admin.display(description="Image")
    def image_thumbnail(self, obj):
        thumb = obj.get_image_thumbnail()
        if thumb:
            return format_html(thumb)
        return "-"

@admin.register(DetectionObject)
class DetectionObjectAdmin(admin.ModelAdmin):
    list_display = ["label", "confidence", "source", "x", "y", "width", "height"]
