from django.contrib import admin
from .models import AnimationJob

@admin.register(AnimationJob)
class AnimationJobAdmin(admin.ModelAdmin):
    list_display = ["project__project_id", "version", "animation_types", "status"]
    readonly_fields = ["gif_file", "video_file"]
