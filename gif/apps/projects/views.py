import logging

from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import redirect
from django.urls import reverse
from django.views import View
from django.views.generic import CreateView, DetailView
from django.views.generic.detail import SingleObjectMixin

from .forms import ProjectUploadForm
from .models import Project
from .services import load_preprocessed_image, run_detection

logger = logging.getLogger(__name__)


def serialize_detection(detection):
    return {
        'id': detection.pk,
        'label': detection.label or detection.text_content or 'object',
        'confidence': detection.confidence,
        'source': detection.source,
        'text_content': detection.text_content,
        'x': detection.x,
        'y': detection.y,
        'width': detection.width,
        'height': detection.height,
    }


class ProjectUploadView(CreateView):
    model = Project
    form_class = ProjectUploadForm
    template_name = 'projects/upload.html'

    def get_success_url(self):
        return reverse('projects:project_detail', kwargs={'pk': self.object.pk})

    def get_context_data(self, **kwargs):
        context = super(ProjectUploadView, self).get_context_data(**kwargs)
        context['recent_projects'] = (
            Project.objects.exclude(image='').order_by('-created')[:6]
        )
        return context


class ProjectDetailView(DetailView):
    model = Project
    template_name = 'projects/detail.html'
    context_object_name = 'project'

    def get_context_data(self, **kwargs):
        context = super(ProjectDetailView, self).get_context_data(**kwargs)
        project = self.object

        preprocessed_size = None
        if project.image:
            preprocessed_size = load_preprocessed_image(project.image).size

        detections = list(project.detections.all())
        context['detections'] = detections
        context['detections_data'] = [serialize_detection(d) for d in detections]
        context['preprocessed_size'] = preprocessed_size
        context['animation_jobs'] = project.animations.all()[:8]
        return context


class ProjectDetectView(SingleObjectMixin, View):
    model = Project
    http_method_names = ['post']

    def post(self, request, *args, **kwargs):
        project = self.get_object()

        try:
            created = run_detection(project)
        except Exception:
            logger.exception("Detection failed for project %s", project.project_id)
            messages.error(request, "Detection failed. Check the logs for details.")
        else:
            messages.success(request, f"Detected {len(created)} object(s).")

        return redirect('projects:project_detail', pk=project.pk)


class ProjectDetectionsJsonView(DetailView):
    model = Project

    def render_to_response(self, context, **response_kwargs):
        project = self.object
        return JsonResponse({
            'project_id': project.project_id,
            'image_url': project.image.url if project.image else None,
            'detections': [serialize_detection(d) for d in project.detections.all()],
        })
