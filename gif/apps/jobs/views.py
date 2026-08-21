import logging

from django.contrib import messages
from django.core.exceptions import ValidationError
from django.db import transaction
from django.shortcuts import redirect
from django.utils.decorators import method_decorator
from django.views import View
from django.views.generic import DetailView
from django.views.generic.detail import SingleObjectMixin

from apps.projects.models import Project

from .generation import generate_gif
from .models import AnimationJob
from .services import create_animation_job, parse_detection_ids, parse_manual_regions

logger = logging.getLogger(__name__)


@method_decorator(transaction.non_atomic_requests, name='dispatch')
class AnimationJobCreateView(SingleObjectMixin, View):
    model = Project
    http_method_names = ['post']

    def post(self, request, *args, **kwargs):
        project = self.get_object()

        try:
            detection_ids = parse_detection_ids(request.POST.get('detection_ids', ''))
            manual_regions = parse_manual_regions(request.POST.get('manual_regions', ''))
            job = create_animation_job(project, detection_ids, manual_regions)
        except ValidationError as exc:
            messages.error(request, ' '.join(exc.messages))
            return redirect('projects:project_detail', pk=project.pk)

        try:
            generate_gif(job)
        except Exception:
            logger.exception("GIF generation failed for job %s", job.pk)
            messages.error(request, "Job saved, but GIF generation failed. You can retry from the job page.")
        else:
            messages.success(request, f"Animation job v{job.version} is ready.")

        return redirect('jobs:job_detail', pk=job.pk)


class AnimationJobDetailView(DetailView):
    model = AnimationJob
    template_name = 'jobs/detail.html'
    context_object_name = 'job'

    def get_queryset(self):
        return (
            AnimationJob.objects
            .select_related('project')
            .prefetch_related('selected_objects')
        )


@method_decorator(transaction.non_atomic_requests, name='dispatch')
class AnimationJobGenerateView(SingleObjectMixin, View):
    model = AnimationJob
    http_method_names = ['post']

    def post(self, request, *args, **kwargs):
        job = self.get_object()
        try:
            generate_gif(job)
        except Exception:
            logger.exception("GIF generation failed for job %s", job.pk)
            messages.error(request, "GIF generation failed. Check the logs for details.")
        else:
            messages.success(request, f"GIF for v{job.version} is ready.")
        return redirect('jobs:job_detail', pk=job.pk)
