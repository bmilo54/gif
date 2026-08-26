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
from .services import (
    create_animation_job,
    parse_animation_types,
    parse_detection_ids,
    parse_manual_regions,
    parse_regions,
    save_job_adjustments,
)

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
            animation_types = parse_animation_types(request.POST.getlist('animation_types'))
            job = create_animation_job(project, detection_ids, manual_regions, animation_types)
        except ValidationError as exc:
            messages.error(request, ' '.join(exc.messages))
            return redirect('projects:project_detail', pk=project.pk)

        return redirect('jobs:job_adjust', pk=job.pk)


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

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['job_regions'] = self.object.get_regions()
        return context


class AnimationJobAdjustView(DetailView):
    model = AnimationJob
    template_name = 'jobs/adjust.html'
    context_object_name = 'job'

    def get_queryset(self):
        return AnimationJob.objects.select_related('project')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Effects are now per-region and rendered by adjust.js;
        # we only need to pass the regions data to seed the canvas.
        context['regions_data'] = self.object.get_regions()
        return context


@method_decorator(transaction.non_atomic_requests, name='dispatch')
class AnimationJobGenerateView(SingleObjectMixin, View):
    model = AnimationJob
    http_method_names = ['post']

    def post(self, request, *args, **kwargs):
        job = self.get_object()
        try:
            regions = parse_regions(request.POST.get('regions', '')) if request.POST.get('regions') else job.get_regions()
            # Effects are embedded inside each region dict; animation_types is
            # kept on the job only as a historical record (pass empty list).
            save_job_adjustments(job, regions, [])
            generate_gif(job)
        except ValidationError as exc:
            messages.error(request, ' '.join(exc.messages))
            return redirect('jobs:job_adjust', pk=job.pk)
        except Exception:
            logger.exception("GIF generation failed for job %s", job.pk)
            messages.error(request, "GIF generation failed. Check the logs for details.")
            return redirect('jobs:job_adjust', pk=job.pk)

        messages.success(request, f"GIF for v{job.version} is ready.")
        return redirect('jobs:job_detail', pk=job.pk)
