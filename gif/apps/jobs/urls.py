from django.urls import path

from . import views

app_name = 'jobs'

urlpatterns = [
    path(
        r'project/<int:pk>/animate/',
        views.AnimationJobCreateView.as_view(),
        name='job_create',
    ),
    path(r'jobs/<int:pk>/', views.AnimationJobDetailView.as_view(), name='job_detail'),
    path(r'jobs/<int:pk>/adjust/', views.AnimationJobAdjustView.as_view(), name='job_adjust'),
    path(
        r'jobs/<int:pk>/generate/',
        views.AnimationJobGenerateView.as_view(),
        name='job_generate',
    ),
]
