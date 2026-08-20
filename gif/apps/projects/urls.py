from django.urls import path

from . import views

app_name = 'projects'

urlpatterns = [
    path(r'', views.ProjectUploadView.as_view(), name='project_upload'),
    path(r'project/<int:pk>/', views.ProjectDetailView.as_view(), name='project_detail'),
    path(r'project/<int:pk>/detect/', views.ProjectDetectView.as_view(), name='project_detect'),
    path(
        r'project/<int:pk>/detections.json',
        views.ProjectDetectionsJsonView.as_view(),
        name='project_detections_json',
    ),
]
