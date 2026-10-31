from django.urls import path
from . import views


urlpatterns = [
    path("", views.home, name="home"),
    path("analysis/", views.analysis, name="analysis"),
    path("download-pdf/<str:filename>/", views.download_pdf, name="download_pdf"),
]
