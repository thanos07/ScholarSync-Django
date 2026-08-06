from django.urls import path

from . import views

urlpatterns = [
    path("", views.demo_home, name="demo-home"),
    path("paper/<slug:document_id>/", views.demo_paper, name="demo-paper"),
    path("clear/", views.demo_clear, name="demo-clear"),
    path("export/pdf/", views.demo_export_pdf, name="demo-export-pdf"),
]
