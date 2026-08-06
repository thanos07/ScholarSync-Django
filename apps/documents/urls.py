from django.urls import path
from . import views
urlpatterns = [
    path("upload/<uuid:workspace_id>/", views.upload_document, name="document-upload"),
    path("<uuid:document_id>/view/", views.view_document, name="document-view"),
    path("<uuid:document_id>/delete/", views.delete_document, name="document-delete"),
]
