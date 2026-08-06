from django.urls import path
from . import views
urlpatterns = [
    path("new/<uuid:workspace_id>/", views.conversation_new, name="conversation-new"),
    path("<uuid:conversation_id>/", views.conversation_detail, name="conversation-detail"),
    path("<uuid:conversation_id>/export/pdf/", views.export_pdf, name="conversation-export-pdf"),
]
