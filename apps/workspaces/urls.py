from django.urls import path
from . import views
urlpatterns = [
    path("", views.workspace_list, name="workspace-list"),
    path("new/", views.workspace_create, name="workspace-create"),
    path("<uuid:workspace_id>/", views.workspace_detail, name="workspace-detail"),
]
