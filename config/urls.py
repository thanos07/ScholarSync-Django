from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("django.contrib.auth.urls")),
    path("", include("apps.core.urls")),
    path("demo/", include("apps.demo.urls")),
    path("app/", include("apps.workspaces.urls")),
    path("documents/", include("apps.documents.urls")),
    path("conversations/", include("apps.conversations.urls")),
    path("notes/", include("apps.notes.urls")),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
