import uuid
from django.conf import settings
from django.db import models
class Note(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="notes")
    workspace = models.ForeignKey("workspaces.Workspace", on_delete=models.CASCADE, related_name="notes")
    document = models.ForeignKey("documents.Document", on_delete=models.SET_NULL, null=True, blank=True, related_name="notes")
    page_number = models.PositiveIntegerField(null=True, blank=True)
    title = models.CharField(max_length=180)
    content = models.TextField()
    tags = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
